#!/usr/bin/env python3
"""Colab Panel — 轻量 Google Colab Web 管理面板（基于 google-colab-cli）。

所有管理能力包装 colab CLI 子进程；OAuth 远程复制粘贴流程直接调
google_auth_oauthlib（与 colab_cli.auth 同一机制）。

借鉴 EasyColab 的增强：
- 闲置看门狗（app.py _idle_watchdog_loop）：可配置超时自动释放实例
- CU 燃烧速率估算（app.py estimate_cu_burn_rate）：GPU 型号 → CU/h + 会话已花费
- 实例就绪探测（colab_bridge.py _wait_for_runtime_ready）：区分 404 与初始化中
- 孤儿实例纳管（colab_bridge.py cmd_wake）：云端分配直接接管为本地会话
- 保活心跳（colab_bridge.py cmd_keepalive）：刷 runtime proxy token + 内核脉冲
- headless Drive 挂载（colab_bridge.py _build_rclone_drive_conf）：rclone 免浏览器授权
"""
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from urllib.parse import quote

import requests
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse

# ---------------------------------------------------------------------------
# API_TOKEN：未设置时自动生成随机口令并落盘复用（容器重启不变）
# ---------------------------------------------------------------------------
API_TOKEN = os.environ.get("API_TOKEN", "").strip()
TOKEN_FILE = os.path.expanduser("~/.config/colab-cli/panel_api_token")
if not API_TOKEN:
    # 优先读上次自动生成的
    try:
        with open(TOKEN_FILE, "r", encoding="utf-8") as f:
            API_TOKEN = f.read().strip()
    except Exception:
        API_TOKEN = ""
    if not API_TOKEN:
        import secrets
        API_TOKEN = secrets.token_urlsafe(24)
        os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
        with open(TOKEN_FILE, "w", encoding="utf-8") as f:
            f.write(API_TOKEN)
        os.chmod(TOKEN_FILE, 0o600)
        print(f"API_TOKEN 未设置，已自动生成并保存到 {TOKEN_FILE}")
    else:
        print(f"API_TOKEN 未设置，复用已保存的口令（{TOKEN_FILE}）")

CLI_TIMEOUT_DEFAULT = 120
CLI_TIMEOUT_LONG = 300  # new / exec / install / drivemount
CLI_BIN = shutil.which("colab") or "colab"

TOKEN_PATH = os.path.expanduser("~/.config/colab-cli/token.json")
OAUTH_CFG_PATH = os.path.expanduser("~/.colab-cli-oauth-config.json")

# 闲置看门狗默认超时（分钟），0 = 禁用；可经 API 调整
IDLE_TIMEOUT_MINUTES = int(os.environ.get("IDLE_TIMEOUT_MINUTES", "30"))
# 保活心跳间隔（秒），0 = 禁用
HEARTBEAT_INTERVAL_SECONDS = int(os.environ.get("HEARTBEAT_INTERVAL_SECONDS", "240"))

app = FastAPI(title="Colab Panel", docs_url=None, redoc_url=None)

# ---------------------------------------------------------------------------
# 鉴权
# ---------------------------------------------------------------------------
def check_auth(request: Request):
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer ") and auth[7:].strip() == API_TOKEN:
        return
    q = request.query_params.get("token", "")
    if q == API_TOKEN:
        return
    raise HTTPException(status_code=401, detail="unauthorized")


# ---------------------------------------------------------------------------
# CLI 封装
# ---------------------------------------------------------------------------
def run_cli(args, timeout=CLI_TIMEOUT_DEFAULT, input_text=None):
    """运行 colab 子命令，返回 (returncode, stdout, stderr)。"""
    cmd = [CLI_BIN] + args
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            input=input_text, env={**os.environ, "TERM": "dumb", "NO_COLOR": "1"},
        )
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"命令超时 ({timeout}s): {' '.join(args)}"
    except FileNotFoundError:
        return 127, "", "colab CLI 未找到"


def cli_json(args, timeout=CLI_TIMEOUT_DEFAULT):
    """运行子命令并解析 stdout JSON；失败抛 HTTPException 502。"""
    rc, out, err = run_cli(args, timeout=timeout)
    if rc != 0:
        raise HTTPException(status_code=502, detail=f"colab {' '.join(args)} 失败: {(err or out).strip()[-500:]}")
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail=f"无法解析 colab 输出为 JSON: {out.strip()[-500:]}")


# ---------------------------------------------------------------------------
# OAuth（远程复制粘贴流程，与 colab_cli/auth.py 相同机制）
# ---------------------------------------------------------------------------
_pkce_lock = threading.Lock()
_flow_holder = {}

def _load_client_config() -> dict:
    if os.path.exists(OAUTH_CFG_PATH):
        with open(OAUTH_CFG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    from importlib import resources
    res = resources.files("colab_cli").joinpath("oauth_config.json")
    return json.loads(res.read_text(encoding="utf-8"))


def build_auth_url() -> str:
    from google_auth_oauthlib.flow import InstalledAppFlow
    from colab_cli.auth import PUBLIC_SCOPES, REMOTE_REDIRECT_URI
    with _pkce_lock:
        flow = InstalledAppFlow.from_client_config(_load_client_config(), PUBLIC_SCOPES)
        flow.redirect_uri = REMOTE_REDIRECT_URI
        auth_url, _ = flow.authorization_url(prompt="consent", token_usage="remote")
        _flow_holder["flow"] = flow
    return auth_url


def exchange_auth_code(code: str) -> dict:
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request as GARequest
    from colab_cli.auth import PUBLIC_SCOPES, REMOTE_REDIRECT_URI, TOKEN_CONFIG_PATH
    with _pkce_lock:
        flow = _flow_holder.get("flow")
        if flow is None:
            flow = InstalledAppFlow.from_client_config(_load_client_config(), PUBLIC_SCOPES)
            flow.redirect_uri = REMOTE_REDIRECT_URI
        try:
            flow.fetch_token(code=code.strip())
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"授权码交换失败: {e}")
        creds = flow.credentials
        _flow_holder.pop("flow", None)
    os.makedirs(os.path.dirname(TOKEN_PATH), exist_ok=True)
    with open(TOKEN_PATH, "w", encoding="utf-8") as f:
        f.write(creds.to_json())
    return {"ok": True, "logged_in": bool(creds.valid or creds.refresh_token)}


def has_valid_token() -> bool:
    if not os.path.exists(TOKEN_PATH):
        return False
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request as GARequest
        from colab_cli.auth import PUBLIC_SCOPES
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, PUBLIC_SCOPES)
        if creds and creds.valid:
            return True
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(GARequest())
            with open(TOKEN_PATH, "w", encoding="utf-8") as f:
                f.write(creds.to_json())
            return True
    except Exception:
        return False
    return False


def account_status() -> dict:
    """登录态 + CU 信息（逻辑等价 colab_bridge.py cmd_auth_status）。"""
    if not has_valid_token():
        return {"ok": True, "logged_in": False}
    try:
        from colab_cli.common import state
        info = state.client.get_consumption_user_info()
        # info.assignments_count 会滞后（刚释放的实例还计一段时间），
        # 实际列表才是权威，两者不一致时以列表为准。
        count = int(info.assignments_count)
        try:
            count = len(state.client.list_assignments())
        except Exception:
            pass
        return {
            "ok": True,
            "logged_in": True,
            "email": _account_email(),
            "paid_balance": round(float(info.paid_compute_units_balance), 2),
            "burn_rate_hourly": round(float(info.consumption_rate_hourly), 2),
            "active_assignments": count,
        }
    except Exception as e:
        return {"ok": False, "logged_in": True, "email": _account_email(), "error": str(e)}


def _account_email() -> str:
    import requests as rq
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request as GARequest
        from colab_cli.auth import PUBLIC_SCOPES, TOKEN_CONFIG_PATH
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, PUBLIC_SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(GARequest())
        if creds and creds.token:
            r = rq.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {creds.token}"}, timeout=8,
            )
            if r.status_code == 200:
                return str(r.json().get("email") or "")
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# 本地会话表读取
# ---------------------------------------------------------------------------
def _sync_local_sessions():
    """返回 (本地会话 dict, 云端 assignments)。

    不能用原版 state.sync_sessions()：它把结果缓存进 state._sessions 后不再回读
    sessions.json，新建/释放实例后服务进程会一直拿到旧快照——新会话被误判成
    孤儿、已释放的会话仍留在列表里。缓存唯一省的是读一次本地文件，
    网络调用（list_assignments）本来每次都会发，所以直接清掉缓存重读。
    """
    from colab_cli.common import state
    state._sessions = None
    return state.sync_sessions()


# ---------------------------------------------------------------------------
# CU 燃烧速率估算（借鉴 app.py estimate_cu_burn_rate）
# ---------------------------------------------------------------------------
CU_BURN_RATE = {
    "H100": 11.78,
    "G4": 10.05,   # RTX PRO 6000 Blackwell
    "A100": 5.37,
    "L4": 1.71,
    "T4": 1.17,
    "V100": 3.80,
    "CPU": 0.10,
}


def estimate_cu_burn_rate(gpu_name: str) -> float:
    name_up = (gpu_name or "").upper()
    if "H100" in name_up:
        return CU_BURN_RATE["H100"]
    if "PRO 6000" in name_up or "BLACKWELL" in name_up or "G4" in name_up:
        return CU_BURN_RATE["G4"]
    if "A100" in name_up:
        return CU_BURN_RATE["A100"]
    if "L4" in name_up:
        return CU_BURN_RATE["L4"]
    if "T4" in name_up:
        return CU_BURN_RATE["T4"]
    if "V100" in name_up:
        return CU_BURN_RATE["V100"]
    if "CPU" in name_up or "NONE" in name_up:
        return CU_BURN_RATE["CPU"]
    return 1.50


# ---------------------------------------------------------------------------
# 远程硬件遥测（借鉴 app.py _sample_live_hardware_telemetry_once + _get_storage_metrics_dict）
# 采集代码在 VM 内核里跑，返回结构化快照；本地面板只做转发与缓存。
# ---------------------------------------------------------------------------
_TELEMETRY_CACHE = {}          # name -> {"ts": epoch, "data": {...}}
_TELEMETRY_TTL = 5.0
_TELEMETRY_LOCK = threading.Lock()

_TELEMETRY_CODE = r'''
import json as _json, os as _os, shutil as _shutil
def _gb(v):
    return round(v / (1024 ** 3), 2)
_out = {"cuda_available": False, "gpu_name": "CPU Mode", "vram_free_gb": 0.0,
        "vram_total_gb": 0.0, "vram_used_gb": 0.0}
try:
    import torch as _torch
    if _torch.cuda.is_available():
        _free, _total = _torch.cuda.mem_get_info(0)
        _out.update(cuda_available=True, gpu_name=_torch.cuda.get_device_name(0),
                    vram_free_gb=_gb(_free), vram_total_gb=_gb(_total),
                    vram_used_gb=_gb(_total - _free))
except Exception:
    pass
try:
    with open("/proc/meminfo") as _f:
        for _line in _f:
            if _line.startswith("MemTotal:"):
                _out["ram_total_gb"] = round(int(_line.split()[1]) / (1024 ** 2), 1)
            elif _line.startswith("MemAvailable:"):
                _out["ram_available_gb"] = round(int(_line.split()[1]) / (1024 ** 2), 1)
    if "ram_total_gb" in _out and "ram_available_gb" in _out:
        _out["ram_used_gb"] = round(max(0.0, _out["ram_total_gb"] - _out["ram_available_gb"]), 1)
except Exception:
    pass
for _key, _path in (("disk", "/content"), ("shm", "/dev/shm"), ("drive", "/content/drive/MyDrive")):
    try:
        if _os.path.exists(_path):
            _u = _shutil.disk_usage(_path)
            _out[_key + "_total_gb"] = _gb(_u.total)
            _out[_key + "_used_gb"] = _gb(_u.total - _u.free)
            _out[_key + "_free_gb"] = _gb(_u.free)
    except Exception:
        pass
try:
    _u2 = _os.getloadavg()
    _out["loadavg_1m"], _out["loadavg_5m"] = round(_u2[0], 2), round(_u2[1], 2)
except Exception:
    pass
try:
    _out["cpu_count"] = _os.cpu_count()
except Exception:
    pass
print("PANEL_TELEMETRY:" + _json.dumps(_out))
'''


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07")
_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\)")
_EXC_LINE_RE = re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception|Interrupt|Exit)\b.*$", re.MULTILINE)


def _looks_failed(output: str) -> bool:
    """colab exec 远程抛异常时 CLI 仍返回 rc=0，只能从输出里认 traceback。"""
    out = _clean_remote_output(output)
    return bool(_TRACEBACK_RE.search(out) and _EXC_LINE_RE.search(out))


def _run_code_on_session(name: str, code: str, timeout_s: int = 60):
    """把代码写到临时 .py 用 colab exec -f 执行，返回 (rc, stdout, stderr)。"""
    import tempfile
    fd, tmp_path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(code)
        rc, out, err = run_cli(
            ["exec", "-s", name, "-f", tmp_path, "--timeout", str(min(280, timeout_s))],
            timeout=max(CLI_TIMEOUT_DEFAULT, timeout_s + 60),
        )
        return rc, _ANSI_RE.sub("", out or ""), _ANSI_RE.sub("", err or "")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def collect_telemetry(name: str, force: bool = False) -> dict:
    """在远程实例采样硬件指标；带 TTL 缓存避免频繁 exec。"""
    now = time.time()
    with _TELEMETRY_LOCK:
        cached = _TELEMETRY_CACHE.get(name)
        if cached and not force and now - cached["ts"] < _TELEMETRY_TTL:
            return cached["data"]
    rc, out, err = _run_code_on_session(name, _TELEMETRY_CODE, timeout_s=60)
    data = {"ok": False, "error": ""}
    if rc == 0:
        for line in out.splitlines():
            if line.startswith("PANEL_TELEMETRY:"):
                try:
                    data = {"ok": True, **json.loads(line[len("PANEL_TELEMETRY:"):])}
                except Exception as e:
                    data = {"ok": False, "error": f"解析失败: {e}"}
                break
        else:
            data = {"ok": False, "error": (out or err).strip()[-400:]}
    else:
        data = {"ok": False, "error": (err or out).strip()[-400:]}
    data["ts"] = now
    with _TELEMETRY_LOCK:
        _TELEMETRY_CACHE[name] = {"ts": now, "data": data}
    return data


@app.get("/api/sessions/{name}/telemetry")
def api_telemetry(name: str, request: Request, force: int = 0):
    check_auth(request)
    if not has_valid_token():
        raise HTTPException(status_code=400, detail="未绑定 Google 账号")
    return collect_telemetry(name, force=bool(force))


@app.get("/api/telemetry/all")
def api_telemetry_all(request: Request, force: int = 0):
    """一次拿全部本地会话的硬件快照（前端轮询用），失败的会话带 error 不阻塞其它。"""
    check_auth(request)
    if not has_valid_token():
        return {"ok": True, "logged_in": False, "sessions": {}}
    try:
        from colab_cli.common import state
        local, _ = _sync_local_sessions()
    except Exception:
        local = {}
    result = {}
    for s in list(local.values()):
        try:
            result[s.name] = collect_telemetry(s.name, force=bool(force))
        except Exception as e:
            result[s.name] = {"ok": False, "error": str(e)[:200]}
    items = []
    for n, t in result.items():
        items.append({"name": n, **t})
    return {"ok": True, "sessions": result, "items": items,
            "cu_rates": {"H100": 11.78, "G4": 10.05, "A100": 5.37, "L4": 1.71, "T4": 1.17, "V100": 3.80, "CPU": 0.10}}


# ---------------------------------------------------------------------------
# /dev/shm 扩容（借鉴 app.py setup_colab_ramdisk）
# ---------------------------------------------------------------------------
_SHM_CODE = r'''
import os as _os, shutil as _shutil, subprocess as _sp
def _gb(v):
    return round(v / (1024 ** 3), 2)
_ram_total = 0.0
try:
    with open("/proc/meminfo") as _f:
        for _l in _f:
            if _l.startswith("MemTotal:"):
                _ram_total = int(_l.split()[1]) / (1024 ** 2)
                break
except Exception:
    pass
_before = _gb(_shutil.disk_usage("/dev/shm").total) if _os.path.exists("/dev/shm") else 0.0
_note = ""
if _ram_total < 16.0:
    _note = "RAM 共 %.1f GB，不做扩容（<16GB）" % _ram_total
elif not _os.path.exists("/dev/shm"):
    _note = "/dev/shm 不存在，跳过"
else:
    _target = min(88, max(30, int(_ram_total * 0.5)))
    if _before + 1.0 < _target:
        try:
            _sp.run(["mount", "-o", "remount,size=%dG" % _target, "/dev/shm"],
                    check=False, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
            _note = "已 remount 到 %d GB" % _target
        except Exception as _e:
            _note = "remount 失败: %s" % _e
    else:
        _note = "当前 %.1f GB 已足够，无需扩容" % _before
_after = _gb(_shutil.disk_usage("/dev/shm").total) if _os.path.exists("/dev/shm") else 0.0
print("PANEL_SHM:" + repr({"ram_total_gb": round(_ram_total, 1), "shm_before_gb": _before,
                           "shm_after_gb": _after, "note": _note}))
'''


@app.post("/api/sessions/{name}/shm-expand")
def api_shm_expand(name: str, request: Request):
    """扩容 /dev/shm 到总内存 50%（30~88GB）。VM 重启后失效，需要再用时重新调。"""
    check_auth(request)
    if not has_valid_token():
        raise HTTPException(status_code=400, detail="未绑定 Google 账号")
    rc, out, err = _run_code_on_session(name, _SHM_CODE, timeout_s=90)
    if rc != 0:
        return {"ok": False, "output": (err or out).strip()[-1500:]}
    for line in out.splitlines():
        if line.startswith("PANEL_SHM:"):
            try:
                info = ast.literal_eval(line[len("PANEL_SHM:"):])
                return {"ok": True, "shm_resized": info["shm_after_gb"] > info["shm_before_gb"], **info}
            except Exception as e:
                return {"ok": False, "error": f"解析失败: {e}", "output": line[:500]}
    return {"ok": False, "output": (out or err).strip()[-1500:]}


# ---------------------------------------------------------------------------
# 会话创建时间登记：name -> {"started_at": epoch, "accelerator": "T4", ...}
# 用于估算"本会话已花费 CU"。加速器信息来自 colab sessions 输出解析或 new 时登记。
_SESSION_META_LOCK = threading.RLock()
_SESSION_META = {}
_SESSION_META_PATH = os.path.expanduser("~/.config/colab-cli/panel_sessions.json")


def _load_session_meta():
    with _SESSION_META_LOCK:
        if not _SESSION_META and os.path.exists(_SESSION_META_PATH):
            try:
                with open(_SESSION_META_PATH, "r", encoding="utf-8") as f:
                    _SESSION_META.update(json.load(f))
            except Exception:
                pass
        return _SESSION_META


def _save_session_meta():
    with _SESSION_META_LOCK:
        try:
            os.makedirs(os.path.dirname(_SESSION_META_PATH), exist_ok=True)
            with open(_SESSION_META_PATH, "w", encoding="utf-8") as f:
                json.dump(_SESSION_META, f)
        except Exception:
            pass


def register_session_meta(name: str, accelerator: str):
    meta = _load_session_meta()
    with _SESSION_META_LOCK:
        if name not in meta:
            meta[name] = {"started_at": time.time(), "accelerator": (accelerator or "CPU").upper()}
        else:
            meta[name]["accelerator"] = (accelerator or meta[name].get("accelerator", "CPU")).upper()
        _save_session_meta()


def forget_session_meta(name: str):
    meta = _load_session_meta()
    with _SESSION_META_LOCK:
        meta.pop(name, None)
        _save_session_meta()


def session_cost_estimate(name: str, accelerator: str = "") -> dict:
    """估算会话已花费 CU：燃烧速率 × 运行时长。"""
    meta = _load_session_meta()
    info = meta.get(name) or {}
    accel = (accelerator or info.get("accelerator") or "CPU").upper()
    started = info.get("started_at")
    if not started:
        return {"accelerator": accel, "burn_rate_per_hour": None, "elapsed_hours": None, "estimated_cu": None}
    rate = estimate_cu_burn_rate(accel)
    hours = max(0.0, (time.time() - started) / 3600.0)
    return {
        "accelerator": accel,
        "burn_rate_per_hour": rate,
        "elapsed_hours": round(hours, 3),
        "estimated_cu": round(rate * hours, 3),
    }


# ---------------------------------------------------------------------------
# 会话列表解析（colab sessions 人类可读输出）
# 两种实际格式：
#   [verify1] m-s-usc1c0-1y1nwxq78cs5k | Hardware: CPU | Shape: Standard | Variant: DEFAULT
#   [?]       m-s-use4a4-378daepl0vgve | Hardware: CPU | Shape: Standard | Variant: DEFAULT
# 方括号内是本地会话名（? = 本地表无记录），其后是 server endpoint。
# ---------------------------------------------------------------------------
_SESSION_PIPE_RE = re.compile(
    r"^\s*(?:\[(?P<label>[^\]]*)\]\s*)?(?P<target>\S+)\s*\|\s*Hardware:\s*(?P<hw>[^|]+?)\s*"
    r"(?:\|\s*Shape:\s*(?P<shape>[^|]+?)\s*)?(?:\|\s*Variant:\s*(?P<variant>[^|\s]*))?\s*$",
    re.IGNORECASE,
)
_SESSION_LEGACY_RE = re.compile(r"^\s*(\S+)\s+(G4|H100|A100|L4|T4|V100|NONE|CPU)\b", re.IGNORECASE)


def _norm_accel(hw: str) -> str:
    h = (hw or "").strip().upper()
    if not h or h in ("NONE", "CPU", "STANDARD"):
        return "CPU"
    return h


def parse_sessions_output(text: str):
    """从 colab sessions 输出提取会话信息；解析不出时返回空表。"""
    out = []
    for line in (text or "").splitlines():
        line = line.rstrip()
        if not line.strip() or line.lstrip().startswith("[colab]"):
            continue
        m = _SESSION_PIPE_RE.match(line)
        if m:
            label = (m.group("label") or "").strip()
            target = m.group("target").strip()
            named = bool(label) and label != "?"
            out.append({
                "name": label if named else target,
                "endpoint": target,
                "accelerator": _norm_accel(m.group("hw")),
                "hardware": (m.group("hw") or "").strip(),
                "shape": (m.group("shape") or "").strip(),
                "variant": (m.group("variant") or "").strip(),
                "has_local": named,
            })
            continue
        m = _SESSION_LEGACY_RE.match(line)
        if m:
            out.append({"name": m.group(1), "endpoint": "", "accelerator": _norm_accel(m.group(2)),
                        "hardware": m.group(2).upper(), "shape": "", "variant": "", "has_local": True})
    return out


# ---------------------------------------------------------------------------
# 实例就绪探测（借鉴 colab_bridge.py _wait_for_runtime_ready）
# ---------------------------------------------------------------------------
def wait_for_runtime_ready(session, timeout_s: int = 45) -> bool:
    """轮询实例 Jupyter /api/kernels：200 就绪；连续 404/410 视为实例已销毁返回 False。"""
    base_url = session.url.rstrip("/")
    headers = {
        "Accept": "application/json",
        "X-Colab-Client-Agent": "colab-cli",
        "X-Colab-Runtime-Proxy-Token": session.token,
        "Authorization": f"Bearer {session.token}",
    }
    params = {"authuser": "0", "colab-runtime-proxy-token": session.token}
    deadline = time.time() + timeout_s
    not_found = 0
    while time.time() < deadline:
        try:
            r = requests.get(base_url + "/api/kernels", headers=headers, params=params, timeout=8)
            if r.status_code == 200:
                return True
            if r.status_code in (404, 410):
                not_found += 1
                if not_found >= 2:
                    return False
        except requests.exceptions.RequestException:
            pass
        time.sleep(2.0)
    return False


# ---------------------------------------------------------------------------
# 保活心跳（借鉴 colab_bridge.py cmd_keepalive，绕开上游残缺分支自行实现）
# ---------------------------------------------------------------------------
_heartbeat_state = {
    "running": False,
    "last_run": 0,
    "touched": 0,
    "last_error": "",
    "thread": None,
}


def _refresh_proxy_token(s):
    """用最新 runtime proxy 信息刷新会话 token（延长有效期）。"""
    from colab_cli.common import state
    by_ep = {a.endpoint: a for a in state.client.list_assignments()}
    if s.endpoint in by_ep:
        info = by_ep[s.endpoint].runtime_proxy_info
        s.token = info.token
        s.url = info.url
        s.token_expires_at = info.expires_at()
        state.store.add(s)
        return True
    return False


def _kernel_pulse(s) -> bool:
    """内核轻脉冲：执行一行代码保持 last_activity 新鲜。"""
    from colab_cli.runtime import ColabRuntime
    base_url = s.url.rstrip("/")
    headers = {
        "Accept": "application/json",
        "X-Colab-Client-Agent": "colab-cli",
        "X-Colab-Runtime-Proxy-Token": s.token,
        "Authorization": f"Bearer {s.token}",
    }
    params = {"authuser": "0", "colab-runtime-proxy-token": s.token}
    active_kid = None
    try:
        r = requests.get(base_url + "/api/kernels", headers=headers, params=params, timeout=10)
        if r.status_code == 200 and r.text:
            kernels = r.json()
            known = [k.get("id") for k in kernels if k.get("id")]
            active_kid = s.kernel_id if (s.kernel_id in known) else (known[0] if known else None)
            # 清理多余空转内核（借鉴上游 keepalive）
            for extra in known:
                if extra != active_kid:
                    try:
                        requests.delete(base_url + "/api/kernels/" + extra, headers=headers, params=params, timeout=5)
                    except Exception:
                        pass
    except Exception:
        return False

    def _save_kid(kid):
        s.kernel_id = kid
        from colab_cli.common import state
        state.store.add(s)

    def _save_sid(sid):
        s.session_id = sid
        from colab_cli.common import state
        state.store.add(s)

    rt = ColabRuntime(
        s.url, s.token, session_name=s.name,
        kernel_id=active_kid, session_id=s.session_id,
        on_kernel_started=_save_kid, on_session_started=_save_sid,
    )
    try:
        rt.execute_code("_panel_hb = 1", allow_stdin=False, timeout=12)
        return True
    finally:
        try:
            rt.stop(shutdown_kernel=False)
        except Exception:
            pass


def _heartbeat_loop():
    while True:
        interval = max(60, HEARTBEAT_INTERVAL_SECONDS)
        time.sleep(interval)
        if HEARTBEAT_INTERVAL_SECONDS <= 0 or not has_valid_token():
            continue
        touched = 0
        err = ""
        try:
            from colab_cli.common import state
            local_sessions, assignments = _sync_local_sessions()
            ep_map = {a.endpoint: a for a in assignments}
            for s in list(local_sessions.values()):
                try:
                    if s.endpoint in ep_map:
                        _refresh_proxy_token(s)
                    if s.url and s.token and _kernel_pulse(s):
                        touched += 1
                except Exception as e:
                    err = str(e)[:300]
        except Exception as e:
            err = str(e)[:300]
        _heartbeat_state["last_run"] = time.time()
        _heartbeat_state["touched"] = touched
        _heartbeat_state["last_error"] = err


def ensure_heartbeat_started():
    if _heartbeat_state["thread"] is None and HEARTBEAT_INTERVAL_SECONDS > 0:
        t = threading.Thread(target=_heartbeat_loop, daemon=True)
        _heartbeat_state["thread"] = t
        _heartbeat_state["running"] = True
        t.start()


# ---------------------------------------------------------------------------
# 闲置看门狗（借鉴 app.py _idle_watchdog_loop）
# ---------------------------------------------------------------------------
_watchdog_state = {
    "last_activity": time.time(),
    "thread": None,
    "triggered": False,
    "last_action": "",
}


def touch_activity():
    """外部活动（exec/keepalive/用户操作）刷新闲置计时。"""
    _watchdog_state["last_activity"] = time.time()
    _watchdog_state["triggered"] = False


def _watchdog_loop():
    while True:
        time.sleep(15)
        if IDLE_TIMEOUT_MINUTES <= 0 or not has_valid_token():
            continue
        idle_sec = time.time() - _watchdog_state["last_activity"]
        if idle_sec >= IDLE_TIMEOUT_MINUTES * 60 and not _watchdog_state["triggered"]:
            _watchdog_state["triggered"] = True
            # 复用 stop-all 逻辑释放全部实例，停止 CU 计费
            try:
                from colab_cli.common import state
                from colab_cli.runtime import ColabRuntime
                local_sessions, assignments = _sync_local_sessions()
                ep_seen = set()
                stopped = []
                for s in list(local_sessions.values()):
                    try:
                        rt = ColabRuntime(s.url, s.token, kernel_id=s.kernel_id)
                        try:
                            rt.stop(shutdown_kernel=True)
                        except Exception:
                            pass
                        state.client.unassign(s.endpoint)
                        stopped.append(s.name)
                    except Exception:
                        pass
                    ep_seen.add(s.endpoint)
                    state.store.remove(s.name)
                    forget_session_meta(s.name)
                for a in assignments:
                    if a.endpoint in ep_seen:
                        continue
                    try:
                        state.client.unassign(a.endpoint)
                        stopped.append(f"orphan:{a.endpoint}")
                    except Exception:
                        pass
                _watchdog_state["last_action"] = f"{time.strftime('%H:%M:%S')} 闲置 {IDLE_TIMEOUT_MINUTES} 分钟，已自动释放 {len(stopped)} 个实例"
            except Exception as e:
                _watchdog_state["last_action"] = f"看门狗释放失败: {str(e)[:200]}"


def ensure_watchdog_started():
    if _watchdog_state["thread"] is None:
        t = threading.Thread(target=_watchdog_loop, daemon=True)
        _watchdog_state["thread"] = t
        t.start()


# ---------------------------------------------------------------------------
# headless Drive 挂载（借鉴 colab_bridge.py _build_rclone_drive_conf + boot_code）
# ---------------------------------------------------------------------------
def build_rclone_conf() -> str:
    """复用本地 OAuth token 生成 rclone 配置（需 drive.file scope）。"""
    if not os.path.exists(TOKEN_PATH):
        return ""
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request as GARequest
        from colab_cli.auth import PUBLIC_SCOPES
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, PUBLIC_SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(GARequest())
        raw = json.loads(creds.to_json())
        scopes = raw.get("scopes") or []
        drive_scope = "drive" if any(s.endswith("/drive") and not s.endswith("drive.file") for s in scopes) else (
            "drive.file" if any("drive.file" in s for s in scopes) else ""
        )
        if not drive_scope or not raw.get("token") or not raw.get("refresh_token"):
            return ""
        tok_obj = {
            "access_token": raw["token"],
            "token_type": "Bearer",
            "refresh_token": raw["refresh_token"],
        }
        if raw.get("expiry"):
            tok_obj["expiry"] = raw["expiry"]
        lines = [
            "[panel_gdrive]",
            "type = drive",
            f"client_id = {raw.get('client_id', '')}",
            f"client_secret = {raw.get('client_secret', '')}",
            f"scope = {drive_scope}",
            f"token = {json.dumps(tok_obj, separators=(',', ':'))}",
            "",
        ]
        return "\n".join(lines)
    except Exception:
        return ""


# VM 内执行的 rclone 安装+挂载脚本（照抄 colab_bridge.py boot_code 第 0 段逻辑）
_RCLONE_BOOT_CODE = r"""
import os, shutil, subprocess, time

rc_stage = "/content/.panel_rclone.conf"
rc_dir = os.path.expanduser("~/.config/rclone")
os.makedirs(rc_dir, mode=0o700, exist_ok=True)
rc_dst = os.path.join(rc_dir, "rclone.conf")
shutil.move(rc_stage, rc_dst)
os.chmod(rc_dst, 0o600)

if not shutil.which("rclone"):
    subprocess.run(
        ["bash", "-lc", "curl -fsSL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rclone.zip && unzip -q -o /tmp/rclone.zip -d /tmp && cp /tmp/rclone-*-linux-amd64/rclone /usr/local/bin/rclone && chmod +x /usr/local/bin/rclone && rm -rf /tmp/rclone*"],
        check=False,
    )

if shutil.which("rclone") and not os.path.ismount("/content/drive/MyDrive"):
    subprocess.run(["rclone", "mkdir", "panel_gdrive:Panel_Drive"], check=False)
    if os.path.exists("/dev/fuse") and (shutil.which("fusermount3") or shutil.which("fusermount")):
        os.makedirs("/content/drive/MyDrive", exist_ok=True)
        subprocess.Popen(
            ["rclone", "mount", "panel_gdrive:Panel_Drive", "/content/drive/MyDrive",
             "--vfs-cache-mode", "writes", "--vfs-write-back", "3s",
             "--dir-cache-time", "15s", "--drive-chunk-size", "64M",
             "--buffer-size", "32M", "--allow-non-empty"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for _ in range(10):
            if os.path.ismount("/content/drive/MyDrive"):
                print("DRIVE_MOUNTED")
                break
            time.sleep(0.5)
        else:
            print("FUSE_UNAVAILABLE")  # 容器无 FUSE 权限，可用 rclone copy 同步代替
    else:
        print("FUSE_UNAVAILABLE")
else:
    print("DRIVE_MOUNTED")
"""


# ---------------------------------------------------------------------------
# API 路由
# ---------------------------------------------------------------------------
@app.get("/api/ping")
def ping(request: Request):
    check_auth(request)
    return {"ok": True}


@app.get("/api/auth/url")
def api_auth_url(request: Request):
    check_auth(request)
    try:
        return {"ok": True, "auth_url": build_auth_url()}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"生成授权 URL 失败: {e}")


@app.post("/api/auth/code")
async def api_auth_code(request: Request):
    check_auth(request)
    body = await request.json()
    code = str(body.get("code", "")).strip()
    if not code:
        raise HTTPException(status_code=400, detail="缺少 code")
    return exchange_auth_code(code)


@app.get("/api/auth/status")
def api_auth_status(request: Request):
    check_auth(request)
    return account_status()


@app.post("/api/auth/logout")
def api_auth_logout(request: Request):
    check_auth(request)
    removed = []
    for p in (TOKEN_PATH, os.path.expanduser("~/.config/colab-cli/pkce_verifier.txt")):
        if os.path.exists(p):
            os.remove(p)
            removed.append(os.path.basename(p))
    # 清空本地成本追踪
    with _SESSION_META_LOCK:
        _SESSION_META.clear()
        if os.path.exists(_SESSION_META_PATH):
            os.remove(_SESSION_META_PATH)
    _invalidate_sessions_cache()
    return {"ok": True, "removed": removed}


# colab sessions 单次约 1.7~2.2s，并发时 CLI 内部 filelock 串行排队更慢。
# 短 TTL 缓存吸收刷新/多标签页造成的重复调用；refresh=1 强制穿透。
_SESSIONS_CACHE = {"at": 0.0, "data": None}
_SESSIONS_CACHE_TTL = 3.0


def _invalidate_sessions_cache():
    _SESSIONS_CACHE["at"] = 0.0
    _SESSIONS_CACHE["data"] = None


@app.get("/api/sessions")
def api_sessions(request: Request, refresh: int = 0):
    check_auth(request)
    # 未登录时 colab sessions 会进交互式 OAuth 流程挂起——先查 token
    if not has_valid_token():
        return {"ok": True, "logged_in": False, "output": "未绑定 Google 账号，请先完成账号绑定。", "returncode": 0, "sessions": []}
    now = time.time()
    if not refresh and _SESSIONS_CACHE["data"] is not None and now - _SESSIONS_CACHE["at"] < _SESSIONS_CACHE_TTL:
        return _SESSIONS_CACHE["data"]
    rc, out, err = run_cli(["sessions"], timeout=60)
    parsed = parse_sessions_output(out)
    # 本地会话表端点集合——用于把云端多出来的 assignment 标成孤儿
    local_eps = set()
    listed_eps = {s.get("endpoint") for s in parsed if s.get("endpoint")}
    try:
        from colab_cli.common import state
        local, assignments = _sync_local_sessions()
        local_eps = {s.endpoint for s in local.values()}
        # 云端有、sessions 输出没列出的 assignment 也补进列表，否则用户看不到
        for a in assignments:
            if a.endpoint in listed_eps:
                continue
            accel = str(getattr(a, "accelerator", "")).split(".")[-1]
            parsed.append({
                "name": a.endpoint,
                "endpoint": a.endpoint,
                "accelerator": _norm_accel(accel),
                "hardware": accel,
                "shape": "",
                "variant": str(getattr(a, "variant", "")).split(".")[-1],
                "has_local": False,
                "orphan": True,
            })
    except Exception:
        pass
    orphan_eps = []
    for s in parsed:
        s["orphan"] = bool(s.get("endpoint") and s["endpoint"] not in local_eps)
        if s["orphan"]:
            orphan_eps.append(s["endpoint"])
    # 登记成本追踪（孤儿不进 meta，避免释放后仍计费）
    for s in parsed:
        if not s["orphan"]:
            register_session_meta(s["name"], s["accelerator"])
    for s in parsed:
        s["cost"] = session_cost_estimate(s["name"], s["accelerator"])
    result = {"ok": rc == 0, "output": out.strip() or err.strip(), "returncode": rc,
              "sessions": parsed, "orphans": orphan_eps, "count": len(parsed)}
    _SESSIONS_CACHE["at"] = now
    _SESSIONS_CACHE["data"] = result
    return result


@app.post("/api/sessions")
async def api_new_session(request: Request):
    check_auth(request)
    if not has_valid_token():
        raise HTTPException(status_code=400, detail="未绑定 Google 账号，请先完成账号绑定")
    body = await request.json()
    gpu = str(body.get("gpu", "cpu")).strip().lower()
    name = str(body.get("name", "")).strip()
    high_mem = bool(body.get("high_mem", False))
    reuse = bool(body.get("reuse", True))  # 幂等复用：已有同名同型号实例直接返回
    valid_gpus = {"cpu", "t4", "l4", "a100", "h100", "g4"}
    if gpu not in valid_gpus:
        raise HTTPException(status_code=400, detail=f"gpu 必须是 {sorted(valid_gpus)} 之一")
    accel_label = "CPU" if gpu == "cpu" else gpu.upper()

    # 幂等复用（借鉴 colab_bridge.py cmd_wake）：查本地会话 + 云端 assignments
    if reuse:
        try:
            from colab_cli.common import state
            from colab_cli.client import Accelerator, Variant, Shape, PostAssignmentResponse
            local_sessions, assignments = _sync_local_sessions()
            target = None
            if name:
                target = local_sessions.get(name)
            elif local_sessions:
                target = next((s for s in local_sessions.values()
                               if (s.accelerator or "NONE").upper() == accel_label.upper()
                               or (accel_label == "CPU" and (s.accelerator or "NONE").upper() in ("NONE", "CPU"))), None)
                if target:
                    name = target.name
            if target is None and assignments:
                want = None if accel_label == "CPU" else getattr(Accelerator, accel_label, None)
                for a in assignments:
                    a_accel = (a.accelerator.value or "NONE").upper() if hasattr(a.accelerator, "value") else str(a.accelerator).upper()
                    if accel_label == "CPU" and a_accel in ("NONE", "CPU"):
                        target = a
                        break
                    if want is not None and a_accel == accel_label:
                        target = a
                        break
                if target is not None:
                    # 孤儿纳管：云端分配 → 本地会话登记（借鉴 cmd_wake）
                    from colab_cli.state import SessionState
                    from colab_cli.client import Shape as _Shape
                    s_name = name or f"adopted-{uuid.uuid4().hex[:6]}"
                    s = SessionState(
                        name=s_name,
                        token=target.runtime_proxy_info.token,
                        url=target.runtime_proxy_info.url,
                        endpoint=target.endpoint,
                        token_expires_at=target.runtime_proxy_info.expires_at(),
                        variant="GPU",
                        accelerator=accel_label if accel_label != "CPU" else "NONE",
                        machine_shape="STANDARD",
                    )
                    state.store.add(s)
                    register_session_meta(s_name, accel_label)
                    ready = wait_for_runtime_ready(s, timeout_s=30)
                    return {"ok": True, "reused": True, "adopted_orphan": True, "name": s_name,
                            "ready": ready, "output": f"已纳管云端已有 {accel_label} 实例: {s_name}"}
            if target is not None:
                # 校验已有实例探活；失效则清理后走新建
                if wait_for_runtime_ready(target, timeout_s=15):
                    register_session_meta(target.name, accel_label)
                    return {"ok": True, "reused": True, "adopted_orphan": False, "name": target.name,
                            "ready": True, "output": f"复用已有实例: {target.name} ({accel_label})"}
                try:
                    state.client.unassign(target.endpoint)
                except Exception:
                    pass
                state.prune_session(target.name)
                forget_session_meta(target.name)
        except HTTPException:
            raise
        except Exception as e:
            pass  # 复用检查失败不影响新建

    # 新建
    if not name:
        name = f"panel-{uuid.uuid4().hex[:6]}"
    args = ["new", "-s", name]
    if gpu != "cpu":
        args += ["--gpu", gpu.upper()]
    if high_mem:
        args.append("--high-mem")
    rc, out, err = run_cli(args, timeout=CLI_TIMEOUT_LONG)
    text = (out + "\n" + err).strip()
    _invalidate_sessions_cache()
    if rc == 0:
        register_session_meta(name, accel_label)
        # 就绪探测（借鉴 _wait_for_runtime_ready）：新建后等内核可用
        ready = None
        try:
            from colab_cli.common import state
            s = state.store.get(name)
            if s:
                ready = wait_for_runtime_ready(s, timeout_s=50)
        except Exception:
            pass
        return {"ok": True, "reused": False, "name": name, "ready": ready, "output": text[-2000:], "returncode": rc}
    return {"ok": False, "name": name, "output": text[-2000:], "returncode": rc}


@app.get("/api/sessions/{name}")
def api_session_status(name: str, request: Request):
    check_auth(request)
    if not has_valid_token():
        raise HTTPException(status_code=400, detail="未绑定 Google 账号")
    rc, out, err = run_cli(["status", "-s", name], timeout=60)
    return {"ok": rc == 0, "output": (out.strip() or err.strip()), "returncode": rc,
            "cost": session_cost_estimate(name)}


@app.post("/api/sessions/{name}/restart-kernel")
def api_restart_kernel(name: str, request: Request):
    check_auth(request)
    if not has_valid_token():
        raise HTTPException(status_code=400, detail="未绑定 Google 账号")
    touch_activity()
    rc, out, err = run_cli(["restart-kernel", "-s", name], timeout=CLI_TIMEOUT_LONG)
    return {"ok": rc == 0, "output": (out + "\n" + err).strip()[-2000:], "returncode": rc}


@app.post("/api/sessions/{name}/exec")
async def api_exec(name: str, request: Request):
    check_auth(request)
    if not has_valid_token():
        raise HTTPException(status_code=400, detail="未绑定 Google 账号")
    body = await request.json()
    code = str(body.get("code", "")).strip()
    timeout_s = min(float(body.get("timeout", 120)), 280)
    if not code:
        raise HTTPException(status_code=400, detail="缺少 code")
    touch_activity()  # 代码执行视为活跃，重置看门狗计时
    rc, out, err = _run_code_on_session(name, code, timeout_s=timeout_s)
    combined = _clean_remote_output(out + ("\n" + err if err.strip() else "")).strip()
    ok = rc == 0 and not _looks_failed(combined)
    return {"ok": ok, "output": combined[-5000:], "returncode": rc if ok else (rc or -1)}


# ---------------------------------------------------------------------------
# 后台任务 + 实时控制台（借鉴 app.py TerminalProgressMirror 的思路：
# 长任务不阻塞 HTTP，输出逐行入环形缓冲，前端轮询增量拉取）
# ---------------------------------------------------------------------------
_JOBS = {}
_JOBS_LOCK = threading.Lock()
_JOB_MAX_LINES = 3000


def _clean_remote_output(text: str) -> str:
    """去掉 ANSI、把 jupyter 的短横线横幅和紧贴的 Traceback 拆成独立行。"""
    out = _ANSI_RE.sub("", text or "").replace("\r\n", "\n").replace("\r", "\n")
    out = re.sub(r"-{20,}", "\n", out)
    out = re.sub(r"(?<!\n)(Traceback \(most recent call last\))", r"\n\1", out)
    out = re.sub(r"(Traceback \(most recent call last\))(?!\n)", r"\1\n", out)
    return out


def _job_append(job_id: str, text: str):
    for piece in _clean_remote_output(text).split("\n"):
        piece = piece.rstrip()
        if not piece:
            continue
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
            if not job:
                return
            job["lines"].append(piece)
            if len(job["lines"]) > _JOB_MAX_LINES:
                del job["lines"][: len(job["lines"]) - _JOB_MAX_LINES]
            job["updated_at"] = time.time()


def _job_worker(job_id: str, name: str, py_path: str, timeout_s: int):
    proc = None
    try:
        cmd = [CLI_BIN, "exec", "-s", name, "-f", py_path, "--timeout", str(timeout_s)]
        env = {**os.environ, "TERM": "dumb", "NO_COLOR": "1", "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        deadline = time.time() + max(CLI_TIMEOUT_LONG, timeout_s + 60)
        tail = ""
        killed = False
        for line in proc.stdout:
            _job_append(job_id, line)
            tail = (tail + line)[-6000:]
            if time.time() > deadline:
                proc.kill()
                _job_append(job_id, f"[panel] 超过 {timeout_s}s 上限，已终止")
                killed = True
                break
        rc = proc.wait(timeout=30)
        # 远程抛异常时 CLI 仍返回 rc=0，靠 traceback 特征判定失败
        failed = killed or rc != 0 or _looks_failed(tail)
        with _JOBS_LOCK:
            if job_id in _JOBS:
                _JOBS[job_id]["status"] = "failed" if failed else "done"
                _JOBS[job_id]["returncode"] = rc if not failed else (rc or -1)
                _JOBS[job_id]["finished_at"] = time.time()
    except Exception as e:
        _job_append(job_id, f"[panel] 执行异常: {e}")
        with _JOBS_LOCK:
            if job_id in _JOBS:
                _JOBS[job_id]["status"] = "error"
                _JOBS[job_id]["returncode"] = -1
                _JOBS[job_id]["finished_at"] = time.time()
    finally:
        if proc and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
        if os.path.exists(py_path):
            os.unlink(py_path)


@app.post("/api/sessions/{name}/exec-async")
async def api_exec_async(name: str, request: Request):
    """后台执行代码，立即返回 job_id；用 /api/jobs/{job_id} 轮询增量输出。"""
    check_auth(request)
    if not has_valid_token():
        raise HTTPException(status_code=400, detail="未绑定 Google 账号")
    body = await request.json()
    code = str(body.get("code", "")).strip()
    timeout_s = min(int(float(body.get("timeout", 120))), 280)
    if not code:
        raise HTTPException(status_code=400, detail="缺少 code")
    touch_activity()
    job_id = uuid.uuid4().hex[:12]
    import tempfile
    fd, py_path = tempfile.mkstemp(prefix=f"paneljob_{job_id}_", suffix=".py")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(code)
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "id": job_id, "session": name, "status": "running",
            "lines": [], "started_at": time.time(), "updated_at": time.time(),
            "finished_at": 0, "returncode": None, "timeout": timeout_s,
        }
    threading.Thread(target=_job_worker, args=(job_id, name, py_path, timeout_s), daemon=True).start()
    return {"ok": True, "job_id": job_id, "session": name}


@app.get("/api/jobs")
def api_jobs(request: Request):
    check_auth(request)
    with _JOBS_LOCK:
        items = [{"id": j["id"], "session": j["session"], "status": j["status"],
                  "lines": len(j["lines"]), "started_at": j["started_at"],
                  "returncode": j["returncode"]}
                 for j in sorted(_JOBS.values(), key=lambda x: x["started_at"], reverse=True)[:50]]
    return {"ok": True, "jobs": items}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str, request: Request, from_line: int = 0):
    check_auth(request)
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="任务不存在或已清理")
        lines = job["lines"][from_line:]
        return {
            "ok": True, "id": job["id"], "session": job["session"],
            "status": job["status"], "returncode": job["returncode"],
            "lines": lines, "next_line": from_line + len(lines),
            "total_lines": len(job["lines"]),
            "started_at": job["started_at"], "finished_at": job["finished_at"],
            "elapsed": round((job["finished_at"] or time.time()) - job["started_at"], 1),
        }


@app.delete("/api/jobs/{job_id}")
def api_job_delete(job_id: str, request: Request):
    check_auth(request)
    with _JOBS_LOCK:
        _JOBS.pop(job_id, None)
    return {"ok": True, "removed": job_id}


@app.post("/api/sessions/{name}/upload")
async def api_upload(name: str, request: Request, file: UploadFile = File(...), remote_path: str = "content/"):
    check_auth(request)
    if not has_valid_token():
        raise HTTPException(status_code=400, detail="未绑定 Google 账号")
    import tempfile
    suffix = os.path.splitext(file.filename or "file")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        rp = remote_path.rstrip("/")
        dest = f"{rp}/{file.filename}" if rp.endswith("content") or "/" in remote_path else f"content/{remote_path}"
        rc, out, err = run_cli(["upload", "-s", name, tmp_path, dest], timeout=CLI_TIMEOUT_LONG)
        return {"ok": rc == 0, "output": (out + "\n" + err).strip()[-2000:], "returncode": rc, "remote": dest}
    finally:
        os.unlink(tmp_path)


@app.get("/api/sessions/{name}/download")
def api_download(name: str, request: Request, remote_path: str):
    check_auth(request)
    if not has_valid_token():
        raise HTTPException(status_code=400, detail="未绑定 Google 账号")
    import tempfile
    suffix = os.path.splitext(remote_path)[1] or ".bin"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        rc, out, err = run_cli(["download", "-s", name, remote_path, tmp_path], timeout=CLI_TIMEOUT_LONG)
        if rc != 0:
            return JSONResponse({"ok": False, "output": (out + "\n" + err).strip()[-2000:]}, status_code=502)
        with open(tmp_path, "rb") as f:
            data = f.read()
        from fastapi.responses import Response
        return Response(
            content=data, media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{os.path.basename(remote_path)}"'},
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.delete("/api/sessions/{name}")
def api_stop_session(name: str, request: Request):
    check_auth(request)
    if not has_valid_token():
        raise HTTPException(status_code=400, detail="未绑定 Google 账号")
    rc, out, err = run_cli(["stop", "-s", name], timeout=CLI_TIMEOUT_LONG)
    if rc == 0:
        forget_session_meta(name)
        touch_activity()
        _invalidate_sessions_cache()
    return {"ok": rc == 0, "output": (out + "\n" + err).strip()[-2000:], "returncode": rc}


@app.post("/api/stop-all")
def api_stop_all(request: Request):
    check_auth(request)
    # 释放账号下全部实例（含孤儿）——复刻 colab_bridge.py cmd_stop_all 但走 colab_cli 库。
    results = {"stopped": [], "failed": []}
    try:
        from colab_cli.common import state
        from colab_cli.runtime import ColabRuntime
        local_sessions, assignments = _sync_local_sessions()
        ep_seen = set()
        for s in list(local_sessions.values()):
            try:
                rt = ColabRuntime(s.url, s.token, kernel_id=s.kernel_id)
                try:
                    rt.stop(shutdown_kernel=True)
                except Exception:
                    pass
                state.client.unassign(s.endpoint)
                results["stopped"].append(s.name)
            except Exception as e:
                results["failed"].append({"name": s.name, "error": str(e)})
            ep_seen.add(s.endpoint)
            state.store.remove(s.name)
            forget_session_meta(s.name)
        for a in assignments:
            if a.endpoint in ep_seen:
                continue
            try:
                state.client.unassign(a.endpoint)
                results["stopped"].append(f"orphan:{a.endpoint}")
            except Exception as e:
                results["failed"].append({"name": f"orphan:{a.endpoint}", "error": str(e)})
        results["ok"] = True
        touch_activity()
    except Exception as e:
        results["ok"] = False
        results["error"] = str(e)
    _invalidate_sessions_cache()
    return results


@app.get("/api/usage")
def api_usage(request: Request):
    check_auth(request)
    # 未登录时 colab usage 同样会进交互式 OAuth 流程挂起——走库直查
    st = account_status()
    st["sessions_cost"] = [
        {"name": n, **session_cost_estimate(n, info.get("accelerator", ""))}
        for n, info in _load_session_meta().items()
    ]
    return st


# ---------------------------------------------------------------------------
# 看门狗 / 心跳 / 成本 API
# ---------------------------------------------------------------------------
@app.get("/api/settings")
def api_settings(request: Request):
    check_auth(request)
    return {
        "idle_timeout_minutes": IDLE_TIMEOUT_MINUTES,
        "watchdog": {"last_activity": _watchdog_state["last_activity"],
                     "idle_seconds": round(time.time() - _watchdog_state["last_activity"], 1),
                     "triggered": _watchdog_state["triggered"],
                     "last_action": _watchdog_state["last_action"]},
        "heartbeat": {"interval_seconds": HEARTBEAT_INTERVAL_SECONDS,
                      "running": _heartbeat_state["running"],
                      "last_run": _heartbeat_state["last_run"],
                      "touched": _heartbeat_state["touched"],
                      "last_error": _heartbeat_state["last_error"]},
    }


class _SettingsBody(dict):
    pass


@app.post("/api/settings")
async def api_set_settings(request: Request):
    global IDLE_TIMEOUT_MINUTES, HEARTBEAT_INTERVAL_SECONDS
    check_auth(request)
    body = await request.json()
    if "idle_timeout_minutes" in body:
        v = int(body["idle_timeout_minutes"])
        if v < 0 or v > 1440:
            raise HTTPException(status_code=400, detail="idle_timeout_minutes 取值 0~1440（0=禁用）")
        IDLE_TIMEOUT_MINUTES = v
        touch_activity()
    if "heartbeat_interval_seconds" in body:
        v = int(body["heartbeat_interval_seconds"])
        if v < 0 or v > 3600:
            raise HTTPException(status_code=400, detail="heartbeat_interval_seconds 取值 0~3600（0=禁用）")
        HEARTBEAT_INTERVAL_SECONDS = v
    return {"ok": True, "idle_timeout_minutes": IDLE_TIMEOUT_MINUTES,
            "heartbeat_interval_seconds": HEARTBEAT_INTERVAL_SECONDS}


@app.post("/api/heartbeat/now")
def api_heartbeat_now(request: Request):
    check_auth(request)
    if not has_valid_token():
        raise HTTPException(status_code=400, detail="未绑定 Google 账号")
    touch_activity()
    touched, errors = 0, []
    try:
        from colab_cli.common import state
        local_sessions, assignments = _sync_local_sessions()
        for s in list(local_sessions.values()):
            try:
                _refresh_proxy_token(s)
                if s.url and s.token and _kernel_pulse(s):
                    touched += 1
            except Exception as e:
                errors.append({"session": s.name, "error": str(e)[:200]})
        _heartbeat_state["last_run"] = time.time()
        _heartbeat_state["touched"] = touched
    except Exception as e:
        return {"ok": False, "touched": touched, "error": str(e)[:300]}
    return {"ok": True, "touched": touched, "errors": errors}


@app.get("/api/costs")
def api_costs(request: Request):
    check_auth(request)
    items = []
    for n, info in _load_session_meta().items():
        items.append({"name": n, **session_cost_estimate(n, info.get("accelerator", ""))})
    return {"ok": True, "sessions": items}


@app.post("/api/sessions/{name}/ready")
def api_session_ready(name: str, request: Request):
    """主动探测实例是否就绪（404=已销毁 / 超时=初始化中或网络问题）。"""
    check_auth(request)
    if not has_valid_token():
        raise HTTPException(status_code=400, detail="未绑定 Google 账号")
    try:
        from colab_cli.common import state
        s = state.get_session(name, ignore_missing_session=True)
        if s is None:
            return {"ok": False, "ready": False, "state": "missing", "detail": "本地无此会话记录"}
        ready = wait_for_runtime_ready(s, timeout_s=30)
        return {"ok": True, "ready": ready, "state": "ready" if ready else "not_ready",
                "endpoint": s.endpoint, "accelerator": s.accelerator}
    except Exception as e:
        return {"ok": False, "ready": False, "state": "error", "detail": str(e)[:300]}


@app.post("/api/sessions/{name}/drivemount-rclone")
async def api_drivemount_rclone(name: str, request: Request):
    """无浏览器挂载 Google Drive：上传 rclone 配置到 VM 后由内核安装 rclone 并 FUSE 挂载。"""
    check_auth(request)
    if not has_valid_token():
        raise HTTPException(status_code=400, detail="未绑定 Google 账号")
    conf = build_rclone_conf()
    if not conf:
        raise HTTPException(status_code=400, detail="本地 OAuth 凭证缺少 drive scope 或不存在，无法生成 rclone 配置")
    touch_activity()
    import tempfile
    from colab_cli.common import state
    from colab_cli.contents import ContentsClient
    s = state.get_session(name, ignore_missing_session=True)
    if s is None:
        raise HTTPException(status_code=404, detail=f"本地无此会话记录: {name}")
    fd, tmp_path = tempfile.mkstemp(prefix="panel_rc_", suffix=".conf")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(conf)
        os.chmod(tmp_path, 0o600)
        contents = ContentsClient(s)
        contents.upload(tmp_path, "content/.panel_rclone.conf")
        # colab exec 不支持 stdin 传代码——写临时 py 走 -f
        fd2, py_path = tempfile.mkstemp(suffix=".py")
        with os.fdopen(fd2, "w", encoding="utf-8") as f:
            f.write(_RCLONE_BOOT_CODE)
        try:
            rc, out, err = run_cli(["exec", "-s", name, "-f", py_path, "--timeout", "120"], timeout=CLI_TIMEOUT_LONG)
        finally:
            os.unlink(py_path)
        text = (out + "\n" + err).strip()
        mounted = "DRIVE_MOUNTED" in text
        fuse_fail = "FUSE_UNAVAILABLE" in text
        return {"ok": rc == 0 and (mounted or fuse_fail), "mounted": mounted,
                "fuse_unavailable": fuse_fail,
                "detail": ("已挂载到 /content/drive/MyDrive" if mounted else
                           "VM 容器无 FUSE 权限，可在 VM 内用 rclone copy/sync 命令同步代替" if fuse_fail else
                           text[-800:]),
                "output": text[-2000:]}
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/api/sessions/{name}/web-url")
def api_web_url(name: str, request: Request):
    check_auth(request)
    if not has_valid_token():
        raise HTTPException(status_code=400, detail="未绑定 Google 账号")
    rc, out, err = run_cli(["url", "-s", name], timeout=60)
    text = (out + "\n" + err).strip()
    # colab url 把链接单独打一行（无 [colab] 前缀，便于管道）
    url_line = ""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("https://colab.research.google.com/"):
            url_line = line
            break
    return {"ok": rc == 0, "url": url_line, "output": text[-1000:], "returncode": rc}


@app.post("/api/sessions/{name}/drivemount")
async def api_drivemount(name: str, request: Request):
    check_auth(request)
    if not has_valid_token():
        raise HTTPException(status_code=400, detail="未绑定 Google 账号")
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    path = str(body.get("path", "/content/drive"))
    rc, out, err = run_cli(["drivemount", "-s", name, path], timeout=CLI_TIMEOUT_LONG)
    return {"ok": rc == 0, "output": (out + "\n" + err).strip()[-2000:], "returncode": rc}


@app.post("/api/sessions/{name}/install")
async def api_install(name: str, request: Request):
    check_auth(request)
    if not has_valid_token():
        raise HTTPException(status_code=400, detail="未绑定 Google 账号")
    body = await request.json()
    packages = body.get("packages") or []
    if isinstance(packages, str):
        packages = packages.split()
    if not packages:
        raise HTTPException(status_code=400, detail="缺少 packages")
    rc, out, err = run_cli(["install", "-s", name] + list(packages), timeout=CLI_TIMEOUT_LONG)
    return {"ok": rc == 0, "output": (out + "\n" + err).strip()[-3000:], "returncode": rc}


# ---------------------------------------------------------------------------
# 远程文件浏览 / 删除（colab ls / colab rm）
# ---------------------------------------------------------------------------
@app.get("/api/sessions/{name}/ls")
def api_ls(name: str, request: Request, path: str = "content"):
    check_auth(request)
    if not has_valid_token():
        raise HTTPException(status_code=400, detail="未绑定 Google 账号")
    rc, out, err = run_cli(["ls", "-s", name, path], timeout=120)
    text = (out + "\n" + err).strip()
    # 解析 "d  name/  size" 之类的行，解析不出就原样返回
    entries = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("[colab]"):
            continue
        entries.append(line)
    return {"ok": rc == 0, "path": path, "entries": entries, "output": text[-4000:], "returncode": rc}


@app.post("/api/sessions/{name}/rm")
async def api_rm(name: str, request: Request):
    check_auth(request)
    if not has_valid_token():
        raise HTTPException(status_code=400, detail="未绑定 Google 账号")
    body = await request.json()
    path = str(body.get("path", "")).strip()
    if not path:
        raise HTTPException(status_code=400, detail="缺少 path")
    rc, out, err = run_cli(["rm", "-s", name, path], timeout=120)
    return {"ok": rc == 0, "output": (out + "\n" + err).strip()[-2000:], "returncode": rc}


# ---------------------------------------------------------------------------
# 会话历史日志（colab log）
# ---------------------------------------------------------------------------
@app.get("/api/logs")
def api_logs(request: Request, session: str = "", lines: int = 200, type: str = ""):
    check_auth(request)
    args = ["log"]
    if session:
        args += ["-s", session]
    if lines:
        args += ["-n", str(lines)]
    if type:
        args += ["-t", type]
    rc, out, err = run_cli(args, timeout=60)
    text = (out + "\n" + err).strip()
    history_dir = os.path.expanduser("~/.config/colab-cli/history")
    files, sessions = [], []
    if os.path.isdir(history_dir):
        for fn in sorted(os.listdir(history_dir)):
            if fn.endswith(".jsonl"):
                sessions.append(fn[:-6])
    return {"ok": rc == 0, "output": text[-8000:], "returncode": rc,
            "history_sessions": sessions, "history_dir": history_dir}


@app.get("/api/logs/{session}")
def api_log_one(session: str, request: Request, lines: int = 300, type: str = ""):
    check_auth(request)
    args = ["log", "-s", session, "-n", str(lines)]
    if type:
        args += ["-t", type]
    rc, out, err = run_cli(args, timeout=60)
    return {"ok": rc == 0, "session": session,
            "output": (out + "\n" + err).strip()[-20000:], "returncode": rc}


# ---------------------------------------------------------------------------
# 一次性任务（colab run：租实例 → 跑脚本 → 自动释放）
# ---------------------------------------------------------------------------
@app.post("/api/run")
async def api_run(request: Request):
    check_auth(request)
    if not has_valid_token():
        raise HTTPException(status_code=400, detail="未绑定 Google 账号")
    body = await request.json()
    code = str(body.get("code", "")).strip()
    if not code:
        raise HTTPException(status_code=400, detail="缺少 code")
    gpu = str(body.get("gpu", "cpu")).strip().lower()
    if gpu not in {"cpu", "t4", "l4", "a100", "h100", "g4"}:
        raise HTTPException(status_code=400, detail="gpu 取值非法")
    high_mem = bool(body.get("high_mem", False))
    keep = bool(body.get("keep", False))
    timeout_s = min(float(body.get("timeout", 120)), 280)
    name = str(body.get("name", "")).strip()
    import tempfile
    fd, py_path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(code)
        args = ["run", py_path, "--timeout", str(timeout_s)]
        if name:
            args += ["-s", name]
        if gpu != "cpu":
            args += ["--gpu", gpu.upper()]
        if high_mem:
            args.append("--high-mem")
        if keep:
            args.append("--keep")
        touch_activity()
        rc, out, err = run_cli(args, timeout=max(CLI_TIMEOUT_LONG, int(timeout_s) + 180))
        _invalidate_sessions_cache()
        return {"ok": rc == 0, "kept": keep, "name": name,
                "output": (out + "\n" + err).strip()[-8000:], "returncode": rc}
    finally:
        if os.path.exists(py_path):
            os.unlink(py_path)


# ---------------------------------------------------------------------------
# EasyColab 模型服务端部署（移植 colab_bridge.py cmd_wake 云端部分）
#
# EasyColab 的模型部署本体是 app.py：一个 Gradio WebUI + FastAPI 网关，跑在
# Colab VM 的 7860 端口。它的 FastAPI 路由（/minimax_api/*、/v1/*）会被
# gradio.live 隧道原样暴露，所以面板不需要直连 VM 端口，拿到隧道链接即可
# 全权代理模型管理、推理与 OpenAI/Anthropic 兼容网关。
#
# 部署流程 = 上传 app.py → 后台拉起 → 从日志抓 gradio.live 链接。
# 依赖（torch/gradio/DiffSynth）在 VM 镜像里已有；DiffSynth 缺失时才装。
# ---------------------------------------------------------------------------
EASYCOLAB_APP_LOCAL = os.environ.get(
    "EASYCOLAB_APP_PATH", "/opt/easycolab/app.py",
)
_EASYCOLAB_STATE_PATH = os.path.expanduser("~/.config/colab-cli/panel_easycolab.json")
_EASYCOLAB_STATE = {}          # session name -> {"gradio_url":..., "pid":..., "started_at":...}
_EASYCOLAB_LOCK = threading.Lock()
_EASYCOLAB_BOOT_LOG = "/content/easycolab_panel_server.log"

# VM 内引导脚本：清旧进程 → 确保 DiffSynth → 后台拉起 app.py → 等隧道链接
_EASYCOLAB_BOOT_CODE = r'''
import os, sys, re, time, shutil, subprocess

LOG = "__LOG_PATH__"
APP = "/content/app.py"

subprocess.run(["pkill", "-f", "python.*/content/app.py"], check=False)
time.sleep(1.5)
if os.path.exists(LOG):
    os.remove(LOG)

# DiffSynth-Studio 只在缺失时装（Colab 镜像通常已带；缺失会拖 3~10 分钟）
if not os.path.exists("/content/DiffSynth-Studio"):
    print("PANEL_DEPLOY:正在克隆 DiffSynth-Studio（首次部署会慢）...", flush=True)
    rc = subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/modelscope/DiffSynth-Studio.git", "/content/DiffSynth-Studio"],
        capture_output=True, text=True,
    )
    if rc.returncode == 0:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "/content/DiffSynth-Studio"], check=False)
    else:
        print("PANEL_WARN:DiffSynth 克隆失败，图像/视频模型可能不可用", flush=True)

with open(LOG, "w") as _f:
    _p = subprocess.Popen(
        [sys.executable, "-u", APP], cwd="/content",
        stdout=_f, stderr=subprocess.STDOUT,
        start_new_session=True,   # 脱离内核会话，exec 返回后继续跑
    )
print("PANEL_DEPLOY:已启动 app.py (pid %d)，等待隧道链接..." % _p.pid, flush=True)

_url_re = re.compile(r"https://[a-zA-Z0-9.-]+\.gradio\.live")
_found = None
_deadline = time.time() + 150
while time.time() < _deadline:
    time.sleep(2)
    if _p.poll() is not None:
        print("PANEL_DEPLOY_ERROR:app.py 已退出，rc=%s" % _p.returncode, flush=True)
        break
    try:
        _txt = open(LOG, errors="replace").read()
    except Exception:
        continue
    _m = _url_re.search(_txt)
    if _m:
        _found = _m.group(0)
        break
    if "Traceback (most recent call last)" in _txt:
        print("PANEL_DEPLOY_ERROR:启动异常，日志尾部如下\n" + _txt[-1500:], flush=True)
        break

if _found:
    print("PANEL_GRADIO_URL:" + _found, flush=True)
else:
    _tail = open(LOG, errors="replace").read()[-800:] if os.path.exists(LOG) else "(无日志)"
    print("PANEL_DEPLOY_ERROR:未在 150s 内拿到隧道链接\n" + _tail, flush=True)
'''


def _load_easycolab_state():
    with _EASYCOLAB_LOCK:
        if not _EASYCOLAB_STATE and os.path.exists(_EASYCOLAB_STATE_PATH):
            try:
                with open(_EASYCOLAB_STATE_PATH, "r", encoding="utf-8") as f:
                    _EASYCOLAB_STATE.update(json.load(f))
            except Exception:
                pass
        return _EASYCOLAB_STATE


def _save_easycolab_state():
    with _EASYCOLAB_LOCK:
        try:
            os.makedirs(os.path.dirname(_EASYCOLAB_STATE_PATH), exist_ok=True)
            with open(_EASYCOLAB_STATE_PATH, "w", encoding="utf-8") as f:
                json.dump(_EASYCOLAB_STATE, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


def _set_easycolab_state(name: str, **kw):
    _load_easycolab_state()
    with _EASYCOLAB_LOCK:
        rec = _EASYCOLAB_STATE.setdefault(name, {})
        rec.update(kw)
    _save_easycolab_state()


def easycolab_url(name: str) -> str:
    """已部署实例的 gradio.live 隧道地址；没部署返回空串。"""
    rec = _load_easycolab_state().get(name) or {}
    return str(rec.get("gradio_url") or "").rstrip("/")


def _easycolab_api(name: str, path: str, method: str = "GET", body=None, timeout: int = 120):
    """转发到实例上的 EasyColab FastAPI 网关（经 gradio.live 隧道）。"""
    base = easycolab_url(name)
    if not base:
        raise HTTPException(status_code=404, detail=f"实例 {name} 尚未部署 EasyColab 服务端")
    url = base + path
    try:
        if method.upper() == "GET":
            r = requests.get(url, timeout=timeout)
        else:
            r = requests.request(method.upper(), url, json=body, timeout=timeout)
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"隧道不可达: {str(e)[:300]}")
    try:
        return r.json()
    except ValueError:
        return {"ok": r.status_code < 400, "status": r.status_code, "text": r.text[:2000]}


def _easycolab_upload(name: str, local_path: str, remote_path: str) -> tuple:
    rc, out, err = run_cli(
        ["upload", "-s", name, local_path, remote_path], timeout=CLI_TIMEOUT_LONG
    )
    return rc, (out + "\n" + err).strip()


# ---------------------------------------------------------------------------
# EasyColab 模型服务端：部署 / 状态 / 模型 / 推理 代理
# ---------------------------------------------------------------------------
_EASYCOLAB_MODELS = [
    {"id": "minimax_h3", "name": "MiniMax-H3 (33B)", "size_gb": 134.1,
     "modality": "video_audio", "vram_gb": 64.5,
     "desc": "全模态音视频生成 · BF16 原生满血 · 32kHz 立体声"},
    {"id": "qwen38_27b", "name": "Qwen3.8-27B (BF16)", "size_gb": 51.8,
     "modality": "llm_vlm", "vram_gb": 53.0,
     "desc": "视觉理解与深度推理大模型 · BF16 原生满血"},
    {"id": "qwen38_27b_gguf", "name": "Qwen3.8-27B-GGUF", "size_gb": 18.7,
     "modality": "llm_vlm", "vram_gb": 20.8,
     "desc": "Unsloth UD-Q4_K_M + MTP 投机加速 · 支持 256K 上下文"},
    {"id": "qwen_image_21", "name": "Qwen-Image-2.1", "size_gb": 30.9,
     "modality": "image", "vram_gb": 24.0,
     "desc": "旗舰文生图与多图指令修图"},
    {"id": "z_image_turbo", "name": "Z-Image-Turbo", "size_gb": 30.6,
     "modality": "image", "vram_gb": 24.0,
     "desc": "8 步极速超清写实与中英文字生图"},
]


@app.get("/api/easycolab/models")
def api_easycolab_models(request: Request, name: str = ""):
    """可用模型清单 + 各型号显存需求。部署了服务端时并入远端自定义 GGUF 模型。"""
    check_auth(request)
    models = list(_EASYCOLAB_MODELS)
    if name and easycolab_url(name):
        try:
            remote = _easycolab_api(name, "/minimax_api/model_catalog", timeout=15)
            for m in remote.get("models") or []:
                if not m.get("is_custom"):
                    continue
                models.append({
                    "id": m["id"], "name": m.get("name") or m["id"],
                    "size_gb": m.get("approx_size_gb") or 0.0,
                    "modality": m.get("modality") or "llm",
                    "vram_gb": m.get("vram_gb") or m.get("approx_size_gb") or 0.0,
                    "desc": m.get("category") or "",
                    "is_custom": True,
                })
        except HTTPException:
            pass
    return {"ok": True, "models": models}


@app.get("/api/easycolab/status")
def api_easycolab_status(request: Request, name: str = ""):
    """部署状态 + 远端服务快照。未部署时 deployed=false。"""
    check_auth(request)
    rec = _load_easycolab_state()
    if not name:
        return {"ok": True, "deployed": {k: v for k, v in rec.items()}}
    url = easycolab_url(name)
    if not url:
        return {"ok": True, "name": name, "deployed": False}
    try:
        remote = _easycolab_api(name, "/minimax_api/status", timeout=20)
        return {"ok": True, "name": name, "deployed": True, "gradio_url": url,
                "record": rec.get(name, {}), "remote": remote}
    except HTTPException as e:
        return {"ok": True, "name": name, "deployed": True, "gradio_url": url,
                "record": rec.get(name, {}), "reachable": False, "error": str(e.detail)[:300]}


@app.post("/api/easycolab/deploy")
def api_easycolab_deploy(request: Request, name: str, hf_token: str = ""):
    """上传 app.py 到实例并在 VM 内后台拉起，返回 gradio.live 隧道地址。

    首次部署若需克隆 DiffSynth-Studio 会明显变慢（3~10 分钟）。
    """
    check_auth(request)
    if not has_valid_token():
        raise HTTPException(status_code=400, detail="未绑定 Google 账号")
    app_path = os.path.abspath(EASYCOLAB_APP_LOCAL)
    if not os.path.exists(app_path):
        raise HTTPException(
            status_code=400,
            detail=f"找不到 EasyColab 服务端源码：{app_path}。把 easycolab/app.py 放到该路径，"
                   f"或用 EASYCOLAB_APP_PATH 环境变量指定。",
        )
    touch_activity()
    rc, out = _easycolab_upload(name, app_path, "content/app.py")
    if rc != 0:
        raise HTTPException(status_code=502, detail=f"上传 app.py 失败: {out[-800:]}")

    boot_code = _EASYCOLAB_BOOT_CODE.replace("__LOG_PATH__", _EASYCOLAB_BOOT_LOG)
    if hf_token:
        boot_code = boot_code.replace(
            'APP = "/content/app.py"',
            'APP = "/content/app.py"\nos.environ["HF_TOKEN"] = %r' % hf_token,
        )
    rc, out, err = _run_code_on_session(name, boot_code, timeout_s=280)
    combined = out + ("\n" + err if err.strip() else "")
    gradio_url = ""
    for line in combined.splitlines():
        line = line.strip()
        if line.startswith("PANEL_GRADIO_URL:"):
            gradio_url = line[len("PANEL_GRADIO_URL:"):].strip()
            break
    if not gradio_url:
        return {"ok": False, "name": name, "output": combined[-3000:],
                "error": "未拿到 gradio.live 隧道链接（见 output）"}
    _set_easycolab_state(name, gradio_url=gradio_url, deployed_at=time.time())
    probe = {}
    try:
        probe = _easycolab_api(name, "/minimax_api/status", timeout=25)
    except HTTPException as e:
        probe = {"error": str(e.detail)[:200]}
    return {"ok": True, "name": name, "gradio_url": gradio_url, "status": probe,
            "output": combined[-1500:]}


@app.post("/api/easycolab/stop")
def api_easycolab_stop(request: Request, name: str):
    """停掉 VM 里的 app.py 进程（不释放实例）。"""
    check_auth(request)
    code = (
        'import subprocess\n'
        'r = subprocess.run(["pkill","-f","python.*/content/app.py"], capture_output=True)\n'
        'print("KILL_RC:%d" % r.returncode)\n'
    )
    rc, out, err = _run_code_on_session(name, code, timeout_s=60)
    # 只更新已有记录，不给不存在的会话凭空造状态
    if name in _load_easycolab_state():
        _set_easycolab_state(name, gradio_url="", stopped_at=time.time())
    return {"ok": True, "name": name, "output": (out + err).strip()[-800:]}


@app.post("/api/easycolab/select-model")
async def api_easycolab_select(name: str, request: Request):
    check_auth(request)
    body = await request.json()
    return _easycolab_api(name, "/minimax_api/select_model", "POST", {"model_id": body.get("model_id")})


@app.post("/api/easycolab/download-model")
async def api_easycolab_download(name: str, request: Request):
    """下载模型权重（后台线程，用 status 轮询进度）。"""
    check_auth(request)
    body = await request.json()
    touch_activity()
    return _easycolab_api(name, "/minimax_api/download_model", "POST", {
        "model_id": body.get("model_id"), "hf_token": body.get("hf_token", ""),
    }, timeout=60)


@app.post("/api/easycolab/load-model")
async def api_easycolab_load(name: str, request: Request):
    """加载模型进显存（多模型共驻，显存不足时 LRU 驱逐）。

    远端 load_model_to_vram_core 在线程里跑，失败原因（无 CUDA、权重不全）直接
    被丢弃、状态也不会更新，前端只会看到 task_message 不变。所以这里先查一次
    状态做前置校验，把明确原因直接回给调用方。
    """
    check_auth(request)
    body = await request.json()
    model_id = body.get("model_id") or ""
    touch_activity()
    try:
        st_path = "/minimax_api/status"
        if model_id:
            st_path += "?model_id=" + requests.utils.quote(model_id)
        st = _easycolab_api(name, st_path, timeout=25)
    except HTTPException:
        st = {}
    if isinstance(st, dict) and st.get("ok"):
        # GGUF 走 llama.cpp，能在纯 CPU 实例上跑；其余模型必须常驻显卡
        is_gguf = False
        if model_id:
            try:
                cat = _easycolab_api(name, "/minimax_api/model_catalog", timeout=25)
                is_gguf = any(
                    m.get("id") == model_id and m.get("backend") == "gguf"
                    for m in (cat or {}).get("models") or []
                )
            except HTTPException:
                is_gguf = False
        if not st.get("cuda_available") and not is_gguf:
            raise HTTPException(
                status_code=409,
                detail=f"实例 {name} 没有 GPU（{st.get('gpu_name') or 'CPU Mode'}），"
                       f"该模型需要显卡常驻。请新建 T4/L4/A100/H100/G4 实例；"
                       f"GGUF 量化模型可直接在 CPU 实例上运行。",
            )
        need = None
        if not is_gguf:
            need = next((m["vram_gb"] for m in _EASYCOLAB_MODELS if m["id"] == model_id), None)
        total = float(st.get("vram_total_gb") or 0)
        if need and total and need > total:
            raise HTTPException(
                status_code=409,
                detail=f"{model_id} 需要约 {need} GB 显存，实例只有 {total} GB。",
            )
        files = st.get("files") or []
        if files and st.get("total_count") and st.get("downloaded_count") == 0:
            raise HTTPException(
                status_code=409,
                detail=f"{st.get('selected_model_name') or model_id} 权重尚未下载（0/{st['total_count']} 个文件），"
                       f"先点「下载权重」。",
            )
    return _easycolab_api(name, "/minimax_api/load_model", "POST", {
        "model_id": model_id,
        "precision_mode": body.get("precision_mode", "bf16"),
        "offload_mode": body.get("offload_mode", "disk"),
        "hf_token": body.get("hf_token", ""),
        "context_length": body.get("context_length"),
    }, timeout=60)


@app.post("/api/easycolab/unload-model")
async def api_easycolab_unload(name: str, request: Request):
    check_auth(request)
    body = await request.json()
    return _easycolab_api(name, "/minimax_api/unload_model", "POST",
                          {"model_id": body.get("model_id")}, timeout=60)


@app.post("/api/easycolab/config")
async def api_easycolab_config(name: str, request: Request):
    """改上下文长度 / 闲置超时 / API Key 鉴权。"""
    check_auth(request)
    body = await request.json()
    out = {}
    if "context_length" in body:
        out["context"] = _easycolab_api(name, "/minimax_api/config_context", "POST",
                                        {"context_length": int(body["context_length"])})
    if "idle_timeout_minutes" in body:
        out["idle"] = _easycolab_api(name, "/minimax_api/config_idle", "POST",
                                     {"idle_timeout_minutes": int(body["idle_timeout_minutes"])})
    if "api_key" in body:
        out["api_key"] = _easycolab_api(name, "/minimax_api/config_api_key", "POST",
                                        {"api_key": str(body["api_key"])})
    if not out:
        raise HTTPException(status_code=400, detail="需要 context_length / idle_timeout_minutes / api_key 之一")
    return {"ok": True, **out}


@app.get("/api/easycolab/model-catalog")
def api_easycolab_model_catalog(request: Request, name: str):
    """远端完整模型清单（内置 + 自定义 GGUF），含下载/加载状态。未部署时回静态表。"""
    check_auth(request)
    if not name:
        raise HTTPException(status_code=400, detail="需要 name 参数指定实例")
    if not easycolab_url(name):
        return {"ok": True, "deployed": False, "models": _EASYCOLAB_MODELS}
    remote = _easycolab_api(name, "/minimax_api/model_catalog", timeout=25)
    out = dict(remote or {})
    out["deployed"] = True
    out["name"] = name
    return out


@app.get("/api/easycolab/hf-files")
def api_easycolab_hf_files(request: Request, name: str, repo: str, hf_token: str = ""):
    """探测 HuggingFace 仓库里的 .gguf 文件（量化级别 + 体积）。"""
    check_auth(request)
    if not name:
        raise HTTPException(status_code=400, detail="需要 name 参数指定实例")
    if not repo or "/" not in repo:
        raise HTTPException(status_code=400, detail="repo 需要形如 unsloth/Qwen2.5-7B-Instruct-GGUF")
    q = f"/minimax_api/hf_gguf_files?repo={quote(repo, safe='')}"
    if hf_token:
        q += f"&hf_token={quote(hf_token, safe='')}"
    try:
        return _easycolab_api(name, q, timeout=45)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"探测 HF 仓库失败: {str(e)[:300]}")


@app.post("/api/easycolab/register-model")
async def api_easycolab_register_model(name: str, request: Request):
    """把任意 HF GGUF 注册成远端可下载 / 可加载的模型条目。"""
    check_auth(request)
    body = await request.json()
    if not body.get("repo_id") or not body.get("filename"):
        raise HTTPException(status_code=400, detail="需要 repo_id 与 filename")
    return _easycolab_api(name, "/minimax_api/register_custom_model", "POST", {
        "repo_id": body.get("repo_id"),
        "filename": body.get("filename"),
        "name": body.get("name") or "",
        "mmproj_file": body.get("mmproj_file") or "",
        "vram_gb": float(body.get("vram_gb") or 0.0),
        "approx_size_gb": float(body.get("approx_size_gb") or 0.0),
        "context_length": int(body.get("context_length") or 32768),
    }, timeout=60)


@app.post("/api/easycolab/remove-custom-model")
async def api_easycolab_remove_custom_model(name: str, request: Request):
    """删除自定义模型注册（已下载权重保留，可重新注册）。"""
    check_auth(request)
    body = await request.json()
    mid = (body.get("model_id") or "").strip()
    if not mid:
        raise HTTPException(status_code=400, detail="需要 model_id")
    return _easycolab_api(name, "/minimax_api/remove_custom_model", "POST", {"model_id": mid}, timeout=60)


@app.post("/api/easycolab/generate")
async def api_easycolab_generate(name: str, request: Request):
    """提交生成任务（视频/音频/图像），返回 job_id；用 job 端点轮询。"""
    check_auth(request)
    body = await request.json()
    touch_activity()
    return _easycolab_api(name, "/minimax_api/generate", "POST", body, timeout=60)


@app.post("/api/easycolab/cancel")
def api_easycolab_cancel(name: str, request: Request):
    """停止当前生成任务。"""
    check_auth(request)
    return _easycolab_api(name, "/minimax_api/cancel", "POST", {}, timeout=60)


@app.get("/api/easycolab/job/{job_id}")
def api_easycolab_job(job_id: str, request: Request, name: str):
    check_auth(request)
    return _easycolab_api(name, f"/minimax_api/job/{job_id}", timeout=30)


@app.get("/api/easycolab/console")
def api_easycolab_console(request: Request, name: str):
    """EasyColab 的实时控制台（下载/推理进度行）。"""
    check_auth(request)
    return _easycolab_api(name, "/minimax_api/status", timeout=25)


@app.get("/api/easycolab/comfyui")
def api_easycolab_comfyui(name: str, request: Request):
    check_auth(request)
    return _easycolab_api(name, "/minimax_api/comfyui/status", timeout=30)


@app.post("/api/easycolab/comfyui/start")
async def api_easycolab_comfyui_start(name: str, request: Request):
    check_auth(request)
    body = await request.json()
    return _easycolab_api(name, "/minimax_api/comfyui/start", "POST",
                          {"unload_existing": bool(body.get("unload_existing")),
                           "force_restart": bool(body.get("force_restart"))}, timeout=60)


@app.post("/api/easycolab/comfyui/stop")
def api_easycolab_comfyui_stop(name: str, request: Request):
    check_auth(request)
    return _easycolab_api(name, "/minimax_api/comfyui/stop", "POST", {}, timeout=60)


# ---------------------------------------------------------------------------
# /v1 兼容网关透传
#
# 远端 app.py 自带 OpenAI（/v1/chat/completions、/v1/images/generations）与
# Anthropic（/v1/messages）兼容层，鉴权由它自己的 API Key 管。面板这边原样转发
# 请求体与上游返回，不做协议改写——客户端换 base_url 即可直连。
# ---------------------------------------------------------------------------
def _resolve_ec_name(name: str) -> str:
    """没传 name 时自动挑唯一已部署的实例，方便 OpenAI 客户端只配 base_url。"""
    if name:
        return name
    deployed = [k for k, v in _load_easycolab_state().items() if (v or {}).get("gradio_url")]
    if len(deployed) == 1:
        return deployed[0]
    if not deployed:
        raise HTTPException(status_code=404, detail="没有已部署 EasyColab 的实例")
    raise HTTPException(status_code=400,
                        detail=f"有多个已部署实例，请用 ?name= 指定：{', '.join(sorted(deployed))}")


async def _easycolab_forward_raw(name: str, path: str, request: Request, timeout: int = 900):
    name = _resolve_ec_name(name)
    base = easycolab_url(name)
    if not base:
        raise HTTPException(status_code=404, detail=f"实例 {name} 尚未部署 EasyColab 服务端")
    headers = {}
    for h in ("content-type", "authorization", "x-api-key", "anthropic-version",
              "anthropic-beta", "accept"):
        v = request.headers.get(h)
        if not v:
            continue
        # 面板口令不作为远端 API Key 透传，避免两边混用
        if h == "authorization" and v[7:].strip() == API_TOKEN and v.lower().startswith("bearer "):
            continue
        headers[h] = v
    body = None
    if request.method.upper() not in ("GET", "HEAD"):
        body = await request.body()
    try:
        r = requests.request(
            request.method.upper(), base + path, headers=headers, data=body,
            timeout=timeout, stream=True,
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"隧道不可达: {str(e)[:300]}")
    ctype = r.headers.get("content-type", "application/json")
    return StreamingResponse(
        r.iter_content(8192),
        status_code=r.status_code,
        media_type=ctype,
        headers={"X-Upstream-Status": str(r.status_code)},
    )


@app.get("/v1/models")
@app.get("/models")
async def api_v1_models(request: Request, name: str = ""):
    check_auth(request)
    return await _easycolab_forward_raw(name, "/v1/models", request, timeout=60)


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def api_v1_chat(request: Request, name: str = ""):
    check_auth(request)
    touch_activity()
    return await _easycolab_forward_raw(name, "/v1/chat/completions", request)


@app.post("/v1/images/generations")
async def api_v1_images(request: Request, name: str = ""):
    check_auth(request)
    touch_activity()
    return await _easycolab_forward_raw(name, "/v1/images/generations", request)


@app.post("/v1/messages")
async def api_v1_messages(request: Request, name: str = ""):
    check_auth(request)
    touch_activity()
    return await _easycolab_forward_raw(name, "/v1/messages", request)


# ---------------------------------------------------------------------------
# 云端 assignments（含本地表里没有的孤儿实例）
# ---------------------------------------------------------------------------
@app.get("/api/assignments")
def api_assignments(request: Request):
    check_auth(request)
    if not has_valid_token():
        raise HTTPException(status_code=400, detail="未绑定 Google 账号")
    try:
        from colab_cli.common import state
        local, assignments = _sync_local_sessions()
        local_eps = {s.endpoint for s in local.values()}
        items = []
        for a in assignments:
            items.append({
                "endpoint": a.endpoint,
                "accelerator": str(getattr(a, "accelerator", "")).split(".")[-1],
                "variant": str(getattr(a, "variant", "")).split(".")[-1],
                "managed": a.endpoint in local_eps,
                "local_name": next((s.name for s in local.values() if s.endpoint == a.endpoint), ""),
            })
        return {"ok": True, "assignments": items, "local_sessions": [s.name for s in local.values()]}
    except Exception as e:
        return {"ok": False, "error": str(e)[:400]}


@app.post("/api/assignments/{endpoint}/release")
def api_release_assignment(endpoint: str, request: Request):
    check_auth(request)
    if not has_valid_token():
        raise HTTPException(status_code=400, detail="未绑定 Google 账号")
    try:
        from colab_cli.common import state
        state.client.unassign(endpoint)
        _invalidate_sessions_cache()
        return {"ok": True, "released": endpoint}
    except Exception as e:
        return {"ok": False, "error": str(e)[:400]}


# ---------------------------------------------------------------------------
# Web UI
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    ensure_watchdog_started()
    ensure_heartbeat_started()
    with open(os.path.join(os.path.dirname(__file__), "ui.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.on_event("startup")
def _startup():
    ensure_watchdog_started()
    ensure_heartbeat_started()
