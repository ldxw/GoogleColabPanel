#!/usr/bin/env python3
"""MCP 接入层：把 Colab Panel 的能力暴露成 MCP 工具。

设计要点：
- 与面板同进程同端口：`/mcp` 与 `/api/*`、`/` 并列，不额外起服务。
- 工具内部用 httpx 的 ASGITransport 直接打进面板自己的 ASGI app，不经过网络端口，
  所以反代、改端口、绑 127.0.0.1 都不影响。
- 复用面板的 API_TOKEN 做鉴权，不引入第二套凭据。
- 危险操作（停止实例、删文件、DELETE 转发）必须显式 confirm=true，
  否则只返回预演结果。
"""
from __future__ import annotations

import asyncio
import base64
import hmac
import json
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Any, Optional
from urllib.parse import parse_qs

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.routing import Route

try:                                  # mcp 2.x 依赖 httpx2
    import httpx2 as _httpx
except ImportError:                   # 兼容只装了 httpx 的环境
    import httpx as _httpx

MCP_MOUNT = "/mcp"
MAX_INLINE_BYTES = 2 * 1024 * 1024

DEFAULT_INSTRUCTIONS = """\
Colab Panel MCP：管理 Google Colab 实例并调用其上部署的模型服务。

开始之前先调 panel_status 确认登录态与余额；实例会持续消耗 CU，用完调 panel_stop 释放。

工具分组：
- 账号：panel_status / panel_usage / panel_auth_url / panel_auth_code
- 实例：panel_sessions / panel_create / panel_stop / panel_stop_all / panel_ready
- 执行：panel_exec / panel_exec_async / panel_job / panel_run / panel_install
- 文件：panel_upload / panel_download / panel_ls / panel_rm
- 模型：panel_models / panel_deploy / panel_model / panel_generate
- 网关：panel_chat / panel_image
- 逃生口：panel_api（任意端点）

约定：
- panel_run 默认 CPU 实例；要显卡显式传 gpu="t4"/"l4"/"a100"/"h100"/"g4"。
- 停止/删除类操作需要 confirm=true，不带时只返回将影响的对象。
- panel_chat / panel_image 不传 name 时，面板会挑唯一已部署 EasyColab 的实例。
"""


# ---------------------------------------------------------------------------
# 进程内调用面板端点
# ---------------------------------------------------------------------------
class _PanelClient:
    def __init__(self, app: Any, token: str):
        self._app = app
        self._token = token
        self._transport = _httpx.ASGITransport(app=app)

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict] = None,
        body: Optional[dict] = None,
        files: Optional[dict] = None,
        headers: Optional[dict] = None,
        timeout: float = 120.0,
        auth: bool = True,
    ) -> tuple[int, Any]:
        hdrs = dict(headers or {})
        if auth:
            hdrs["Authorization"] = f"Bearer {self._token}"
        # ASGITransport 不走网络，httpx 的 timeout 管不到应用内部执行时间，
        # 用 wait_for 兜底，避免某个端点卡死拖住整个 MCP 会话。
        async with _httpx.AsyncClient(
            transport=self._transport, base_url="http://panel"
        ) as c:
            r = await asyncio.wait_for(
                c.request(
                    method.upper(), path, params=params, json=body, files=files, headers=hdrs
                ),
                timeout=timeout,
            )
            ctype = r.headers.get("content-type", "")
            if ctype.startswith("application/json"):
                try:
                    return r.status_code, r.json()
                except Exception:
                    return r.status_code, r.text
            if ctype.startswith("text/") or ctype == "":
                return r.status_code, r.text
            return r.status_code, r.content

    async def get(self, path: str, **kw) -> tuple[int, Any]:
        return await self.request("GET", path, **kw)

    async def post(self, path: str, **kw) -> tuple[int, Any]:
        return await self.request("POST", path, **kw)

    async def delete(self, path: str, **kw) -> tuple[int, Any]:
        return await self.request("DELETE", path, **kw)


def _brief(payload: Any, limit: int = 6000) -> str:
    """把返回压成适合放进上下文的短文本。"""
    if isinstance(payload, (dict, list)):
        text = json.dumps(payload, ensure_ascii=False, indent=1)
    else:
        text = str(payload)
    if len(text) > limit:
        return text[:limit] + f"\n…（截断，原始 {len(text)} 字符）"
    return text


def _err(status: int, payload: Any) -> str:
    detail = payload.get("detail") if isinstance(payload, dict) else None
    return f"HTTP {status}: {_brief(detail if detail is not None else payload, 1500)}"


def _sessions(body: Any) -> list:
    if isinstance(body, dict) and isinstance(body.get("sessions"), list):
        return body["sessions"]
    return []


# ---------------------------------------------------------------------------
# 构建 MCP server
# ---------------------------------------------------------------------------
def build_mcp_server(
    panel_app: Any, token: str, instructions: str = DEFAULT_INSTRUCTIONS
) -> MCPServer:
    mcp = MCPServer("colab-panel", instructions=instructions)
    api = _PanelClient(panel_app, token)

    # ---------------- 账号 ----------------

    @mcp.tool()
    async def panel_status() -> str:
        """面板总览：连通性、Google 登录态、CU 余额 / 消耗速率、活跃实例数、闲置超时与心跳配置。"""
        out: dict[str, Any] = {}
        st, body = await api.get("/api/ping", timeout=30)
        out["ping"] = body if st == 200 else _err(st, body)
        st, body = await api.get("/api/auth/status", timeout=90)
        out["account"] = body if st == 200 else _err(st, body)
        st, body = await api.get("/api/settings", timeout=30)
        out["settings"] = body if st == 200 else _err(st, body)
        return _brief(out)

    @mcp.tool()
    async def panel_usage() -> str:
        """账号用量：CU 余额、每小时消耗、活跃实例数，以及各实例的估算花费。"""
        st, body = await api.get("/api/usage", timeout=90)
        return _brief(body) if st == 200 else _err(st, body)

    @mcp.tool()
    async def panel_auth_url() -> str:
        """生成 Google OAuth 授权链接。

        链接需要人在浏览器里打开并登录，把授权码用 panel_auth_code 提交。
        这一步无法由 agent 独立完成。
        """
        st, body = await api.get("/api/auth/url", timeout=60)
        if st != 200:
            return _err(st, body)
        url = body.get("auth_url", "") if isinstance(body, dict) else ""
        return (
            f"授权链接（需要人打开并登录 Google）：\n{url}\n\n"
            "登录后把页面显示的授权码交给 panel_auth_code。"
        )

    @mcp.tool()
    async def panel_auth_code(code: str) -> str:
        """提交 OAuth 授权码，完成 Google 账号绑定。

        Args:
            code: 授权页显示的授权码
        """
        st, body = await api.post("/api/auth/code", body={"code": code}, timeout=120)
        return _brief(body) if st == 200 else _err(st, body)

    # ---------------- 实例 ----------------

    @mcp.tool()
    async def panel_sessions(refresh: bool = False) -> str:
        """列出本地会话与云端实例（含未纳管的孤儿实例）。

        Args:
            refresh: true 时绕过 3 秒缓存，直接查云端
        """
        st, body = await api.get(
            "/api/sessions", params={"refresh": 1 if refresh else 0}, timeout=120
        )
        return _brief(body) if st == 200 else _err(st, body)

    @mcp.tool()
    async def panel_create(
        gpu: str = "cpu",
        name: str = "",
        high_mem: bool = False,
        reuse: bool = True,
    ) -> str:
        """新建 Colab 实例（分配需 1~3 分钟，会持续消耗 CU）。

        Args:
            gpu: cpu / t4 / l4 / a100 / h100 / g4
            name: 实例名，留空自动生成
            high_mem: 是否申请大内存机型
            reuse: 默认 true，已有同名同型号实例时直接复用，不重复开新实例
        """
        st, body = await api.post(
            "/api/sessions",
            body={"gpu": gpu, "name": name, "high_mem": high_mem, "reuse": reuse},
            timeout=420,
        )
        return _brief(body) if st == 200 else _err(st, body)

    @mcp.tool()
    async def panel_stop(name: str, confirm: bool = False) -> str:
        """释放指定实例，停止计费。不可逆。

        Args:
            name: 实例名
            confirm: 必须为 true 才执行；false 时只回显将被释放的实例
        """
        if not confirm:
            st, body = await api.get("/api/sessions", timeout=120)
            target = next((s for s in _sessions(body) if s.get("name") == name), None)
            return (
                f"将释放实例 `{name}`（不可逆，CU 停止计费）。\n"
                f"当前信息：{_brief(target or '未在会话列表中找到同名实例', 800)}\n\n"
                "确认后重新调用并传 confirm=true。"
            )
        st, body = await api.delete(f"/api/sessions/{name}", timeout=360)
        return _brief(body) if st == 200 else _err(st, body)

    @mcp.tool()
    async def panel_stop_all(confirm: bool = False) -> str:
        """释放全部实例（含云端孤儿实例）。不可逆。

        Args:
            confirm: 必须为 true 才执行；false 时只列出将受影响的实例
        """
        if not confirm:
            st, body = await api.get("/api/sessions", timeout=120)
            names = [s.get("name") for s in _sessions(body) if s.get("name")]
            return (
                f"将释放全部 {len(names)} 个实例：{', '.join(names) or '（无）'}\n"
                "确认后重新调用并传 confirm=true。"
            )
        st, body = await api.post("/api/stop-all", timeout=600)
        return _brief(body) if st == 200 else _err(st, body)

    @mcp.tool()
    async def panel_ready(name: str) -> str:
        """探测实例是否就绪（区分「已销毁」「初始化中」「就绪」）。

        Args:
            name: 实例名
        """
        st, body = await api.post(f"/api/sessions/{name}/ready", body={}, timeout=90)
        return _brief(body) if st == 200 else _err(st, body)

    # ---------------- 执行 ----------------

    @mcp.tool()
    async def panel_exec(name: str, code: str, timeout: float = 120) -> str:
        """在实例的 Python 内核里同步执行代码并返回输出（最多保留末尾 5000 字符）。

        Args:
            name: 实例名
            code: Python 源码
            timeout: 秒，上限 280
        """
        t = min(float(timeout), 280)
        st, body = await api.post(
            f"/api/sessions/{name}/exec",
            body={"code": code, "timeout": t},
            timeout=t + 60,
        )
        return _brief(body) if st == 200 else _err(st, body)

    @mcp.tool()
    async def panel_exec_async(name: str, code: str, timeout: float = 120) -> str:
        """后台执行代码，立刻返回 job_id，用 panel_job 轮询输出。适合长任务。

        Args:
            name: 实例名
            code: Python 源码
            timeout: 秒，上限 280
        """
        st, body = await api.post(
            f"/api/sessions/{name}/exec-async",
            body={"code": code, "timeout": timeout},
            timeout=60,
        )
        if st != 200:
            return _err(st, body)
        job_id = body.get("job_id") if isinstance(body, dict) else None
        return f"job_id={job_id} 实例={name}\n用 panel_job 轮询该 id。"

    @mcp.tool()
    async def panel_job(job_id: str, from_line: int = 0) -> str:
        """拉取后台任务的状态与增量输出。

        Args:
            job_id: panel_exec_async 返回的 id
            from_line: 从第几行开始取，用上次返回的 next_line 续拉
        """
        st, body = await api.get(
            f"/api/jobs/{job_id}", params={"from_line": from_line}, timeout=60
        )
        return _brief(body, 12000) if st == 200 else _err(st, body)

    @mcp.tool()
    async def panel_run(
        code: str,
        gpu: str = "cpu",
        name: str = "",
        high_mem: bool = False,
        keep: bool = False,
        timeout: float = 120,
    ) -> str:
        """一次性跑完：按需建实例 → 执行代码 → 默认跑完即释放。

        Args:
            code: Python 源码
            gpu: cpu / t4 / l4 / a100 / h100 / g4，默认 cpu
            name: 实例名，留空自动生成
            high_mem: 是否申请大内存机型
            keep: true 时保留实例供后续复用；false 跑完即释放
            timeout: 秒，上限 280
        """
        st, body = await api.post(
            "/api/run",
            body={
                "code": code, "gpu": gpu, "name": name,
                "high_mem": high_mem, "keep": keep, "timeout": timeout,
            },
            timeout=900,
        )
        return _brief(body, 8000) if st == 200 else _err(st, body)

    @mcp.tool()
    async def panel_install(name: str, packages: str) -> str:
        """在实例内核里远程 pip 安装依赖。

        Args:
            name: 实例名
            packages: 空格分隔的包名，如 "transformers accelerate"
        """
        st, body = await api.post(
            f"/api/sessions/{name}/install",
            body={"packages": packages.split()},
            timeout=360,
        )
        return _brief(body, 8000) if st == 200 else _err(st, body)

    # ---------------- 文件 ----------------

    @mcp.tool()
    async def panel_upload(
        name: str,
        local_path: str = "",
        content: str = "",
        remote_path: str = "content/",
    ) -> str:
        """把文件上传到实例。

        二选一：local_path 指向面板容器内已存在的文件，或 content 直接给文本。

        Args:
            name: 实例名
            local_path: 面板侧可取到的文件路径
            content: 直接作为文件内容的文本
            remote_path: 远端目标目录，默认 content/
        """
        tmp = None
        try:
            if local_path:
                if not os.path.isfile(local_path):
                    return f"本地文件不存在：{local_path}"
                path, filename = local_path, os.path.basename(local_path)
            elif content:
                fd, tmp = tempfile.mkstemp(suffix=".txt")
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                path, filename = tmp, "payload.txt"
            else:
                return "需要 local_path 或 content 之一"
            with open(path, "rb") as f:
                data = f.read()
            st, body = await api.post(
                f"/api/sessions/{name}/upload",
                params={"remote_path": remote_path},
                files={"file": (filename, data, "application/octet-stream")},
                timeout=900,
            )
            return _brief(body) if st == 200 else _err(st, body)
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)

    @mcp.tool()
    async def panel_download(name: str, remote_path: str, inline: bool = False) -> str:
        """从实例下载文件。

        默认存到面板侧 MCP_DOWNLOAD_DIR（默认 /data/mcp-downloads/）并返回路径与大小；
        inline=true 且文件不超过 2 MB 时改为返回 base64（会占用大量上下文，按需开启）。

        Args:
            name: 实例名
            remote_path: 远端文件路径
            inline: 小文件直接内联返回
        """
        st, body = await api.get(
            f"/api/sessions/{name}/download",
            params={"remote_path": remote_path},
            timeout=900,
        )
        if st != 200:
            return _err(st, body)
        if not isinstance(body, (bytes, bytearray)):
            return _brief(body)
        data = bytes(body)
        base = os.environ.get("MCP_DOWNLOAD_DIR", "/data/mcp-downloads")
        saved = ""
        try:
            os.makedirs(base, exist_ok=True)
            dest = os.path.join(base, os.path.basename(remote_path) or "download.bin")
            with open(dest, "wb") as f:
                f.write(data)
            saved = dest
        except Exception as e:
            saved = f"（保存失败：{e}）"
        head = f"远端 {remote_path} → {len(data)} 字节\n面板侧副本：{saved}"
        if inline:
            if len(data) > MAX_INLINE_BYTES:
                return head + "\n文件超过 2 MB，未内联。"
            return head + "\n\nbase64:\n" + base64.b64encode(data).decode()
        return head + "\n需要内容时重新调用并传 inline=true（限 2 MB）。"

    @mcp.tool()
    async def panel_ls(name: str, path: str = "content") -> str:
        """列出实例内的目录。

        Args:
            name: 实例名
            path: 远端目录，默认 content
        """
        st, body = await api.get(
            f"/api/sessions/{name}/ls", params={"path": path}, timeout=240
        )
        return _brief(body) if st == 200 else _err(st, body)

    @mcp.tool()
    async def panel_rm(name: str, path: str, confirm: bool = False) -> str:
        """删除实例内的文件或目录。不可逆。

        Args:
            name: 实例名
            path: 远端路径
            confirm: 必须为 true 才执行
        """
        if not confirm:
            return f"将删除 {name}:{path}（不可逆）。确认后重新调用并传 confirm=true。"
        st, body = await api.post(
            f"/api/sessions/{name}/rm", body={"path": path}, timeout=240
        )
        return _brief(body) if st == 200 else _err(st, body)

    # ---------------- 模型 ----------------

    @mcp.tool()
    async def panel_models(name: str = "") -> str:
        """列出模型目录（内置 + 远端注册的自定义 GGUF），含显存需求。

        Args:
            name: 实例名，传了就并入该实例上注册的自定义模型
        """
        st, body = await api.get(
            "/api/easycolab/models", params={"name": name}, timeout=90
        )
        return _brief(body) if st == 200 else _err(st, body)

    @mcp.tool()
    async def panel_deploy(name: str, hf_token: str = "") -> str:
        """把 EasyColab 服务端部署到实例并拉起，返回 gradio.live 隧道地址。

        首次部署可能 3~10 分钟。

        Args:
            name: 实例名
            hf_token: 访问 gated 仓库时用的 HuggingFace token
        """
        st, body = await api.post(
            "/api/easycolab/deploy",
            params={"name": name, "hf_token": hf_token},
            body={},
            timeout=1200,
        )
        return _brief(body, 8000) if st == 200 else _err(st, body)

    @mcp.tool()
    async def panel_model(
        name: str,
        action: str,
        model_id: str = "",
        context_length: int = 0,
        idle_timeout_minutes: int = 0,
        api_key: Optional[str] = None,
        hf_token: str = "",
    ) -> str:
        """模型工作台操作。

        Args:
            name: 实例名
            action: select / download / load / unload / status / config / comfyui-start / comfyui-stop
            model_id: 目标模型 id
            context_length: action=config 时设置的上下文长度
            idle_timeout_minutes: action=config 时设置的闲置回收分钟数，0 表示不传
            api_key: action=config 时设置的网关 API Key；传空串表示关闭鉴权，不传则保持原值
            hf_token: 下载 gated 仓库时用
        """
        act = action.strip().lower()
        base = "/api/easycolab"
        if act == "status":
            st, body = await api.get(f"{base}/status", params={"name": name}, timeout=120)
        elif act == "select":
            if not model_id:
                return "action=select 需要 model_id"
            st, body = await api.post(
                f"{base}/select-model", params={"name": name},
                body={"model_id": model_id}, timeout=120,
            )
        elif act == "download":
            if not model_id:
                return "action=download 需要 model_id"
            st, body = await api.post(
                f"{base}/download-model", params={"name": name},
                body={"model_id": model_id, "hf_token": hf_token}, timeout=120,
            )
        elif act == "load":
            if not model_id:
                return "action=load 需要 model_id"
            st, body = await api.post(
                f"{base}/load-model", params={"name": name},
                body={"model_id": model_id, "hf_token": hf_token}, timeout=240,
            )
        elif act == "unload":
            if not model_id:
                return "action=unload 需要 model_id"
            st, body = await api.post(
                f"{base}/unload-model", params={"name": name},
                body={"model_id": model_id}, timeout=120,
            )
        elif act == "config":
            payload: dict[str, Any] = {}
            if context_length:
                payload["context_length"] = context_length
            if idle_timeout_minutes:
                payload["idle_timeout_minutes"] = idle_timeout_minutes
            if api_key is not None:
                payload["api_key"] = api_key
            if not payload:
                return "action=config 需要 context_length / idle_timeout_minutes / api_key 之一"
            st, body = await api.post(
                f"{base}/config", params={"name": name}, body=payload, timeout=120
            )
        elif act in ("comfyui-start", "comfyui-stop"):
            suffix = "start" if act.endswith("start") else "stop"
            st, body = await api.post(
                f"{base}/comfyui/{suffix}", params={"name": name}, body={}, timeout=240
            )
        else:
            return (
                f"未知 action：{action}（可选 select/download/load/unload/status/"
                "config/comfyui-start/comfyui-stop）"
            )
        return _brief(body, 8000) if st == 200 else _err(st, body)

    @mcp.tool()
    async def panel_generate(
        name: str,
        prompt: str,
        model_id: str = "",
        task_mode: str = "t2va",
        negative_prompt: str = "",
        width: int = 0,
        height: int = 0,
        num_inference_steps: int = 0,
        seed: int = -1,
        wait_seconds: float = 0,
        poll_interval: float = 3.0,
    ) -> str:
        """提交一次生成任务（文生图 / 文生视频 / 对话视觉），可选等待结果。

        远端请求模型必填 visual_prompt；task_mode=chat 时会自动把 prompt 包成
        chat_messages=[{"role":"user","content":prompt}]。

        Args:
            name: 实例名
            prompt: 正向提示词
            model_id: 目标模型 id，留空用实例当前选中模型
            task_mode: t2va（视频）/ t2i（文生图）/ i2i_edit（修图）/ fl2va（首尾帧）/ chat（LLM）
            negative_prompt: 负向提示词
            width: 宽，0 表示用远端默认
            height: 高，0 表示用远端默认
            num_inference_steps: 采样步数，0 表示用远端默认
            seed: 随机种子，-1 表示用远端默认
            wait_seconds: 0 表示立即返回 job_id；大于 0 时轮询到完成或超时
            poll_interval: 轮询间隔秒
        """
        body: dict[str, Any] = {"visual_prompt": prompt, "task_mode": task_mode}
        if model_id:
            body["model_id"] = model_id
        if negative_prompt:
            body["negative_prompt"] = negative_prompt
        if width:
            body["width"] = width
        if height:
            body["height"] = height
        if num_inference_steps:
            body["num_inference_steps"] = num_inference_steps
        if seed >= 0:
            body["seed"] = seed
        if task_mode == "chat":
            body["chat_messages"] = [{"role": "user", "content": prompt}]

        st, submitted = await api.post(
            "/api/easycolab/generate", params={"name": name}, body=body, timeout=120
        )
        if st != 200:
            return _err(st, submitted)
        job_id = None
        if isinstance(submitted, dict):
            job_id = (
                submitted.get("job_id")
                or submitted.get("id")
                or (submitted.get("job") or {}).get("job_id")
            )
        if not job_id or job_id == "none":
            return _brief(submitted, 8000)
        if wait_seconds <= 0:
            return (
                f"job_id={job_id} 实例={name}\n"
                "用 panel_generate 之外的 panel_api GET /api/easycolab/job/<job_id>?name=<实例> "
                "轮询，或下次传 wait_seconds。"
            )

        deadline = asyncio.get_running_loop().time() + min(float(wait_seconds), 900)
        last: Any = None
        while asyncio.get_running_loop().time() < deadline:
            st, last = await api.get(
                f"/api/easycolab/job/{job_id}", params={"name": name}, timeout=90
            )
            if st == 200 and isinstance(last, dict):
                status = str(last.get("status") or "").lower()
                if status in ("completed", "done", "success", "failed", "error", "cancelled"):
                    return _brief(last, 8000)
            await asyncio.sleep(max(0.5, poll_interval))
        return f"job_id={job_id} 等待超时，最后一次状态：\n{_brief(last, 4000)}"

    # ---------------- 网关 ----------------

    @mcp.tool()
    async def panel_chat(
        messages: str,
        name: str = "",
        model: str = "",
        remote_api_key: str = "",
        max_tokens: int = 1024,
    ) -> str:
        """调用实例上的 OpenAI 兼容对话接口。

        Args:
            messages: JSON 字符串，如 [{"role":"user","content":"hi"}]
            name: 实例名；留空时面板挑唯一已部署 EasyColab 的实例
            model: 模型 id，留空由远端决定
            remote_api_key: 远端 EasyColab 设置的 API Key（没设就留空）
            max_tokens: 最大输出 token
        """
        try:
            msgs = json.loads(messages)
        except Exception as e:
            return f"messages 不是合法 JSON：{e}"
        if not isinstance(msgs, list):
            return "messages 必须是数组"
        payload: dict[str, Any] = {"messages": msgs, "max_tokens": max_tokens}
        if model:
            payload["model"] = model
        headers = {"x-api-key": remote_api_key} if remote_api_key else None
        st, body = await api.request(
            "POST", "/v1/chat/completions", params={"name": name},
            body=payload, headers=headers, timeout=900,
        )
        return _brief(body, 12000) if st == 200 else _err(st, body)

    @mcp.tool()
    async def panel_image(
        prompt: str,
        name: str = "",
        model: str = "",
        size: str = "1024x1024",
        n: int = 1,
        remote_api_key: str = "",
    ) -> str:
        """调用实例上的 OpenAI 兼容文生图接口。

        Args:
            prompt: 提示词
            name: 实例名；留空自动挑唯一已部署实例
            model: 模型 id
            size: 尺寸，如 1024x1024
            n: 生成张数
            remote_api_key: 远端 EasyColab 的 API Key
        """
        payload: dict[str, Any] = {"prompt": prompt, "size": size, "n": n}
        if model:
            payload["model"] = model
        headers = {"x-api-key": remote_api_key} if remote_api_key else None
        st, body = await api.request(
            "POST", "/v1/images/generations", params={"name": name},
            body=payload, headers=headers, timeout=900,
        )
        return _brief(body, 12000) if st == 200 else _err(st, body)

    # ---------------- 逃生口 ----------------

    @mcp.tool()
    async def panel_api(
        method: str,
        path: str,
        body: str = "",
        confirm: bool = False,
    ) -> str:
        """直接调用面板任意端点，覆盖未单独封装的能力。

        常用于：/api/costs、/api/logs、/api/assignments、/api/easycolab/hf-files、
        /api/easycolab/register-model、/api/easycolab/cancel、/api/sessions/{name}/web-url、
        /api/sessions/{name}/restart-kernel 等。

        Args:
            method: GET / POST / DELETE / PUT
            path: 以 / 开头的路径，可带查询串
            body: JSON 字符串，非 GET 时使用
            confirm: DELETE / PUT 或 path 含 rm/stop/release/remove 时必须为 true
        """
        m = method.strip().upper()
        p = path.strip()
        if not p.startswith("/"):
            return "path 必须以 / 开头"
        risky = m in ("DELETE", "PUT") or any(
            k in p for k in ("/rm", "/stop", "/release", "/remove")
        )
        if risky and not confirm:
            return f"将对 {m} {p} 执行破坏性操作。确认后重新调用并传 confirm=true。"
        payload = None
        if m not in ("GET", "HEAD") and body.strip():
            try:
                payload = json.loads(body)
            except Exception as e:
                return f"body 不是合法 JSON：{e}"
        st, resp = await api.request(m, p, body=payload, timeout=900)
        return _brief(resp, 12000) if st < 400 else _err(st, resp)

    return mcp


# ---------------------------------------------------------------------------
# 挂载到现有的 FastAPI app
# ---------------------------------------------------------------------------
class _TokenGuard:
    """ASGI 层鉴权，避免 BaseHTTPMiddleware 破坏 SSE 流。"""

    def __init__(self, app: Any, token: str):
        self.app = app
        self.token = token

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if scope.get("method", "").upper() == "OPTIONS":
            # 浏览器端 MCP 客户端会发预检；这里直接放行，不碰令牌
            await send({
                "type": "http.response.start",
                "status": 204,
                "headers": [
                    (b"access-control-allow-origin", b"*"),
                    (b"access-control-allow-methods", b"GET, POST, DELETE, OPTIONS"),
                    (b"access-control-allow-headers", b"authorization, content-type, mcp-session-id, mcp-protocol-version"),
                    (b"access-control-expose-headers", b"mcp-session-id, mcp-protocol-version"),
                    (b"access-control-max-age", b"86400"),
                    (b"content-length", b"0"),
                ],
            })
            await send({"type": "http.response.body", "body": b""})
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers") or []}
        auth = headers.get("authorization", "")
        provided = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        if not provided:
            qs = parse_qs((scope.get("query_string") or b"").decode("latin-1"))
            provided = (qs.get("token") or [""])[0]
        if not provided or not hmac.compare_digest(provided, self.token):
            body = b'{"error":"unauthorized"}'
            await send({
                "type": "http.response.start",
                "status": 401,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode())],
            })
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)


class _MountShim:
    """把精确路径 /mcp 伪装成 /mcp/ 交给挂载子应用。

    Starlette 的 Mount 正则是 `/mcp/{path:path}`，不匹配 `/mcp` 本身，请求会先被
    redirect_slashes 转成 307。MCP 客户端未必跟随 307，这里直接改写 scope 抹平。
    """

    def __init__(self, app: Any, prefix: str):
        self.app = app
        self.prefix = prefix

    async def __call__(self, scope, receive, send):
        scope = dict(scope)
        scope["path"] = self.prefix + "/"
        scope["root_path"] = scope.get("root_path", "") + self.prefix
        await self.app(scope, receive, send)


def install_mcp(app: Any, token: str) -> bool:
    """把 MCP 端点挂到 app 上；没有开启 MCP 时返回 False。"""
    if os.environ.get("MCP_ENABLED", "1").strip().lower() in ("0", "false", "no", "off"):
        print("[mcp] 已通过 MCP_ENABLED=0 关闭")
        return False
    if not token:
        print("[mcp] 未拿到 API_TOKEN，跳过")
        return False

    mcp = build_mcp_server(app, token)

    allowed = [h.strip() for h in os.environ.get("MCP_ALLOWED_HOSTS", "").split(",") if h.strip()]
    if allowed:
        security = TransportSecuritySettings(
            allowed_hosts=allowed,
            allowed_origins=[f"https://{h}" for h in allowed],
        )
    else:
        # 令牌是唯一门槛；Host 校验关掉，否则内网 IP / 反代域名都会被拒
        security = TransportSecuritySettings(enable_dns_rebinding_protection=False)

    mcp_app = mcp.streamable_http_app(streamable_http_path="/", transport_security=security)

    # 原 lifespan 可能承担 @app.on_event 钩子（FastAPI 的 _DefaultLifespan 会跑
    # on_startup/on_shutdown），直接替换会让它们全部失效，必须包一层。
    prev_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(_app: Any):
        async with prev_lifespan(_app) as state:
            async with mcp.session_manager.run():
                yield state

    app.router.lifespan_context = lifespan

    guarded = _TokenGuard(mcp_app, token)
    # 精确路径 /mcp 单独注册：Mount 正则不含自身，会先落到 redirect_slashes 的 307，
    # 而部分 MCP 客户端不跟随 307。
    app.router.routes.append(
        Route(
            MCP_MOUNT,
            endpoint=_MountShim(guarded, MCP_MOUNT),
            methods=["GET", "HEAD", "POST", "DELETE", "OPTIONS"],
            include_in_schema=False,
        )
    )
    app.mount(MCP_MOUNT, guarded)
    print(f"[mcp] 已挂载 {MCP_MOUNT}（streamable-http，令牌复用 API_TOKEN）")
    return True
