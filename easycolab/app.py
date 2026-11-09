import os
import io
import gc
import re
import sys
import uuid
import time
import glob
import queue
import atexit
import socket
import base64
import subprocess
import threading
from collections import deque
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any

# ==============================================================================
# 实时终端进度与 tqdm 捕获器 (将 Colab 网页端的实时下载进度同步给 macOS App)
# ==============================================================================
class TerminalProgressMirror:
    def __init__(self, orig_stream, buffer_deque: deque, active_bars: dict, lock: threading.Lock):
        self.orig_stream = orig_stream
        self.buffer_deque = buffer_deque
        self.active_bars = active_bars
        self.lock = lock
        self._tls = threading.local()
        self._ansi_re = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

    def write(self, data: str):
        try:
            self.orig_stream.write(data)
        except Exception:
            pass
        if not data:
            return
        clean = self._ansi_re.sub("", data)
        partial = getattr(self._tls, "partial", "")
        with self.lock:
            for ch in clean:
                if ch in ("\r", "\n"):
                    line = partial.strip()
                    if line:
                        # 识别类似 "model-00001-of-00014.safetensors: 22%..." 或无前缀采样/VAE进度条 "54%|#####4 | 19/35 [08:31<07:11...]"
                        if "%" in line and ("it/s" in line or "B/s" in line or "step" in line or "|" in line):
                            pct_idx = line.find("%")
                            colon_idx = line.find(":")
                            if 0 < colon_idx < pct_idx:
                                key = line[:colon_idx].strip()
                                self.active_bars[key] = line
                            else:
                                stage_lbl = CURRENT_STAGE_BAR_LABEL
                                if stage_lbl:
                                    # 若在 VAE 解码等子阶段内部触发 tqdm，标注阶段名避免子阶段 100% 被误认为整条视频已生成完毕
                                    if line.startswith("100%"):
                                        self.active_bars["sampling"] = f"{stage_lbl}: 子阶段已完成，正在继续后续合成封装..."
                                    else:
                                        self.active_bars["sampling"] = f"{stage_lbl}: {line}"
                                else:
                                    self.active_bars["sampling"] = line
                        else:
                            self.buffer_deque.append(f"[{datetime.now().strftime('%H:%M:%S')}] {line}")
                    partial = ""
                else:
                    partial += ch
        self._tls.partial = partial

    def flush(self):
        try:
            self.orig_stream.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self.orig_stream, "isatty", lambda: False)()


CONSOLE_LOCK = threading.Lock()
CONSOLE_LINES: deque = deque(maxlen=120)
ACTIVE_TQDM_BARS: Dict[str, str] = {}
CURRENT_STAGE_BAR_LABEL: Optional[str] = None

if not isinstance(sys.stdout, TerminalProgressMirror):
    sys.stdout = TerminalProgressMirror(sys.__stdout__, CONSOLE_LINES, ACTIVE_TQDM_BARS, CONSOLE_LOCK)
if not isinstance(sys.stderr, TerminalProgressMirror):
    sys.stderr = TerminalProgressMirror(sys.__stderr__, CONSOLE_LINES, ACTIVE_TQDM_BARS, CONSOLE_LOCK)


def get_live_console_Lines() -> List[str]:
    with CONSOLE_LOCK:
        bars = list(ACTIVE_TQDM_BARS.values())[-10:]
        logs = list(CONSOLE_LINES)[-60:]
        return logs + (["─── 实时下载/采样进度流 ───"] + bars if bars else [])


# ==============================================================================
# 自动补齐音视频依赖 (防止出现 No module named 'av' 等错误)
# ==============================================================================
def ensure_runtime_dependencies():
    missing = []
    for pkg, mod in [
        ("av", "av"),
        ("hf_transfer", "hf_transfer"),
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"🔧 正在自动安装缺失的音视频依赖: {', '.join(missing)} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
        print("✅ 缺失依赖自动安装完成！")


ensure_runtime_dependencies()
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
# 强制 DiffSynth-Studio 所有内置下载走 HuggingFace 国际高速源 (彻底杜绝回退到国内 ModelScope 导致 1~2MB/s)
os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"
# 将 DiskMap 的句柄缓冲阈值扩大至 1TB，避免加载 33B 模型时频繁关闭/重开 safetensors 映射
os.environ["DIFFSYNTH_DISK_MAP_BUFFER_SIZE"] = str(10**12)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import shutil
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import gradio as gr
from PIL import Image


def _ensure_torchaudio_compatible():
    """防止 pip -U 升级覆盖 Colab 预装的 CUDA 定制版 torchaudio 导致 libtorchaudio.so ABI 不兼容崩溃。"""
    try:
        import torchaudio  # noqa: F401
        return
    except (OSError, ImportError) as e:
        for k in [m for m in list(sys.modules.keys()) if m == "torchaudio" or m.startswith("torchaudio.")]:
            sys.modules.pop(k, None)
        ver_raw = getattr(torch, "__version__", "")
        base_ver, plus, local_tag = ver_raw.partition("+")
        cu_tag = local_tag if plus and local_tag.startswith("cu") else ""
        print(f"检测到 torchaudio 动态库异常 ({e})，正在自动对齐 PyTorch ({ver_raw}) ABI...")
        repaired = False
        if base_ver and cu_tag:
            rc = subprocess.call([
                sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall",
                "--index-url", f"https://download.pytorch.org/whl/{cu_tag}",
                f"torchaudio=={base_ver}",
            ])
            if rc == 0:
                try:
                    import torchaudio  # noqa: F401
                    print(f"torchaudio 已自动修复对齐至 {torchaudio.__version__}！")
                    repaired = True
                except Exception:
                    for k in [m for m in list(sys.modules.keys()) if m == "torchaudio" or m.startswith("torchaudio.")]:
                        sys.modules.pop(k, None)
        if not repaired:
            try:
                import transformers.utils.import_utils as _iu
                _iu._torchaudio_available = False
            except Exception:
                pass


_ensure_torchaudio_compatible()


def _register_triton_causal_conv1d_shim():
    """当环境无预编译 causal_conv1d wheel (如 Python 3.13 + CUDA 13) 时，注册原生 Triton 融合算子以消除逐 token 多算子回退开销。"""
    if "causal_conv1d" in sys.modules:
        return
    try:
        import types
        import torch.nn.functional as F
        import triton
        import triton.language as tl

        @triton.jit
        def _triton_conv1d_update_k4(
            x_ptr, state_ptr, w_ptr, b_ptr, out_ptr,
            D,
            STATE_LEN: tl.constexpr,
            HAS_BIAS: tl.constexpr,
            APPLY_SILU: tl.constexpr,
            BLOCK_D: tl.constexpr,
        ):
            pid_b = tl.program_id(0)
            pid_d = tl.program_id(1)
            offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
            mask = offs_d < D

            x_idx = pid_b * D + offs_d
            x_raw = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
            x0 = x_raw.to(tl.float32)

            st_base = (pid_b * D + offs_d) * STATE_LEN
            st_proto = tl.load(state_ptr + st_base, mask=mask, other=0.0)
            s0 = tl.load(state_ptr + st_base + (STATE_LEN - 3), mask=mask, other=0.0).to(tl.float32)
            s1 = tl.load(state_ptr + st_base + (STATE_LEN - 2), mask=mask, other=0.0).to(tl.float32)
            s2 = tl.load(state_ptr + st_base + (STATE_LEN - 1), mask=mask, other=0.0).to(tl.float32)

            if STATE_LEN == 4:
                tl.store(state_ptr + st_base + 0, s0.to(st_proto.dtype), mask=mask)
                tl.store(state_ptr + st_base + 1, s1.to(st_proto.dtype), mask=mask)
                tl.store(state_ptr + st_base + 2, s2.to(st_proto.dtype), mask=mask)
                tl.store(state_ptr + st_base + 3, x0.to(st_proto.dtype), mask=mask)
            elif STATE_LEN == 3:
                tl.store(state_ptr + st_base + 0, s1.to(st_proto.dtype), mask=mask)
                tl.store(state_ptr + st_base + 1, s2.to(st_proto.dtype), mask=mask)
                tl.store(state_ptr + st_base + 2, x0.to(st_proto.dtype), mask=mask)

            w_base = offs_d * 4
            w0 = tl.load(w_ptr + w_base + 0, mask=mask, other=0.0).to(tl.float32)
            w1 = tl.load(w_ptr + w_base + 1, mask=mask, other=0.0).to(tl.float32)
            w2 = tl.load(w_ptr + w_base + 2, mask=mask, other=0.0).to(tl.float32)
            w3 = tl.load(w_ptr + w_base + 3, mask=mask, other=0.0).to(tl.float32)

            acc = s0 * w0 + s1 * w1 + s2 * w2 + x0 * w3
            if HAS_BIAS:
                b = tl.load(b_ptr + offs_d, mask=mask, other=0.0).to(tl.float32)
                acc += b
            if APPLY_SILU:
                acc = acc * tl.sigmoid(acc)
            tl.store(out_ptr + x_idx, acc.to(x_raw.dtype), mask=mask)

        def _shim_causal_conv1d_update(x, conv_state, weight, bias=None, activation=None, cache_seqlens=None, conv_state_indices=None):
            act_silu = activation in ("silu", "swish")
            if (
                x.is_cuda
                and cache_seqlens is None
                and conv_state_indices is None
                and x.ndim == 3
                and x.shape[2] == 1
                and weight.ndim == 2
                and weight.shape[1] == 4
                and conv_state.ndim == 3
                and conv_state.shape[2] in (3, 4)
                and conv_state.is_contiguous()
                and (activation is None or act_silu)
            ):
                x_c = x if x.is_contiguous() else x.contiguous()
                w_c = weight if weight.is_contiguous() else weight.contiguous()
                b_c = (bias if bias.is_contiguous() else bias.contiguous()) if bias is not None else w_c
                out = torch.empty_like(x_c)
                B, D, _ = x_c.shape
                grid = (B, triton.cdiv(D, 256))
                _triton_conv1d_update_k4[grid](
                    x_c, conv_state, w_c, b_c, out,
                    D,
                    STATE_LEN=conv_state.shape[2],
                    HAS_BIAS=bias is not None,
                    APPLY_SILU=act_silu,
                    BLOCK_D=256,
                )
                return out

            _, hidden_size, seq_len = x.shape
            state_len = conv_state.shape[-1]
            h_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)
            conv_state.copy_(h_new[:, :, -state_len:])
            out = F.conv1d(h_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)[:, :, -seq_len:]
            if act_silu:
                out = F.silu(out)
            return out.to(x.dtype)

        def _shim_causal_conv1d_fn(x, weight, bias=None, activation=None, **kwargs):
            _, hidden_size, seq_len = x.shape
            padding = weight.shape[-1] - 1
            out = F.conv1d(
                x.to(weight.dtype),
                weight=weight.unsqueeze(1),
                bias=bias,
                padding=padding,
                groups=hidden_size,
            )[:, :, :seq_len]
            if activation in ("silu", "swish"):
                out = F.silu(out)
            return out.to(x.dtype)

        mod = types.ModuleType("causal_conv1d")
        mod.__version__ = "1.5.0+triton"
        mod.causal_conv1d_update = _shim_causal_conv1d_update
        mod.causal_conv1d_fn = _shim_causal_conv1d_fn
        sys.modules["causal_conv1d"] = mod
    except Exception as e:
        print(f"[加速引擎] Triton causal_conv1d 注册跳过: {e}")


def _ensure_qwen_fast_kernels():
    """确保 Qwen3.8 线性注意力 Triton 算子 (fla-core) 与融合 causal_conv1d_update 算子真正生效。"""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    need_rebind = False
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule  # noqa: F401
    except Exception:
        print("正在安装 Qwen3.8 GatedDeltaNet 原生 Triton 算子 (fla-core + flash-linear-attention)...")
        for k in [m for m in list(sys.modules.keys()) if m == "fla" or m.startswith("fla.")]:
            sys.modules.pop(k, None)
        subprocess.call([sys.executable, "-m", "pip", "install", "-q", "einops"])
        subprocess.call([
            sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall",
            "fla-core", "flash-linear-attention",
        ])
        for k in [m for m in list(sys.modules.keys()) if m == "fla" or m.startswith("fla.")]:
            sys.modules.pop(k, None)
        need_rebind = True

    try:
        import causal_conv1d  # noqa: F401
    except ImportError:
        if sys.version_info >= (3, 13):
            _register_triton_causal_conv1d_shim()
            need_rebind = True
        else:
            rc = subprocess.call([
                sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                "--only-binary=:all:", "causal-conv1d",
            ])
            if rc != 0:
                _register_triton_causal_conv1d_shim()
                need_rebind = True

    if need_rebind and "transformers.models.qwen3_5.modeling_qwen3_5" in sys.modules:
        import importlib
        try:
            if "transformers.integrations.hub_kernels" in sys.modules:
                importlib.reload(sys.modules["transformers.integrations.hub_kernels"])
            importlib.reload(sys.modules["transformers.models.qwen3_5.modeling_qwen3_5"])
        except Exception:
            pass


def _optimize_qwen35_for_fast_inference(mdl, max_context_len: int = 32768):
    """
    针对 Qwen3.8-27B BF16 (64层混合 GatedDeltaNet + Attention) 的深度算子融合与全模型 CUDAGraph 解码引擎：
    1. 融合 Qwen3_5RMSNorm -> 原生 C++ F.rms_norm (消除每 token 1,700+ 次 Python 标量/张量调度)
    2. 融合 Qwen3_5MLP [gate_proj; up_proj] 单次 GEMV + FLA Triton SwiGLU
    3. 融合 Qwen3_5Attention [q_proj; k_proj; v_proj] 单次 GEMV + PyTorch GQA SDPA
    4. 融合 Qwen3_5GatedDeltaNet [qkv; z; b; a] 单次 GEMV + Triton g/beta + 原生 GQA FLA + Triton RMSNormGated
    5. 预分配持久化 StaticCache 捕获 CUDAGraph 极速解码，同时兼容超长上下文 (最高 256K) 动态混合缓存
    """
    try:
        import torch.nn as nn
        import torch.nn.functional as F
        import triton
        import triton.language as tl
        import transformers.models.qwen3_5.modeling_qwen3_5 as mq
        from transformers import StaticCache
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
        from fla.modules.activations import swiglu

        text_model = getattr(getattr(mdl, "model", None), "language_model", None)
        if text_model is None:
            return

        # 单卡权重已经在 GPU 上。device_map 留下的 AlignDevicesHook 会在每层
        # forward 里递归扫描混合缓存，NF4 比 BF16 慢的一大截来自这里。
        try:
            from accelerate.hooks import remove_hook_from_submodules
            remove_hook_from_submodules(mdl)
        except Exception as hook_err:
            print(f"[加速引擎] accelerate hook 拆除跳过: {hook_err}")

        @triton.jit
        def _qwen35_rmsnorm_gated_silu(x_ptr, z_ptr, w_ptr, out_ptr, D, BLOCK_D: tl.constexpr, eps):
            row = tl.program_id(0)
            offs = tl.arange(0, BLOCK_D)
            mask = offs < D
            x_raw = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0)
            x = x_raw.to(tl.float32)
            z = tl.load(z_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            var = tl.sum(x * x, axis=0) / D
            rstd = tl.rsqrt(var + eps)
            y = (x * rstd * w) * (z * tl.sigmoid(z))
            tl.store(out_ptr + row * D + offs, y.to(x_raw.dtype), mask=mask)

        @triton.jit
        def _qwen35_g_beta_fused(ba_ptr, neg_exp_A_ptr, dt_bias_ptr, g_ptr, beta_ptr, H, BLOCK_H: tl.constexpr):
            row = tl.program_id(0)
            offs = tl.arange(0, BLOCK_H)
            mask = offs < H
            b_raw = tl.load(ba_ptr + row * (2 * H) + offs, mask=mask, other=0.0)
            b = b_raw.to(tl.float32)
            a = tl.load(ba_ptr + row * (2 * H) + H + offs, mask=mask, other=0.0).to(tl.float32)
            neg_exp_A = tl.load(neg_exp_A_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            dt_b = tl.load(dt_bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            beta = tl.sigmoid(b)
            x = a + dt_b
            sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
            g = neg_exp_A * sp
            tl.store(beta_ptr + row * H + offs, beta.to(b_raw.dtype), mask=mask)
            tl.store(g_ptr + row * H + offs, g, mask=mask)

        # head_dim=256 时把整段 D 放进寄存器会溢出，解码被延迟绑死。
        # 按 BLOCK_D=32 分块，fp32 累加写到捕获前就分配好的全局缓冲。
        _decode_acc_holder = [None]

        def _decode_acc_buf(q):
            hq, hd = q.shape[1], q.shape[-1]
            buf = _decode_acc_holder[0]
            if buf is None or buf.shape != (hq, hd) or buf.device != q.device:
                buf = torch.empty((hq, hd), device=q.device, dtype=torch.float32)
                _decode_acc_holder[0] = buf
            return buf

        @triton.jit
        def _qwen35_gqa_decode_attn(
            q_ptr, k_ptr, v_ptr, cum_ptr, acc_ptr, out_ptr, scale, max_seq,
            D: tl.constexpr, GQA_GROUPS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
        ):
            pid_hq = tl.program_id(0)
            pid_hkv = pid_hq // GQA_GROUPS
            seq_len = tl.load(cum_ptr).to(tl.int32)
            acc_base = acc_ptr + pid_hq * D
            q_base = q_ptr + pid_hq * D
            k_base = k_ptr + pid_hkv * max_seq * D
            v_base = v_ptr + pid_hkv * max_seq * D
            q_dtype = tl.load(q_base).dtype
            for d0 in range(0, D, BLOCK_D):
                offs_d = d0 + tl.arange(0, BLOCK_D)
                tl.store(acc_base + offs_d, tl.zeros([BLOCK_D], dtype=tl.float32), mask=offs_d < D)
            m_i = -float("inf")
            l_i = 0.0
            for start_n in range(0, seq_len, BLOCK_N):
                offs_n = start_n + tl.arange(0, BLOCK_N)
                mask_n = offs_n < seq_len
                qk = tl.zeros([BLOCK_N], dtype=tl.float32)
                for d0 in range(0, D, BLOCK_D):
                    offs_d = d0 + tl.arange(0, BLOCK_D)
                    mask_d = offs_d < D
                    q_tile = tl.load(q_base + offs_d, mask=mask_d, other=0.0).to(tl.float32)
                    ptrs = offs_n[:, None] * D + offs_d[None, :]
                    mask_2d = mask_n[:, None] & mask_d[None, :]
                    k_blk = tl.load(k_base + ptrs, mask=mask_2d, other=0.0).to(tl.float32)
                    qk += tl.sum(k_blk * q_tile[None, :], axis=1)
                qk = qk * scale
                qk = tl.where(mask_n, qk, -float("inf"))
                m_ij = tl.max(qk, axis=0)
                m_new = tl.maximum(m_i, m_ij)
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(qk - m_new)
                l_i = l_i * alpha + tl.sum(tl.where(mask_n, p, 0.0), axis=0)
                for d0 in range(0, D, BLOCK_D):
                    offs_d = d0 + tl.arange(0, BLOCK_D)
                    mask_d = offs_d < D
                    ptrs = offs_n[:, None] * D + offs_d[None, :]
                    mask_2d = mask_n[:, None] & mask_d[None, :]
                    v_blk = tl.load(v_base + ptrs, mask=mask_2d, other=0.0).to(tl.float32)
                    acc_tile = tl.load(acc_base + offs_d, mask=mask_d, other=0.0)
                    acc_tile = acc_tile * alpha + tl.sum(p[:, None] * v_blk, axis=0)
                    tl.store(acc_base + offs_d, acc_tile, mask=mask_d)
                m_i = m_new
            for d0 in range(0, D, BLOCK_D):
                offs_d = d0 + tl.arange(0, BLOCK_D)
                mask_d = offs_d < D
                acc_tile = tl.load(acc_base + offs_d, mask=mask_d, other=0.0)
                tl.store(out_ptr + pid_hq * D + offs_d, (acc_tile / l_i).to(q_dtype), mask=mask_d)

        for mod in text_model.modules():
            if isinstance(mod, mq.Qwen3_5RMSNorm):
                mod._w1 = (1.0 + mod.weight.detach().float()).to(mod.weight.dtype).contiguous()
                mod._norm_shape = (mod.weight.shape[0],)

                def _fast_rmsnorm(self, x):
                    return F.rms_norm(x, self._norm_shape, self._w1, self.eps)

                mod.forward = _fast_rmsnorm.__get__(mod, type(mod))

            elif isinstance(mod, mq.Qwen3_5MLP):
                if (
                    isinstance(mod.gate_proj, nn.Linear)
                    and mod.gate_proj.bias is None
                    and mod.up_proj.bias is None
                    and mod.gate_proj.weight.dtype in (torch.bfloat16, torch.float16)
                ):
                    inter = mod.gate_proj.out_features
                    mod._gate_up_w = torch.cat([mod.gate_proj.weight.detach(), mod.up_proj.weight.detach()], dim=0).contiguous()
                    mod.gate_proj = None
                    mod.up_proj = None
                    mod._inter = inter

                    def _fast_mlp(self, x):
                        gu = F.linear(x, self._gate_up_w)
                        g, u = gu.split(self._inter, dim=-1)
                        return self.down_proj(swiglu(g, u))

                    mod.forward = _fast_mlp.__get__(mod, type(mod))

            elif isinstance(mod, mq.Qwen3_5Attention):
                # NF4 权重是 uint8，拼不成一份 BF16 QKV，但解码仍要走静态 Triton 核。
                # 预填继续用 SDPA，因为静态缓冲里有 padding，必须靠官方 mask。
                can_fuse_qkv = (
                    isinstance(mod.q_proj, nn.Linear)
                    and mod.q_proj.bias is None
                    and mod.k_proj.bias is None
                    and mod.v_proj.bias is None
                    and getattr(getattr(mod.q_proj, "weight", None), "dtype", None) in (torch.bfloat16, torch.float16)
                )
                if can_fuse_qkv:
                    qd, kd, vd = mod.q_proj.out_features, mod.k_proj.out_features, mod.v_proj.out_features
                    mod._qkv_w = torch.cat(
                        [mod.q_proj.weight.detach(), mod.k_proj.weight.detach(), mod.v_proj.weight.detach()], dim=0
                    ).contiguous()
                    mod._qkv_splits = (qd, kd, vd)
                    mod.q_proj = mod.k_proj = mod.v_proj = None

                def _fast_attn(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, self.head_dim)
                    if self.q_proj is None:
                        qkv = F.linear(hidden_states, self._qkv_w)
                        q_raw, k_raw, v_raw = torch.split(qkv, self._qkv_splits, dim=-1)
                    else:
                        q_raw = self.q_proj(hidden_states)
                        k_raw = self.k_proj(hidden_states)
                        v_raw = self.v_proj(hidden_states)
                    query_states, gate = torch.chunk(q_raw.view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
                    gate = gate.reshape(*input_shape, -1)
                    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
                    key_states = self.k_norm(k_raw.view(hidden_shape)).transpose(1, 2)
                    value_states = v_raw.view(hidden_shape).transpose(1, 2)
                    cos, sin = position_embeddings
                    query_states, key_states = mq.apply_rotary_pos_emb(query_states, key_states, cos, sin)
                    if past_key_values is not None:
                        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
                    cache_lyr = (
                        past_key_values.layers[self.layer_idx]
                        if (past_key_values is not None and hasattr(past_key_values, "layers"))
                        else None
                    )
                    use_static = (
                        input_shape == (1, 1)
                        and cache_lyr is not None
                        and hasattr(cache_lyr, "cumulative_length")
                        and torch.is_tensor(cache_lyr.cumulative_length)
                        and getattr(cache_lyr, "keys", None) is not None
                        and getattr(past_key_values, "_no_left_pad", True)
                    )
                    if use_static:
                        q_c = query_states.contiguous()
                        attn_output = torch.empty_like(q_c)
                        k_buf = cache_lyr.keys
                        v_buf = cache_lyr.values
                        hq = q_c.shape[1]
                        hkv = k_buf.shape[1]
                        _qwen35_gqa_decode_attn[(hq,)](
                            q_c,
                            k_buf,
                            v_buf,
                            cache_lyr.cumulative_length,
                            _decode_acc_buf(q_c),
                            attn_output,
                            self.scaling,
                            k_buf.shape[2],
                            D=self.head_dim,
                            GQA_GROUPS=hq // hkv,
                            BLOCK_N=32,
                            BLOCK_D=32,
                            num_warps=4,
                        )
                    else:
                        attn_output = F.scaled_dot_product_attention(
                            query_states,
                            key_states,
                            value_states,
                            attn_mask=attention_mask,
                            dropout_p=0.0,
                            scale=self.scaling,
                            enable_gqa=True,
                        )
                    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1)
                    return self.o_proj(attn_output * torch.sigmoid(gate)), None

                mod.forward = _fast_attn.__get__(mod, type(mod))

            elif isinstance(mod, mq.Qwen3_5GatedDeltaNet):
                if (
                    isinstance(mod.in_proj_qkv, nn.Linear)
                    and mod.in_proj_qkv.bias is None
                    and mod.in_proj_z.bias is None
                    and mod.in_proj_b.bias is None
                    and mod.in_proj_a.bias is None
                    and mod.in_proj_qkv.weight.dtype in (torch.bfloat16, torch.float16)
                ):
                    qkv_d = mod.in_proj_qkv.out_features
                    z_d = mod.in_proj_z.out_features
                    b_d = mod.in_proj_b.out_features
                    a_d = mod.in_proj_a.out_features
                    mod._qkvzba_w = torch.cat(
                        [
                            mod.in_proj_qkv.weight.detach(),
                            mod.in_proj_z.weight.detach(),
                            mod.in_proj_b.weight.detach(),
                            mod.in_proj_a.weight.detach(),
                        ],
                        dim=0,
                    ).contiguous()
                    mod.in_proj_qkv = mod.in_proj_z = mod.in_proj_b = mod.in_proj_a = None
                    mod._splits = (qkv_d, z_d, b_d + a_d)
                    mod._neg_exp_A = (-mod.A_log.detach().float().exp()).contiguous()
                    mod._dt_bias_f32 = mod.dt_bias.detach().float().contiguous()
                    mod._block_h = triton.next_power_of_2(b_d)
                    mod._block_d = triton.next_power_of_2(mod.head_v_dim)

                    def _fast_gdn(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
                        if attention_mask is not None:
                            hidden_states = mq.apply_mask_to_padding_states(hidden_states, attention_mask)
                        B, L, _ = hidden_states.shape
                        conv_state = None
                        rec_state = None
                        recording_past = False
                        if cache_params is not None:
                            if hasattr(cache_params, "get_conv_state"):
                                conv_state = cache_params.get_conv_state(self.layer_idx, state_idx=0)
                                rec_state = cache_params.get_recurrent_state(self.layer_idx, state_idx=0)
                                recording_past = cache_params.is_recording_past(self.layer_idx)
                            elif getattr(cache_params, "has_previous_state", False):
                                cs_list = getattr(cache_params, "conv_states", None)
                                rs_list = getattr(cache_params, "recurrent_states", None)
                                if isinstance(cs_list, (list, tuple)) and self.layer_idx < len(cs_list):
                                    conv_state = cs_list[self.layer_idx]
                                if isinstance(rs_list, (list, tuple)) and self.layer_idx < len(rs_list):
                                    rec_state = rs_list[self.layer_idx]

                        proj = F.linear(hidden_states, self._qkvzba_w)
                        mixed_qkv, z, ba = torch.split(proj, self._splits, dim=-1)
                        mixed_qkv = mixed_qkv.transpose(1, 2)

                        if conv_state is not None and L == 1 and not recording_past:
                            mixed_qkv = mq.causal_conv1d_update(
                                mixed_qkv, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation
                            )
                        else:
                            if cache_params is not None:
                                if hasattr(cache_params, "update_conv_state"):
                                    mixed_qkv = cache_params.update_conv_state(
                                        mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size
                                    )
                                else:
                                    cs_list = getattr(cache_params, "conv_states", None)
                                    if isinstance(cs_list, list) and self.layer_idx < len(cs_list):
                                        ksize = self.conv_kernel_size
                                        padded = F.pad(mixed_qkv, (ksize - mixed_qkv.shape[-1], 0)) if mixed_qkv.shape[-1] < ksize else mixed_qkv
                                        cs_list[self.layer_idx] = padded[..., -ksize:].contiguous()
                            mixed_qkv = mq.causal_conv1d_fn(
                                mixed_qkv, self.conv1d.weight.squeeze(1), self.conv1d.bias, activation=self.activation
                            )
                            if cache_params is not None:
                                mixed_qkv = mixed_qkv[:, :, -L:]

                        mixed_qkv = mixed_qkv.transpose(1, 2)
                        q, k, v = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
                        q = q.view(B, L, -1, self.head_k_dim)
                        k = k.view(B, L, -1, self.head_k_dim)
                        v = v.view(B, L, -1, self.head_v_dim)

                        H = self.num_v_heads
                        g_t = torch.empty((B, L, H), device=hidden_states.device, dtype=torch.float32)
                        beta_t = torch.empty((B, L, H), device=hidden_states.device, dtype=hidden_states.dtype)
                        _qwen35_g_beta_fused[(B * L,)](
                            ba.contiguous(), self._neg_exp_A, self._dt_bias_f32, g_t, beta_t, H, BLOCK_H=self._block_h
                        )

                        if self.num_v_heads // self.num_k_heads > 1:
                            rep = self.num_v_heads // self.num_k_heads
                            q = q.repeat_interleave(rep, dim=2)
                            k = k.repeat_interleave(rep, dim=2)
                        if rec_state is not None and L == 1:
                            core_out, last_state = fused_recurrent_gated_delta_rule(
                                q, k, v, g=g_t, beta=beta_t, initial_state=rec_state, output_final_state=True, use_qk_l2norm_in_kernel=True
                            )
                        else:
                            core_out, last_state = chunk_gated_delta_rule(
                                q, k, v, g=g_t, beta=beta_t, initial_state=rec_state, output_final_state=cache_params is not None, use_qk_l2norm_in_kernel=True
                            )
                        if cache_params is not None:
                            if hasattr(cache_params, "update_recurrent_state"):
                                cache_params.update_recurrent_state(last_state, self.layer_idx)
                            else:
                                rs_list = getattr(cache_params, "recurrent_states", None)
                                if isinstance(rs_list, list) and self.layer_idx < len(rs_list):
                                    rs_list[self.layer_idx] = last_state

                        rows = B * L * H
                        normed = torch.empty((rows, self.head_v_dim), device=hidden_states.device, dtype=hidden_states.dtype)
                        _qwen35_rmsnorm_gated_silu[(rows,)](
                            core_out.contiguous(),
                            z.contiguous(),
                            self.norm.weight,
                            normed,
                            self.head_v_dim,
                            BLOCK_D=self._block_d,
                            eps=self.norm.variance_epsilon,
                        )
                        return self.out_proj(normed.view(B, L, -1))

                    mod.forward = _fast_gdn.__get__(mod, type(mod))

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as fuse_err:
        print(f"[加速引擎] 算子融合未完成 ({type(fuse_err).__name__}: {fuse_err})")
        return

    # 图捕获失败不能把已经装上的融合层一起丢掉。
    try:
        from transformers.modeling_outputs import BaseModelOutputWithPast

        free_gb_now = torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)
        static_cap = 32768 if free_gb_now >= 12.0 else (16384 if free_gb_now >= 8.0 else 4096)
        max_static_seq = max(2048, min(int(max_context_len or 32768), static_cap))
        t_cfg = text_model.config
        embed_dtype = text_model.embed_tokens.weight.dtype
        dev = text_model.embed_tokens.weight.device
        try:
            sc = StaticCache(config=text_model.config, max_cache_len=max_static_seq)
        except Exception:
            sc = StaticCache(config=mdl.config, max_cache_len=max_static_seq)
        for _lyr in sc.layers:
            _lyr.is_compileable = False
        try:
            sc.is_compileable = False
        except Exception:
            pass
        sc._no_left_pad = True

        # 旧版 StaticCache 把 conv 状态长度锁成第一次 prefill 的长度，而且没有 get_conv_state。
        # 这里改成按卷积核宽度保存，这样不同长度的 prefill 都能对上已捕获的 decode 图。
        def _stabilize_linear_cache(cache):
            for lyr in cache.layers:
                if isinstance(getattr(lyr, "conv_states", None), dict) or isinstance(getattr(lyr, "has_previous_state", None), dict):
                    continue
                if not hasattr(lyr, "update_conv_state"):
                    continue
                box = {"conv": None, "rec": None, "has_prev": False, "ksize": None, "batch_shape": None}
                lyr._fast_conv = box
                orig_reset = lyr.reset

                def update_conv_state(conv_states, state_idx=0, conv_kernel_size=None, _box=box, **kwargs):
                    ksize = int(conv_kernel_size or conv_states.shape[-1])
                    batch_shape = tuple(conv_states.shape[:-1])
                    if (
                        _box["conv"] is None
                        or _box["ksize"] != ksize
                        or _box["batch_shape"] != batch_shape
                        or _box["conv"].device != conv_states.device
                    ):
                        _box["ksize"] = ksize
                        _box["batch_shape"] = batch_shape
                        _box["conv"] = torch.zeros(
                            *batch_shape, ksize, dtype=conv_states.dtype, device=conv_states.device
                        )
                        _box["has_prev"] = False
                    if not _box["has_prev"]:
                        full = conv_states
                        if full.shape[-1] < ksize:
                            full = F.pad(full, (ksize - full.shape[-1], 0))
                        _box["has_prev"] = True
                    else:
                        full = torch.cat([_box["conv"], conv_states], dim=-1)
                    _box["conv"].copy_(full[..., -ksize:])
                    return full

                def update_recurrent_state(recurrent_states, state_idx=0, _box=box, **kwargs):
                    if (
                        _box["rec"] is None
                        or _box["rec"].shape != recurrent_states.shape
                        or _box["rec"].device != recurrent_states.device
                    ):
                        _box["rec"] = torch.empty_like(recurrent_states)
                    _box["rec"].copy_(recurrent_states)
                    return _box["rec"]

                def reset(_box=box, _orig=orig_reset):
                    try:
                        _orig()
                    except Exception:
                        pass
                    _box["has_prev"] = False
                    if _box["conv"] is not None:
                        _box["conv"].zero_()
                    if _box["rec"] is not None:
                        _box["rec"].zero_()

                lyr.update_conv_state = update_conv_state
                lyr.update_recurrent_state = update_recurrent_state
                lyr.reset = reset

        def _install_cache_api(cache):
            if not hasattr(cache, "get_conv_state"):
                def get_conv_state(layer_idx, state_idx=0):
                    lyr = cache.layers[layer_idx]
                    box = getattr(lyr, "_fast_conv", None)
                    if box is not None:
                        if not box["has_prev"] or box["conv"] is None:
                            return None
                        return box["conv"]
                    hps = getattr(lyr, "has_previous_state", False)
                    if isinstance(hps, dict):
                        if not hps.get(state_idx, False):
                            return None
                    elif not hps:
                        return None
                    st = getattr(lyr, "conv_states", None)
                    if isinstance(st, dict):
                        return st.get(state_idx)
                    if st is None or not getattr(lyr, "is_conv_states_initialized", False):
                        return None
                    return st

                cache.get_conv_state = get_conv_state
            if not hasattr(cache, "get_recurrent_state"):
                def get_recurrent_state(layer_idx, state_idx=0):
                    lyr = cache.layers[layer_idx]
                    box = getattr(lyr, "_fast_conv", None)
                    if box is not None:
                        if not box["has_prev"] or box["rec"] is None:
                            return None
                        return box["rec"]
                    hps = getattr(lyr, "has_previous_state", False)
                    if isinstance(hps, dict):
                        if not hps.get(state_idx, False):
                            return None
                    elif not hps:
                        return None
                    st = getattr(lyr, "recurrent_states", None)
                    if isinstance(st, dict):
                        return st.get(state_idx)
                    if st is None or not getattr(lyr, "is_recurrent_states_initialized", False):
                        return None
                    return st

                cache.get_recurrent_state = get_recurrent_state
            if not hasattr(cache, "is_recording_past"):
                def is_recording_past(layer_idx):
                    return bool(getattr(cache.layers[layer_idx], "record_past", False))

                cache.is_recording_past = is_recording_past
            orig_len = cache.get_seq_length

            def get_seq_length(layer_idx=0):
                n = orig_len(layer_idx)
                if torch.is_tensor(n):
                    return int(n.item())
                return n

            cache.get_seq_length = get_seq_length

        _stabilize_linear_cache(sc)
        _install_cache_api(sc)

        min_val = torch.finfo(embed_dtype).min
        mask_buf = torch.full((1, 1, 1, max_static_seq), min_val, device=dev, dtype=embed_dtype)
        fast_out_cls = getattr(mq, "Qwen3_5ModelOutputWithPast", None) or BaseModelOutputWithPast

        def _reset_sc_inplace():
            try:
                sc.reset()
                return
            except Exception:
                pass
            for lyr in sc.layers:
                box = getattr(lyr, "_fast_conv", None)
                if box is not None:
                    box["has_prev"] = False
                    if box["conv"] is not None:
                        box["conv"].zero_()
                    if box["rec"] is not None:
                        box["rec"].zero_()
                conv = getattr(lyr, "conv_states", None)
                if isinstance(conv, dict):
                    for st in conv.values():
                        if torch.is_tensor(st):
                            st.zero_()
                    rec = getattr(lyr, "recurrent_states", None)
                    if isinstance(rec, dict):
                        for st in rec.values():
                            if torch.is_tensor(st):
                                st.zero_()
                    hps = getattr(lyr, "has_previous_state", None)
                    if isinstance(hps, dict):
                        for k in list(hps):
                            hps[k] = False
                elif torch.is_tensor(conv):
                    conv.zero_()
                    rec = getattr(lyr, "recurrent_states", None)
                    if torch.is_tensor(rec):
                        rec.zero_()
                    if isinstance(getattr(lyr, "has_previous_state", None), bool):
                        lyr.has_previous_state = False
                if getattr(lyr, "is_initialized", False) and torch.is_tensor(getattr(lyr, "keys", None)):
                    lyr.keys.zero_()
                    lyr.values.zero_()
                cum = getattr(lyr, "cumulative_length", None)
                if torch.is_tensor(cum):
                    cum.zero_()
                elif isinstance(cum, int):
                    lyr.cumulative_length = 0

        orig_tm_forward = text_model.forward
        orig_lm_head_forward = mdl.lm_head.forward

        with torch.inference_mode():
            dummy_emb = torch.zeros((1, 4, t_cfg.hidden_size), device=dev, dtype=embed_dtype)
            orig_tm_forward(inputs_embeds=dummy_emb, past_key_values=sc, use_cache=True)

            static_attn_lyr = next(lyr for lyr in sc.layers if hasattr(lyr, "cumulative_length"))
            s_emb = torch.zeros((1, 1, t_cfg.hidden_size), device=dev, dtype=embed_dtype)
            s_pos3 = torch.zeros((3, 1, 1), device=dev, dtype=torch.long)
            mask_dict = {"full_attention": mask_buf, "linear_attention": None}

            def _decode_graph_step():
                pos_emb = text_model.rotary_emb(s_emb, s_pos3)
                h = s_emb
                for i, dec in enumerate(text_model.layers[: t_cfg.num_hidden_layers]):
                    h = dec(
                        h,
                        position_embeddings=pos_emb,
                        attention_mask=mask_dict[t_cfg.layer_types[i]],
                        past_key_values=sc,
                        use_cache=True,
                    )
                h_norm = text_model.norm(h)
                logits_out = orig_lm_head_forward(h_norm)
                return h_norm, logits_out

            side_stream = torch.cuda.Stream(device=dev)
            side_stream.wait_stream(torch.cuda.current_stream(device=dev))
            with torch.cuda.stream(side_stream):
                for _ in range(3):
                    s_out, s_logits = _decode_graph_step()
            torch.cuda.current_stream(device=dev).wait_stream(side_stream)

            cg = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cg, stream=side_stream):
                s_out, s_logits = _decode_graph_step()
            torch.cuda.synchronize(device=dev)
            _reset_sc_inplace()

        s_out_ptr = s_out.data_ptr()
        s_out_numel = s_out.numel()

        def _fast_lm_head_forward(x):
            if x.is_cuda and x.data_ptr() == s_out_ptr and x.numel() == s_out_numel:
                return s_logits
            return orig_lm_head_forward(x)

        def _hooked_tm_forward(
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            **kwargs,
        ):
            if inputs_embeds is None and input_ids is not None:
                inputs_embeds = text_model.embed_tokens(input_ids)
                input_ids = None
            if past_key_values is sc and use_cache and inputs_embeds is not None and inputs_embeds.shape[0] == 1:
                L = inputs_embeds.shape[1]
                if L > 1:
                    mask_buf.fill_(min_val)
                    if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2 and attention_mask.shape[1] == L:
                        mask_buf[:, :, :, :L] = torch.where(attention_mask.view(1, 1, 1, L).bool(), 0.0, min_val)
                    else:
                        mask_buf[:, :, :, :L] = 0.0
                    return orig_tm_forward(
                        input_ids=None,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=sc,
                        inputs_embeds=inputs_embeds,
                        use_cache=True,
                        **kwargs,
                    )
                elif L == 1:
                    cur_len = static_attn_lyr.cumulative_length
                    if position_ids is None:
                        p3 = cur_len.reshape(1, 1, 1).expand(3, 1, 1)
                    else:
                        pos = position_ids
                        if pos.ndim == 1:
                            pos = pos.reshape(1, 1, -1)
                        elif pos.ndim == 2:
                            pos = pos.reshape(1, pos.shape[0], pos.shape[-1])
                        if pos.shape[0] == 4:
                            pos = pos[1:]
                        elif pos.shape[0] == 1:
                            pos = pos.expand(3, -1, -1)
                        elif pos.shape[0] != 3:
                            pos = pos.reshape(3, 1, -1)
                        p3 = pos[:, :1, -1:].reshape(3, 1, 1)
                    if p3.dtype != s_pos3.dtype or p3.device != s_pos3.device:
                        p3 = p3.to(device=s_pos3.device, dtype=s_pos3.dtype)
                    s_emb.copy_(inputs_embeds)
                    s_pos3.copy_(p3)
                    mask_buf.scatter_(3, cur_len.reshape(1, 1, 1, 1), 0.0)
                    cg.replay()
                    return fast_out_cls(last_hidden_state=s_out, past_key_values=sc)

            return orig_tm_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                **kwargs,
            )

        text_model.forward = _hooked_tm_forward
        mdl.lm_head.forward = _fast_lm_head_forward

        mdl._cg_refs = (cg, sc, side_stream, s_emb, s_pos3, mask_buf, s_out, s_logits)
        if getattr(mdl, "generation_config", None) is not None:
            mdl.generation_config.disable_compile = True

        orig_generate = mdl.generate

        def _fast_generate(*args, **kwargs):
            kwargs.setdefault("disable_compile", True)
            inp_ids = kwargs.get("input_ids")
            if inp_ids is None and len(args) > 0 and isinstance(args[0], torch.Tensor):
                inp_ids = args[0]
            amask = kwargs.get("attention_mask")
            no_pad = True if not isinstance(amask, torch.Tensor) else bool(amask.all().item())
            max_new = int(kwargs.get("max_new_tokens") or 2048)
            if (
                no_pad
                and kwargs.get("past_key_values") is None
                and inp_ids is not None
                and inp_ids.ndim == 2
                and inp_ids.shape[0] == 1
                and (inp_ids.shape[1] + max_new) <= (max_static_seq - 8)
            ):
                _reset_sc_inplace()
                kwargs["past_key_values"] = sc
            return orig_generate(*args, **kwargs)

        mdl.generate = _fast_generate
        print(
            f"[加速引擎] 全 64 层融合算子 + CUDAGraph 静态图引擎已就绪 "
            f"(CUDAGraph 静态加速 {max_static_seq} tokens / 动态最大上下文 {int(max_context_len or 32768)} tokens)！"
        )
    except Exception as opt_err:
        print(f"[加速引擎] CUDAGraph 未捕获，融合算子仍保留 ({type(opt_err).__name__}: {opt_err})")
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

# ==============================================================================
# 全局状态、算力计费账本与 /dev/shm 内存盘加速 (借鉴 comfy-colab-kit)
# ==============================================================================
BOOT_TIME = time.time()
LAST_ACTIVE_TIME = time.time()
IDLE_TIMEOUT_MINUTES = 30  # 默认闲置 30 分钟自动释放 Colab 虚拟机，可在 Mac 端调节

import gc

PIPELINE = None
LLM_BUNDLE = None  # {"backend": "transformers"|"gguf", ...}
LOADED_RUNTIMES: Dict[str, Dict[str, Any]] = {}  # 多模型显存共驻表: model_id -> runtime dict
SELECTED_MODEL_ID = "minimax_h3"
LOADED_MODEL_ID = None
CURRENT_MODEL_MODE = None
QWEN_CONTEXT_LENGTH = 32768  # 用户可自定义上下文长度 (4K ~ 256K = 262144 tokens)
API_AUTH_KEY = ""
API_REQUEST_COUNT = 0
GPU_LOCK = threading.Lock()
ABORT_GENERATION_EVENT = threading.Event()
JOBS: Dict[str, Dict[str, Any]] = {}
_JOB_CANCEL_EVENTS: Dict[str, threading.Event] = {}


class GenerationAbortedError(RuntimeError):
    """用户主动点击停止推理时立即中断 DiffSynth / Transformers GPU 循环的受控异常"""
    pass


def _clamp_context_length(val: Optional[int]) -> int:
    try:
        v = int(val) if val is not None else QWEN_CONTEXT_LENGTH
    except Exception:
        v = QWEN_CONTEXT_LENGTH
    if v <= 0:
        v = QWEN_CONTEXT_LENGTH
    return max(4096, min(262144, v))


def _is_llm_model_id(model_id: Optional[str]) -> bool:
    if not model_id:
        return False
    return MODEL_CATALOG.get(model_id, {}).get("modality") in ("llm_vlm", "llm")


# ==============================================================================
# 自定义 GGUF 模型：从 HuggingFace 任意仓库动态注册，不局限于内置 5 个模型
# ==============================================================================
CUSTOM_MODELS_FILE = os.path.join(
    "/content" if os.path.exists("/content") else os.getcwd(),
    "custom_models.json",
)
CUSTOM_MODEL_SPECS: Dict[str, Dict[str, Any]] = {}


def _quant_from_filename(fname: str) -> str:
    """从 GGUF 文件名推断量化级别，如 -Q4_K_M.gguf -> Q4_K_M"""
    low = (fname or "").lower()
    m = re.search(r"[.-]((?:i?q)\d+(?:_[a-z0-9]+)*)\.gguf$", low)
    if m:
        return m.group(1).upper()
    if "bf16" in low:
        return "BF16"
    if "fp16" in low or "-f16" in low:
        return "F16"
    if "fp32" in low:
        return "F32"
    return "未知量化"


def _approx_gb_from_bytes(n: int) -> float:
    return round(n / (1024 ** 3), 2)


def hf_api_list_gguf_files(repo_id: str, hf_token: str = "") -> Dict[str, Any]:
    """查询 HF 仓库文件清单，筛出 GGUF 主权重与 mmproj 视觉投影器"""
    repo_id = (repo_id or "").strip().strip("/")
    if "/" not in repo_id:
        raise ValueError("repo_id 需要形如 unsloth/Qwen2.5-7B-Instruct-GGUF")
    api_url = f"https://huggingface.co/api/models/{repo_id}"
    headers = {"User-Agent": "EasyColab/2.0"}
    tok = (hf_token or os.environ.get("HF_TOKEN") or "").strip()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    opener = urllib.request.build_opener(_StripAuthRedirectHandler())

    # 一次请求带 blobs=true 就能拿到文件体积，避免二次往返
    try:
        with opener.open(urllib.request.Request(api_url + "?blobs=true", headers=headers), timeout=25) as resp:
            info = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            hint = "该仓库需要授权（gated），请填写有访问权限的 HF Token" if tok else "该仓库需要 HF Token（gated/私有），请填写 Token 后重试"
            raise ValueError(f"{repo_id} 返回 HTTP {e.code}：{hint}") from e
        if e.code == 404:
            raise ValueError(f"{repo_id} 不存在或未公开（HTTP 404）") from e
        raise ValueError(f"{repo_id} 请求失败 HTTP {e.code}") from e

    kids = info.get("siblings") or []
    ggufs: List[Dict[str, Any]] = []
    mmprojs: List[Dict[str, Any]] = []
    for s in kids:
        fname = s.get("rfilename") or ""
        if not fname.lower().endswith(".gguf"):
            continue
        size = int(s.get("size") or 0)
        entry = {
            "filename": fname,
            "quant": _quant_from_filename(fname),
            "size_bytes": size,
            "size_gb": _approx_gb_from_bytes(size),
        }
        # mmproj 是视觉投影器，通常几百 MB，单独归组
        if "mmproj" in fname.lower():
            mmprojs.append(entry)
        else:
            ggufs.append(entry)

    ggufs.sort(key=lambda x: x["size_bytes"])

    return {
        "repo_id": repo_id,
        "gated": bool(info.get("gated")),
        "pipeline_tag": info.get("pipeline_tag"),
        "library_name": info.get("library_name"),
        "downloads": info.get("downloads"),
        "gguf_files": ggufs,
        "mmproj_files": mmprojs,
        "has_sizes": any(f["size_bytes"] for f in ggufs),
    }


def register_custom_gguf_model(
    repo_id: str,
    filename: str,
    name: str = "",
    mmproj_file: str = "",
    vram_gb: float = 0.0,
    approx_size_gb: float = 0.0,
    context_length: int = 32768,
) -> Dict[str, Any]:
    """把任意 HF GGUF 仓库注册成可下载 / 可加载的模型条目（持久化到 custom_models.json）"""
    repo_id = (repo_id or "").strip().strip("/")
    filename = (filename or "").strip().lstrip("/")
    if not repo_id or "/" not in repo_id:
        raise ValueError("repo_id 需要形如 unsloth/Qwen2.5-7B-Instruct-GGUF")
    if not filename.lower().endswith(".gguf"):
        raise ValueError("filename 必须是仓库内的 .gguf 文件路径")

    model_id = "custom_" + re.sub(r"[^a-z0-9]+", "_", f"{repo_id}_{filename}".lower()).strip("_")[:90]
    quant = _quant_from_filename(filename)
    display_name = (name or "").strip() or f"{repo_id.split('/')[-1]} · {quant}"

    files: List[Tuple[str, str]] = [(filename, f"{quant} 主权重")]
    mm = (mmproj_file or "").strip().lstrip("/")
    if mm:
        files.append((mm, "多模态视觉投影器 mmproj"))

    # 体积未知时回查 HF，否则下载路径选择（/dev/shm vs Drive）会误判
    if float(approx_size_gb or 0.0) <= 0.0:
        try:
            probe = hf_api_list_gguf_files(repo_id)
            for f in probe.get("gguf_files", []) + probe.get("mmproj_files", []):
                if f.get("filename") == filename and f.get("size_bytes"):
                    approx_size_gb = f["size_gb"]
                    break
        except Exception as e:
            print(f"[自定义模型] 体积探测失败（可手动指定）: {e}")

    spec = {
        "id": model_id,
        "name": display_name,
        "repo_id": repo_id,
        "modality": "llm_vlm" if mm else "llm",
        "category": f"自定义 GGUF · {quant} · {repo_id}",
        "approx_size_gb": float(approx_size_gb or 0.0),
        "vram_gb": float(vram_gb or approx_size_gb or 0.0),
        "files": files,
        "gguf_main": filename,
        "gguf_mmproj": mm,
        "default_context": _clamp_context_length(context_length),
        "quant": quant,
        "is_custom": True,
    }
    CUSTOM_MODEL_SPECS[model_id] = spec
    MODEL_CATALOG[model_id] = spec
    _save_custom_gguf_models()
    print(f"[自定义模型] 已注册 {display_name} ({model_id})")
    return spec


def remove_custom_gguf_model(model_id: str) -> bool:
    """删除自定义模型注册（保留已下载权重，可重新注册回来）"""
    if model_id not in CUSTOM_MODEL_SPECS:
        return False
    CUSTOM_MODEL_SPECS.pop(model_id, None)
    MODEL_CATALOG.pop(model_id, None)
    _save_custom_gguf_models()
    return True


def _save_custom_gguf_models():
    try:
        with open(CUSTOM_MODELS_FILE, "w", encoding="utf-8") as f:
            json.dump(CUSTOM_MODEL_SPECS, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[自定义模型] 保存失败: {e}")


def _load_custom_gguf_models():
    """启动时恢复自定义模型并注册进统一目录，后续按 model_id 分发的逻辑自动可见"""
    try:
        if os.path.exists(CUSTOM_MODELS_FILE):
            with open(CUSTOM_MODELS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, dict) and v.get("is_custom"):
                        CUSTOM_MODEL_SPECS[k] = v
                        MODEL_CATALOG[k] = v
                print(f"[自定义模型] 已恢复 {len(CUSTOM_MODEL_SPECS)} 个自注册 GGUF 模型")
    except Exception as e:
        print(f"[自定义模型] 读取失败: {e}")


def _is_custom_gguf_model(model_id: Optional[str]) -> bool:
    return bool(model_id) and model_id in CUSTOM_MODEL_SPECS


def _is_gguf_model_id(model_id: Optional[str]) -> bool:
    """判定走 llama-server 后端：内置 GGUF + 全部自定义 GGUF"""
    if not model_id:
        return False
    if model_id in CUSTOM_MODEL_SPECS:
        return True
    return model_id == "qwen38_27b_gguf"


def _gguf_spec_paths(model_id: str) -> Tuple[str, Optional[str]]:
    """返回 (主权重相对文件名, mmproj 相对文件名或 None)"""
    spec = CUSTOM_MODEL_SPECS.get(model_id) or MODEL_CATALOG.get(model_id, {})
    main = spec.get("gguf_main") or ""
    mmproj = spec.get("gguf_mmproj") or None
    return main, mmproj


def _stop_gguf_server_bundle(bundle: Optional[Dict[str, Any]]):
    if not bundle or bundle.get("backend") != "gguf":
        return
    proc = bundle.pop("server_proc", None)
    if proc is not None:
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)
        except Exception:
            pass
        time.sleep(0.25)


def _cleanup_all_gguf_servers():
    for rt in list(LOADED_RUNTIMES.values()):
        _stop_gguf_server_bundle(rt.get("llm_bundle"))
    _stop_gguf_server_bundle(LLM_BUNDLE)


atexit.register(_cleanup_all_gguf_servers)
if os.path.exists("/content"):
    subprocess.run(["pkill", "-f", "llama-server.*Qwen3\\.8-27B"], check=False)


def _make_abort_stopping_criteria(local_cancel: Optional[threading.Event] = None):
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _AbortCriteria(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs) -> bool:
            if ABORT_GENERATION_EVENT.is_set():
                return True
            if local_cancel is not None and local_cancel.is_set():
                return True
            return False

    return StoppingCriteriaList([_AbortCriteria()])


def _make_timed_token_streamer(tokenizer, timeout: float = 120.0):
    from transformers import TextIteratorStreamer

    class _TimedStreamer(TextIteratorStreamer):
        def __init__(self, tok, **kw):
            super().__init__(tok, **kw)
            self.generated_tokens = 0
            self.first_token_ts: Optional[float] = None
            self._saw_prompt = False

        def put(self, value):
            now = time.time()
            if self.skip_prompt and not self._saw_prompt:
                self._saw_prompt = True
            else:
                n = int(value.numel()) if hasattr(value, "numel") else 1
                self.generated_tokens += max(1, n)
                if self.first_token_ts is None:
                    self.first_token_ts = now
            super().put(value)

        @property
        def tok_per_sec(self) -> float:
            if self.first_token_ts is None or self.generated_tokens <= 1:
                return 0.0
            dt = max(0.05, time.time() - self.first_token_ts)
            return round((self.generated_tokens - 1) / dt, 1)

    return _TimedStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=timeout,
    )

MODEL_DOWNLOAD_STATE: Dict[str, Any] = {
    "is_downloading": False,
    "is_loading_vram": False,
    "progress": 0.0,
    "message": "等待检查模型状态...",
}


def _get_system_ram_gb() -> Tuple[float, float]:
    """返回 (系统总内存 GB, 当前可用内存 GB)"""
    try:
        mem_total, mem_avail = 0.0, 0.0
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total = int(line.split()[1]) / (1024 ** 2)
                elif line.startswith("MemAvailable:"):
                    mem_avail = int(line.split()[1]) / (1024 ** 2)
        return round(mem_total, 1), round(mem_avail, 1)
    except Exception:
        return 0.0, 0.0


def setup_colab_ramdisk() -> Optional[str]:
    """
    借鉴 comfy-colab-kit/packages/colab/setup.py：
    在 Colab 大内存实例 (G4 176GB / A100 83GB / L4 53GB RAM) 上自动扩容 /dev/shm (tmpfs 内存盘，读写 2.4GB/s+)，
    为适配容量的模型提供零延迟 mmap 预热加速。
    """
    if not os.path.exists("/dev/shm"):
        return None
    total_ram_gb, avail_ram_gb = _get_system_ram_gb()
    if total_ram_gb >= 48.0:
        # 避免 DiffSynth DiskMap 默认每 2GB (10^9 参数) 反复关闭并重开全部 14 个 safetensors 分片句柄
        os.environ.setdefault("DIFFSYNTH_DISK_MAP_BUFFER_SIZE", str(80 * 10**9))
    if total_ram_gb < 40.0 or avail_ram_gb < 25.0:
        return None
    try:
        cur_shm_total_gb = shutil.disk_usage("/dev/shm").total / (1024 ** 3)
        target_gb = min(88, max(30, int(total_ram_gb * 0.50)))
        if cur_shm_total_gb + 1.0 < target_gb:
            subprocess.run(
                ["mount", "-o", f"remount,size={target_gb}G", "/dev/shm"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        shm_stat = shutil.disk_usage("/dev/shm")
        shm_free_gb = shm_stat.free / (1024 ** 3)
        if shm_free_gb >= 22.5:
            ram_root = "/dev/shm/EasyColab_Models"
            os.makedirs(ram_root, exist_ok=True)
            print(f"⚡ [RAMDisk 内存盘加速已开启] /dev/shm 可用空间 {shm_free_gb:.1f} GB (2.4GB/s 零延迟读写)")
            return ram_root
    except Exception:
        pass
    return None


RAM_ROOT_DIR = setup_colab_ramdisk()

# 如果挂载了 Google Drive，优先将模型持久化保存在云盘 EasyColab_Models（并自动兼容读取历史 Aevix_Models / MiniMax_H3_Models 目录）
DRIVE_MOUNTED = os.path.exists("/content/drive/MyDrive")


def _has_rclone_drive_remote() -> bool:
    rc_path = os.path.expanduser("~/.config/rclone/rclone.conf")
    if not os.path.exists(rc_path) or not shutil.which("rclone"):
        return False
    try:
        with open(rc_path, "r", encoding="utf-8", errors="ignore") as f:
            return "[easycolab_gdrive]" in f.read()
    except Exception:
        return False


COMFYUI_LOCK = threading.Lock()
COMFYUI_STATE: Dict[str, Any] = {
    "is_running": False,
    "is_starting": False,
    "is_syncing_drive": False,
    "is_downloading_weight": False,
    "installing_pack_id": "",
    "progress": 0.0,
    "status_message": "尚未启动（支持一键拉起 ComfyUI + Google Drive 自动软链持久化）",
    "public_url": "",
    "local_port": 8188,
    "queue_running": 0,
    "queue_pending": 0,
    "persistence_mode": "drive_symlink" if DRIVE_MOUNTED else ("rclone_drive_sync" if _has_rclone_drive_remote() else "local_only"),
    "persistence_label": "Google Drive 实时软链挂载" if DRIVE_MOUNTED else ("Google Drive 云盘自动持久化" if _has_rclone_drive_remote() else "实例本地高速存储"),
    "drive_persisted": bool(DRIVE_MOUNTED or _has_rclone_drive_remote()),
    "custom_nodes_count": 0,
    "models_count": 0,
    "workflows_count": 0,
    "last_sync_ts": 0.0,
    "last_history_count": 0,
    "comfy_proc": None,
    "tunnel_proc": None,
}
MODEL_BASE_ROOT = (
    "/content/drive/MyDrive/EasyColab_Models"
    if DRIVE_MOUNTED
    else os.path.join(os.getcwd(), "models")
)
LEGACY_DRIVE_MODEL_ROOT = "/content/drive/MyDrive/MiniMax_H3_Models"
LEGACY_DRIVE_MODEL_ROOTS = [
    "/content/drive/MyDrive/Aevix_Models",
    "/content/drive/MyDrive/MiniMax_H3_Models",
]
os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = MODEL_BASE_ROOT

MINIMAX_H3_BF16_FILES = (
    [
        (f"FL2VA/transformer/model-{i:05d}-of-00013.safetensors", f"DiT 音视频主干 BF16 {i}/13")
        for i in range(1, 14)
    ]
    + [
        (f"FL2VA/text_encoder/model-{i:05d}-of-00014.safetensors", f"多模态文本编码器 BF16 {i}/14")
        for i in range(1, 15)
    ]
    + [
        ("FL2VA/video_vae/source/model.safetensors", "3D 视频 VAE 解码器 BF16 (约 9.7 GB)"),
        ("FL2VA/audio_vae/model.safetensors", "32kHz 立体声 VAE 解码器 BF16 (约 0.56 GB)"),
    ]
)

# ==============================================================================
# 开源旗舰多模型注册中心 (Hugging Face Top Models 2026 · 全量免审核 Gated=False)
# ==============================================================================
MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "minimax_h3": {
        "id": "minimax_h3",
        "name": "MiniMax-H3 (33B)",
        "repo_id": "MiniMaxAI/MiniMax-H3",
        "modality": "video_audio",
        "category": "全模态音视频生成 · BF16 原生满血 · 32kHz 立体声",
        "approx_size_gb": 134.1,
        "files": MINIMAX_H3_BF16_FILES,
    },
    "qwen38_27b": {
        "id": "qwen38_27b",
        "name": "Qwen3.8-27B",
        "repo_id": "Qwen/Qwen3.8-27B",
        "modality": "llm_vlm",
        "category": "视觉理解与深度推理大模型 · BF16 原生满血",
        "approx_size_gb": 51.8,
        "files": [
            (f"model-{i:05d}-of-00018.safetensors", f"Qwen3.8-27B BF16 权重分片 {i}/18")
            for i in range(1, 19)
        ],
    },
    "qwen38_27b_gguf": {
        "id": "qwen38_27b_gguf",
        "name": "Qwen3.8-27B-GGUF",
        "repo_id": "unsloth/Qwen3.8-27B-GGUF",
        "modality": "llm_vlm",
        "category": "Unsloth UD-Q4_K_M + MTP 投机加速 · 支持 256K 上下文",
        "approx_size_gb": 18.7,
        "files": [
            ("Qwen3.8-27B-UD-Q4_K_M.gguf", "Unsloth UD-Q4_K_M 主模型权重 (约 16.5 GB)"),
            ("mmproj-F16.gguf", "多模态视觉投影器 mmproj-F16 (约 0.93 GB)"),
            ("MTP/mtp-Qwen3.8-27B-Q4_0.gguf", "Unsloth MTP 多词元并行投机解码头 (约 1.31 GB)"),
        ],
    },
    "qwen_image_21": {
        "id": "qwen_image_21",
        "name": "Qwen-Image-2.1",
        "repo_id": "Qwen/Qwen-Image-2.1",
        "modality": "image",
        "category": "旗舰文生图与多图指令修图",
        "approx_size_gb": 30.9,
        "files": [
            ("transformer/diffusion_pytorch_model-00001-of-00002.safetensors", "DiT 图像主干 1/2"),
            ("transformer/diffusion_pytorch_model-00002-of-00002.safetensors", "DiT 图像主干 2/2"),
            ("text_encoder/model-00001-of-00004.safetensors", "多模态编码器 1/4"),
            ("text_encoder/model-00002-of-00004.safetensors", "多模态编码器 2/4"),
            ("text_encoder/model-00003-of-00004.safetensors", "多模态编码器 3/4"),
            ("text_encoder/model-00004-of-00004.safetensors", "多模态编码器 4/4"),
            ("vae/diffusion_pytorch_model.safetensors", "高保真图像 VAE"),
        ],
    },
    "z_image_turbo": {
        "id": "z_image_turbo",
        "name": "Z-Image-Turbo",
        "repo_id": "Tongyi-MAI/Z-Image-Turbo",
        "modality": "image",
        "category": "8 步极速超清写实与中英文字生图",
        "approx_size_gb": 30.6,
        "files": [
            ("transformer/diffusion_pytorch_model-00001-of-00003.safetensors", "Z-Image DiT 主干 1/3"),
            ("transformer/diffusion_pytorch_model-00002-of-00003.safetensors", "Z-Image DiT 主干 2/3"),
            ("transformer/diffusion_pytorch_model-00003-of-00003.safetensors", "Z-Image DiT 主干 3/3"),
            ("text_encoder/model-00001-of-00003.safetensors", "文本编码器 1/3"),
            ("text_encoder/model-00002-of-00003.safetensors", "文本编码器 2/3"),
            ("text_encoder/model-00003-of-00003.safetensors", "文本编码器 3/3"),
            ("vae/diffusion_pytorch_model.safetensors", "图像 VAE 解码器"),
        ],
    },
}

DEFAULT_OUTPUT_DIR = (
    "/content/drive/MyDrive/EasyColab_Outputs"
    if DRIVE_MOUNTED
    else os.path.join(os.getcwd(), "outputs")
)
LEGACY_DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/MiniMax_H3_Outputs"
LEGACY_DRIVE_OUTPUT_DIRS = [
    "/content/drive/MyDrive/Aevix_Outputs",
    "/content/drive/MyDrive/MiniMax_H3_Outputs",
]
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(os.getcwd(), "models"), exist_ok=True)


# ==============================================================================
# 借鉴 comfy-colab-kit 的 8 线程 HTTP Range 并行极速下载器 (400~700 MB/s)
# ==============================================================================
class _StripAuthRedirectHandler(urllib.request.HTTPRedirectHandler):
    """重定向到 HuggingFace CDN (CloudFront/Xet) 时剥离 Authorization 头，防止 400/403 拒绝"""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is not None and "Authorization" in new_req.unredirected_hdrs:
            del new_req.unredirected_hdrs["Authorization"]
        if new_req is not None and "Authorization" in new_req.headers:
            del new_req.headers["Authorization"]
        return new_req


def fast_parallel_hf_download(
    repo_id: str,
    filename: str,
    dest_dir: str,
    hf_token: str = "",
    num_connections: int = 8,
) -> str:
    """
    使用 8 路并发 HTTP Range 连接直接从 HuggingFace CDN 下载大权重文件至内存盘/本地盘，
    规避单连接 ~60MB/s 限速，在 Colab 上可达 400~700 MB/s。
    """
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = dest_path + ".part"

    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    opener = urllib.request.build_opener(_StripAuthRedirectHandler())
    headers = {"User-Agent": "EasyColab/2.0"}
    token = (hf_token or os.environ.get("HF_TOKEN") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        # Probe final CDN URL & Content-Length via Range: bytes=0-0
        probe_req = urllib.request.Request(url, headers={**headers, "Range": "bytes=0-0"})
        with opener.open(probe_req, timeout=20) as resp:
            cdn_url = resp.geturl()
            content_range = resp.headers.get("Content-Range", "")
            if "/" in content_range:
                total_size = int(content_range.split("/")[-1])
            else:
                total_size = int(resp.headers.get("Content-Length", 0))

        # 若文件较小 (<32MB) 或未返回支持分片的大小，直接走 hf_hub_download
        if total_size < 32 * 1024 * 1024:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=dest_dir)

        with open(temp_path, "wb") as f:
            f.truncate(total_size)

        downloaded_bytes = 0
        dl_lock = threading.Lock()
        start_ts = time.time()
        last_print_ts = [0.0]

        part_size = (total_size + num_connections - 1) // num_connections
        ranges = [
            (i * part_size, min((i + 1) * part_size - 1, total_size - 1))
            for i in range(num_connections)
            if i * part_size < total_size
        ]

        def _download_chunk(byte_range: Tuple[int, int]):
            nonlocal downloaded_bytes
            start_b, end_b = byte_range
            chunk_req = urllib.request.Request(
                cdn_url,
                headers={"User-Agent": "EasyColab/2.0", "Range": f"bytes={start_b}-{end_b}"},
            )
            with urllib.request.urlopen(chunk_req, timeout=60) as r, open(temp_path, "r+b") as wf:
                wf.seek(start_b)
                while True:
                    buf = r.read(4 * 1024 * 1024)  # 4MB buffer
                    if not buf:
                        break
                    wf.write(buf)
                    now = time.time()
                    with dl_lock:
                        downloaded_bytes += len(buf)
                        if now - last_print_ts[0] >= 0.35 or downloaded_bytes >= total_size:
                            last_print_ts[0] = now
                            elapsed = max(0.001, now - start_ts)
                            speed_mbs = (downloaded_bytes / (1024 ** 2)) / elapsed
                            pct = int(downloaded_bytes * 100 / total_size)
                            cur_gb = downloaded_bytes / (1024 ** 3)
                            tot_gb = total_size / (1024 ** 3)
                            eta_s = int(((total_size - downloaded_bytes) / (1024 ** 2)) / max(0.1, speed_mbs))
                            sys.stdout.write(
                                f"\r{filename}: {pct}%|{'█' * (pct // 10):<10}| "
                                f"{cur_gb:.2f}G/{tot_gb:.2f}G [{int(elapsed):02d}s<{eta_s:02d}s, {speed_mbs:.1f}MB/s]\n"
                            )
                            sys.stdout.flush()

        with ThreadPoolExecutor(max_workers=num_connections) as pool:
            futures = [pool.submit(_download_chunk, r) for r in ranges]
            for fut in as_completed(futures):
                fut.result()

        os.replace(temp_path, dest_path)
        return dest_path

    except Exception as e:
        print(f"ℹ️ 多线程分片直连回退至标准 HF 下载 ({e})...")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=dest_dir)



_LAST_JUPYTER_HEARTBEAT_TS: float = 0.0


def _ping_local_colab_jupyter(force: bool = False):
    """
    在活跃推理/下载期间定期轻触本地 Colab Jupyter Server (127.0.0.1:8080/api/kernels)，
    避免无网页模式下推理流量全走 gradio.live 导致 Colab 误判容器闲置而在生成第 2 条视频中途回收实例。
    """
    global _LAST_JUPYTER_HEARTBEAT_TS
    now = time.time()
    if not force and (now - _LAST_JUPYTER_HEARTBEAT_TS) < 40.0:
        return
    _LAST_JUPYTER_HEARTBEAT_TS = now

    def _do_ping():
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for ep in ("http://127.0.0.1:8080/api/kernels", "http://127.0.0.1:8080/api/status"):
            try:
                req = urllib.request.Request(ep, headers={"X-Colab-Client-Agent": "colab-cli"})
                with opener.open(req, timeout=2.0) as resp:
                    resp.read(256)
            except Exception:
                pass

    threading.Thread(target=_do_ping, daemon=True).start()


def touch_activity():
    global LAST_ACTIVE_TIME
    LAST_ACTIVE_TIME = time.time()
    _ping_local_colab_jupyter(force=False)


def estimate_cu_burn_rate(gpu_name: str) -> float:
    """根据 Colab GPU 型号估算每小时消耗的计算单元 (CU/hr)"""
    name_up = (gpu_name or "").upper()
    if "H100" in name_up:
        return 11.78
    elif "PRO 6000" in name_up or "BLACKWELL" in name_up or "G4" in name_up:
        return 10.05
    elif "A100" in name_up:
        return 5.37
    elif "L4" in name_up:
        return 1.71
    elif "T4" in name_up:
        return 1.17
    elif "V100" in name_up:
        return 3.80
    return 1.50


def _idle_watchdog_loop():
    """借鉴 comfy-colab-kit 的 stopper.py：当超过设定闲置时间无任务时自动释放 Colab 虚拟机"""
    while True:
        time.sleep(15)
        # 若正在下载、加载模型、生成视频/LLM推理或 ComfyUI 正在安装/执行节点队列，持续刷新活跃时间并向本地 Jupyter 发送保活脉冲
        if (
            MODEL_DOWNLOAD_STATE["is_downloading"]
            or MODEL_DOWNLOAD_STATE["is_loading_vram"]
            or GPU_LOCK.locked()
            or COMFYUI_STATE["is_starting"]
            or COMFYUI_STATE["is_downloading_weight"]
            or COMFYUI_STATE["queue_running"] > 0
            or COMFYUI_STATE["queue_pending"] > 0
        ):
            touch_activity()
            continue

        if IDLE_TIMEOUT_MINUTES <= 0:
            _ping_local_colab_jupyter(force=False)
            continue

        idle_sec = time.time() - LAST_ACTIVE_TIME
        if idle_sec >= IDLE_TIMEOUT_MINUTES * 60:
            print(f"🛑 [自动省额度看门狗] 已连续闲置 {IDLE_TIMEOUT_MINUTES} 分钟，正在自动释放 Colab 虚拟机...")
            disconnect_colab_runtime()
            break
        else:
            # 尚在用户设定的活跃保活窗口内，轻触本地 Jupyter 防止 Colab 提前强杀
            _ping_local_colab_jupyter(force=False)


threading.Thread(target=_idle_watchdog_loop, daemon=True).start()


def _get_model_dirs(model_id: str) -> Tuple[Optional[str], str, str]:
    """返回指定模型的 (RAM内存盘目录或None, 本地高速NVMe目录, Google Drive/持久盘目录)"""
    spec = MODEL_CATALOG.get(model_id, MODEL_CATALOG["minimax_h3"])
    repo_sub = spec["repo_id"]
    ram_dir = None
    if os.path.exists("/dev/shm"):
        try:
            shm_free_gb = shutil.disk_usage("/dev/shm").free / (1024 ** 3)
            # 仅当 /dev/shm 可用空间足以容纳模型时才把新模型放入内存盘
            if shm_free_gb >= spec["approx_size_gb"] + 3.0:
                ram_dir = f"/dev/shm/EasyColab_Models/{model_id}"
            elif spec["approx_size_gb"] <= 60.0 and (
                os.path.exists(f"/dev/shm/EasyColab_Models/{model_id}")
                or os.path.exists(f"/dev/shm/Aevix_Models/{model_id}")
                or os.path.exists(f"/dev/shm/OpenModels/{model_id}")
            ):
                ram_dir = f"/dev/shm/EasyColab_Models/{model_id}"
        except Exception:
            pass
    nvme_dir = os.path.join(os.getcwd(), "models", repo_sub)
    drive_dir = os.path.join(MODEL_BASE_ROOT, repo_sub)
    return ram_dir, nvme_dir, drive_dir


def _find_existing_model_file(model_id: str, fname: str) -> Optional[str]:
    """按速度优先级查找已存在的模型权重文件：1. /dev/shm 内存盘 -> 2. 本地 NVMe -> 3. Google Drive (含历史目录与 HuggingFace 缓存兼容)"""
    ram_dir, nvme_dir, drive_dir = _get_model_dirs(model_id)
    spec = MODEL_CATALOG.get(model_id, MODEL_CATALOG["minimax_h3"])
    repo_sub = spec["repo_id"]
    candidates = []
    if ram_dir:
        candidates.append(os.path.join(ram_dir, fname))
    candidates.append(os.path.join(f"/dev/shm/EasyColab_Models/{model_id}", fname))
    candidates.append(os.path.join(f"/dev/shm/Aevix_Models/{model_id}", fname))
    candidates.append(os.path.join(f"/dev/shm/OpenModels/{model_id}", fname))
    candidates.append(os.path.join(nvme_dir, fname))
    candidates.append(os.path.join(drive_dir, fname))
    if DRIVE_MOUNTED:
        for leg_root in LEGACY_DRIVE_MODEL_ROOTS:
            if os.path.isdir(leg_root):
                candidates.append(os.path.join(leg_root, repo_sub, fname))
    for p in candidates:
        if os.path.exists(p) and os.path.getsize(p) > 64 * 1024:
            return p
    return None


def check_model_files_status(model_id: Optional[str] = None) -> Dict[str, Any]:
    """检查指定开源模型在内存盘/磁盘/云盘上的权重文件完整性与大小"""
    target_id = model_id if (model_id and model_id in MODEL_CATALOG) else SELECTED_MODEL_ID
    spec = MODEL_CATALOG[target_id]
    req_files = spec["files"]
    ram_dir, nvme_dir, drive_dir = _get_model_dirs(target_id)

    files_info = []
    downloaded_count = 0
    total_size_gb = 0.0
    ram_cached_count = 0

    for fname, desc in req_files:
        actual_path = _find_existing_model_file(target_id, fname)
        if actual_path:
            size_gb = round(os.path.getsize(actual_path) / (1024 ** 3), 2)
            total_size_gb += size_gb
            downloaded_count += 1
            in_ram = actual_path.startswith("/dev/shm/")
            if in_ram:
                ram_cached_count += 1
            files_info.append({
                "name": fname,
                "desc": desc + (" [RAM内存盘]" if in_ram else ""),
                "exists": True,
                "size_gb": size_gb,
            })
        else:
            files_info.append({
                "name": fname,
                "desc": desc,
                "exists": False,
                "size_gb": 0.0,
            })

    all_downloaded = downloaded_count == len(req_files) and len(req_files) > 0
    return {
        "selected_model_id": target_id,
        "selected_model_name": spec["name"],
        "selected_model_repo": spec["repo_id"],
        "selected_model_modality": spec["modality"],
        "files": files_info,
        "downloaded_count": downloaded_count,
        "total_count": len(req_files),
        "all_downloaded": all_downloaded,
        "total_size_gb": round(total_size_gb, 2),
        "drive_persisted": DRIVE_MOUNTED,
        "ramdisk_enabled": ram_dir is not None,
        "ram_cached_count": ram_cached_count,
        "model_storage_path": ram_dir or drive_dir,
        "bf16_shards_count": downloaded_count if target_id == "minimax_h3" else 0,
    }


_TELEMETRY_LOCK = threading.Lock()
_CACHED_DRIVE_METRICS: Dict[str, Any] = {
    "ts": 0.0,
    "drive_total_gb": 0.0,
    "drive_used_gb": 0.0,
    "drive_free_gb": 0.0,
}
_CACHED_FILE_STATUS_MAP: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_CACHED_LIVE_TELEMETRY: Dict[str, Any] = {}


def _get_cached_model_files_status(target_id: str) -> Dict[str, Any]:
    """对磁盘/云盘 FUSE 权重扫描做轻量缓存，避免高频状态轮询在视频生成期间阻塞 I/O"""
    now = time.time()
    ttl = 1.8 if MODEL_DOWNLOAD_STATE["is_downloading"] else 12.0
    with _TELEMETRY_LOCK:
        cached = _CACHED_FILE_STATUS_MAP.get(target_id)
        if cached and (now - cached[0]) < ttl:
            return dict(cached[1])
    fresh = check_model_files_status(target_id)
    with _TELEMETRY_LOCK:
        _CACHED_FILE_STATUS_MAP[target_id] = (now, dict(fresh))
    return fresh


def _get_storage_metrics_dict(refresh_drive: bool = False) -> Dict[str, Any]:
    """采集 Colab 实例系统内存 (RAM)、本地磁盘、/dev/shm 内存加速盘及 Google Drive 云盘指标 (单位 GB)"""
    ram_total_gb, ram_free_gb = _get_system_ram_gb()
    ram_used_gb = round(max(0.0, ram_total_gb - ram_free_gb), 1) if ram_total_gb > 0 else 0.0

    disk_path = "/content" if os.path.exists("/content") else os.getcwd()
    disk_total_gb, disk_used_gb, disk_free_gb = 0.0, 0.0, 0.0
    try:
        du = shutil.disk_usage(disk_path)
        disk_total_gb = round(du.total / (1024 ** 3), 1)
        disk_free_gb = round(du.free / (1024 ** 3), 1)
        disk_used_gb = round(max(0.0, (du.total - du.free) / (1024 ** 3)), 1)
    except Exception:
        pass

    shm_total_gb, shm_used_gb, shm_free_gb = 0.0, 0.0, 0.0
    if os.path.exists("/dev/shm"):
        try:
            su = shutil.disk_usage("/dev/shm")
            shm_total_gb = round(su.total / (1024 ** 3), 1)
            shm_free_gb = round(su.free / (1024 ** 3), 1)
            shm_used_gb = round(max(0.0, (su.total - su.free) / (1024 ** 3)), 1)
        except Exception:
            pass

    now = time.time()
    if DRIVE_MOUNTED and (refresh_drive or (now - _CACHED_DRIVE_METRICS["ts"]) >= 15.0):
        if os.path.exists("/content/drive/MyDrive"):
            try:
                gu = shutil.disk_usage("/content/drive/MyDrive")
                _CACHED_DRIVE_METRICS["drive_total_gb"] = round(gu.total / (1024 ** 3), 1)
                _CACHED_DRIVE_METRICS["drive_free_gb"] = round(gu.free / (1024 ** 3), 1)
                _CACHED_DRIVE_METRICS["drive_used_gb"] = round(max(0.0, (gu.total - gu.free) / (1024 ** 3)), 1)
                _CACHED_DRIVE_METRICS["ts"] = now
            except Exception:
                pass

    return {
        "ram_total_gb": ram_total_gb,
        "ram_used_gb": ram_used_gb,
        "ram_free_gb": ram_free_gb,
        "disk_total_gb": disk_total_gb,
        "disk_used_gb": disk_used_gb,
        "disk_free_gb": disk_free_gb,
        "shm_total_gb": shm_total_gb,
        "shm_used_gb": shm_used_gb,
        "shm_free_gb": shm_free_gb,
        "drive_total_gb": _CACHED_DRIVE_METRICS["drive_total_gb"],
        "drive_used_gb": _CACHED_DRIVE_METRICS["drive_used_gb"],
        "drive_free_gb": _CACHED_DRIVE_METRICS["drive_free_gb"],
    }


def _sample_live_hardware_telemetry_once() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        gpu_name = "CPU Mode (未检测到 GPU)"
        cuda_avail = False
        free_gb, total_gb, used_gb = 0.0, 0.0, 0.0
        cu_rate = 0.10
    else:
        gpu_name = torch.cuda.get_device_name(0)
        cuda_avail = True
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        free_gb = round(free_mem / (1024 ** 3), 2)
        total_gb = round(total_mem / (1024 ** 3), 2)
        used_gb = round(total_gb - free_gb, 2)
        cu_rate = estimate_cu_burn_rate(gpu_name)

    snap = {
        "cuda_available": cuda_avail,
        "gpu_name": gpu_name,
        "vram_free_gb": free_gb,
        "vram_total_gb": total_gb,
        "vram_used_gb": used_gb,
        "cu_burn_rate_per_hour": cu_rate,
    }
    snap.update(_get_storage_metrics_dict(refresh_drive=False))
    if "_poll_local_comfyui_runtime_quiet" in globals():
        _poll_local_comfyui_runtime_quiet()
    with _TELEMETRY_LOCK:
        _CACHED_LIVE_TELEMETRY.clear()
        _CACHED_LIVE_TELEMETRY.update(snap)
    return snap


def _telemetry_sampler_loop():
    """独立后台线程以 2s 周期采样 GPU 显存/内存/磁盘指标，保证 HTTP 轮询非阻塞秒回"""
    while True:
        try:
            _sample_live_hardware_telemetry_once()
        except Exception:
            pass
        time.sleep(2.0)


threading.Thread(target=_telemetry_sampler_loop, daemon=True).start()


def get_live_telemetry_snapshot() -> Dict[str, Any]:
    with _TELEMETRY_LOCK:
        if _CACHED_LIVE_TELEMETRY:
            return dict(_CACHED_LIVE_TELEMETRY)
    return _sample_live_hardware_telemetry_once()


def get_gpu_status_dict(model_id: Optional[str] = None) -> Dict[str, Any]:
    """返回结构化的 GPU 状态、实例磁盘空间、多模型注册信息、CU 算力消耗账本、闲置倒计时与实时终端进度流"""
    global SELECTED_MODEL_ID
    if model_id and model_id in MODEL_CATALOG:
        SELECTED_MODEL_ID = model_id
    target_id = SELECTED_MODEL_ID

    file_status = _get_cached_model_files_status(target_id)
    hw = get_live_telemetry_snapshot()
    uptime_seconds = int(time.time() - BOOT_TIME)
    idle_seconds = int(max(0.0, time.time() - LAST_ACTIVE_TIME))
    idle_remaining_seconds = (
        max(0, IDLE_TIMEOUT_MINUTES * 60 - idle_seconds) if IDLE_TIMEOUT_MINUTES > 0 else -1
    )

    gpu_name = hw["gpu_name"]
    cuda_avail = hw["cuda_available"]
    free_gb = hw["vram_free_gb"]
    total_gb = hw["vram_total_gb"]
    used_gb = hw["vram_used_gb"]
    cu_rate = hw["cu_burn_rate_per_hour"]

    session_cu_used = round((uptime_seconds / 3600.0) * cu_rate, 3)
    loaded_ids = list(LOADED_RUNTIMES.keys())
    loaded_details = [
        {
            "id": mid,
            "name": rt.get("name", mid),
            "mode": rt.get("mode_label", ""),
            "vram_gb": rt.get("vram_gb", 0.0),
            "context_length": int((rt.get("llm_bundle") or {}).get("context_length") or 0),
        }
        for mid, rt in LOADED_RUNTIMES.items()
    ]
    loaded_ctx = 0
    if target_id in LOADED_RUNTIMES and LOADED_RUNTIMES[target_id].get("llm_bundle"):
        loaded_ctx = int(LOADED_RUNTIMES[target_id]["llm_bundle"].get("context_length") or 0)
    else:
        for rt in LOADED_RUNTIMES.values():
            if rt.get("llm_bundle") and rt["llm_bundle"].get("context_length"):
                loaded_ctx = int(rt["llm_bundle"]["context_length"])
                break

    any_loaded = bool(LOADED_RUNTIMES) or (PIPELINE is not None) or (LLM_BUNDLE is not None)
    loaded_for_selected = (target_id in LOADED_RUNTIMES) or (any_loaded and LOADED_MODEL_ID == target_id)
    summary_mode = (
        LOADED_RUNTIMES[target_id]["mode_label"]
        if target_id in LOADED_RUNTIMES
        else (" + ".join(rt["name"] for rt in LOADED_RUNTIMES.values()) if LOADED_RUNTIMES else (CURRENT_MODEL_MODE or "未加载"))
    )

    base = {
        "ok": True,
        "cuda_available": cuda_avail,
        "gpu_name": gpu_name,
        "vram_free_gb": free_gb,
        "vram_total_gb": total_gb,
        "vram_used_gb": used_gb,
        "model_loaded": loaded_for_selected,
        "any_model_loaded": any_loaded,
        "loaded_model_id": target_id if target_id in LOADED_RUNTIMES else LOADED_MODEL_ID,
        "loaded_model_ids": loaded_ids,
        "loaded_models_detail": loaded_details,
        "current_model": summary_mode,
        "context_length": QWEN_CONTEXT_LENGTH,
        "loaded_context_length": loaded_ctx,
        "is_downloading": MODEL_DOWNLOAD_STATE["is_downloading"],
        "is_loading_vram": MODEL_DOWNLOAD_STATE["is_loading_vram"],
        "task_progress": MODEL_DOWNLOAD_STATE["progress"],
        "task_message": MODEL_DOWNLOAD_STATE["message"],
        "uptime_seconds": uptime_seconds,
        "cu_burn_rate_per_hour": cu_rate,
        "session_cu_used": session_cu_used,
        "idle_timeout_minutes": IDLE_TIMEOUT_MINUTES,
        "idle_remaining_seconds": idle_remaining_seconds,
        "api_key_enabled": bool(API_AUTH_KEY),
        "api_request_count": API_REQUEST_COUNT,
        "console_lines": get_live_console_Lines(),
    }
    for k in (
        "ram_total_gb", "ram_used_gb", "ram_free_gb",
        "disk_total_gb", "disk_used_gb", "disk_free_gb",
        "shm_total_gb", "shm_used_gb", "shm_free_gb",
        "drive_total_gb", "drive_used_gb", "drive_free_gb",
    ):
        base[k] = hw.get(k, 0.0)
    base.update(file_status)
    if _has_rclone_drive_remote():
        base["drive_persisted"] = True
    base["comfyui"] = get_comfyui_status_dict()
    return base


def get_gpu_and_model_summary_text() -> str:
    info = get_gpu_status_dict()
    lines = []
    if not info["cuda_available"]:
        lines.append("GPU 状态: 未检测到 GPU（请在 Colab 中切换为 A100 / L4 GPU）")
    else:
        lines.append(
            f"GPU: {info['gpu_name']} | 显存可用: {info['vram_free_gb']:.1f} / {info['vram_total_gb']:.1f} GB"
        )
    if info.get("ram_total_gb", 0) > 0:
        lines.append(
            f"实例内存 (RAM): 已用 {info['ram_used_gb']:.1f} GB | 可用 {info['ram_free_gb']:.1f} / {info['ram_total_gb']:.1f} GB"
        )
    if info.get("disk_total_gb", 0) > 0:
        shm_part = (
            f" | /dev/shm 内存盘可用: {info['shm_free_gb']:.1f} / {info['shm_total_gb']:.1f} GB"
            if info.get("shm_total_gb", 0) >= 1.0
            else ""
        )
        lines.append(
            f"实例磁盘: 已用 {info['disk_used_gb']:.1f} GB | 可用 {info['disk_free_gb']:.1f} / {info['disk_total_gb']:.1f} GB{shm_part}"
        )
    lines.append(
        f"算力计费: 费率 ≈ {info['cu_burn_rate_per_hour']:.2f} CU/小时 | 本次开机已用: {info['session_cu_used']:.2f} CU (运行 {info['uptime_seconds']//60} 分钟)"
    )
    ram_tag = " + /dev/shm 内存盘(2.4GB/s)" if info.get("ramdisk_enabled") else ""
    storage_tag = ("Google Drive 云盘持久化" if info["drive_persisted"] else "Colab 高速磁盘") + ram_tag
    lines.append(
        f"当前模型 [{info['selected_model_name']}] 权重 [{storage_tag}]: 已就绪 {info['downloaded_count']}/{info['total_count']} 个文件 ({info['total_size_gb']:.2f} GB)"
    )
    for f in info["files"][:8]:
        mark = "[已就绪]" if f["exists"] else "[未下载]"
        sz = f"({f['size_gb']:.2f} GB)" if f["exists"] else ""
        lines.append(f"   {mark} {f['name']} {sz}")
    if len(info["files"]) > 8:
        lines.append(f"   ... 其余 {len(info['files']) - 8} 个分片文件")

    vram_state = f"已加载进显存 ({info['current_model']})" if info["any_model_loaded"] else "尚未加载进显存"
    lines.append(f"显存引擎状态: {vram_state}")
    return "\n".join(lines)


def build_structured_prompt(
    visual_prompt: str,
    soundscape_prompt: str,
    music_prompt: str,
) -> str:
    visual_prompt = (visual_prompt or "").strip()
    soundscape_prompt = (soundscape_prompt or "").strip()
    music_prompt = (music_prompt or "").strip()

    if not soundscape_prompt and not music_prompt:
        return visual_prompt

    parts = [visual_prompt]
    if soundscape_prompt:
        parts.append(f"Soundscape & Dialogue: {soundscape_prompt}")
    if music_prompt:
        parts.append(f"Background Music: {music_prompt}")
    return "\n".join(parts)


# ==============================================================================
# 独立功能 1：多模型通用极速下载器 (8路并发 HTTP Range + 后台 Google Drive 备份)
# ==============================================================================
def _async_backup_to_drive(src_file: str, dst_dir: str, fname: str):
    """在后台线程中将内存盘/NVMe下载好的权重同步到 Google Drive，不阻塞用户加载模型"""
    if not DRIVE_MOUNTED:
        return
    try:
        dst_file = os.path.join(dst_dir, fname)
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        if os.path.exists(dst_file) and os.path.getsize(dst_file) == os.path.getsize(src_file):
            return
        tmp_dst = dst_file + ".syncing"
        with open(src_file, "rb") as rf, open(tmp_dst, "wb") as wf:
            shutil.copyfileobj(rf, wf, length=32 * 1024 * 1024)
        os.replace(tmp_dst, dst_file)
        print(f"[后台云盘同步完成] {fname} 已持久化保存至 Google Drive")
    except Exception as e:
        print(f"[后台云盘同步跳过] {fname}: {e}")


def download_model_weights_core(model_id: Optional[str] = None, hf_token: str = "") -> Tuple[bool, str]:
    global SELECTED_MODEL_ID
    touch_activity()
    if MODEL_DOWNLOAD_STATE["is_downloading"]:
        return False, "模型正在下载中，请稍候..."

    target_id = model_id if (model_id and model_id in MODEL_CATALOG) else SELECTED_MODEL_ID
    SELECTED_MODEL_ID = target_id
    spec = MODEL_CATALOG[target_id]
    repo_id = spec["repo_id"]
    req_files = spec["files"]

    MODEL_DOWNLOAD_STATE["is_downloading"] = True
    MODEL_DOWNLOAD_STATE["progress"] = 0.02
    MODEL_DOWNLOAD_STATE["message"] = f"正在为 {spec['name']} 启动 8 路并发 HTTP Range 极速下载通道..."
    with CONSOLE_LOCK:
        ACTIVE_TQDM_BARS.clear()

    try:
        if hf_token and hf_token.strip():
            os.environ["HF_TOKEN"] = hf_token.strip()
            from huggingface_hub import login
            login(token=hf_token.strip(), add_to_git_credential=False)

        ram_dir, nvme_dir, drive_dir = _get_model_dirs(target_id)
        drive_has_space = False
        if DRIVE_MOUNTED and os.path.exists("/content/drive/MyDrive"):
            try:
                drive_free_gb = shutil.disk_usage("/content/drive/MyDrive").free / (1024 ** 3)
                drive_has_space = drive_free_gb >= spec["approx_size_gb"] + 2.0
                if not drive_has_space:
                    print(
                        f"ℹ️ Google Drive 剩余空间 ({drive_free_gb:.1f} GB) 小于 {spec['name']} 所需容量 "
                        f"({spec['approx_size_gb']:.1f} GB)，自动改用实例本地高速 NVMe 磁盘 ({nvme_dir}) 存储！"
                    )
            except Exception:
                drive_has_space = False

        primary_dl_dir = ram_dir if ram_dir else (drive_dir if (DRIVE_MOUNTED and drive_has_space) else nvme_dir)
        os.makedirs(primary_dl_dir, exist_ok=True)
        total_files = len(req_files)

        for idx, (fname, desc) in enumerate(req_files):
            touch_activity()
            existing = _find_existing_model_file(target_id, fname)
            base_prog = idx / float(total_files + 1)
            MODEL_DOWNLOAD_STATE["progress"] = round(base_prog + 0.04, 2)

            if existing:
                # 若使用 Qwen3.8-27B，确保同一目录内有软链接以便 AutoModel 直接按目录加载
                if target_id == "qwen38_27b":
                    target_link = os.path.join(primary_dl_dir, fname)
                    os.makedirs(os.path.dirname(target_link), exist_ok=True)
                    if os.path.abspath(existing) != os.path.abspath(target_link) and not os.path.exists(target_link):
                        try:
                            os.symlink(existing, target_link)
                        except Exception:
                            pass
                size_gb = os.path.getsize(existing) / (1024 ** 3)
                MODEL_DOWNLOAD_STATE["message"] = f"[{idx+1}/{total_files}] {fname} 已就绪 ({size_gb:.2f} GB)，跳过下载"
                print(MODEL_DOWNLOAD_STATE["message"])
                continue

            MODEL_DOWNLOAD_STATE["message"] = f"[{idx+1}/{total_files}] 正在 8 线程并发下载 {fname} ({desc})..."
            print(MODEL_DOWNLOAD_STATE["message"])
            dl_path = fast_parallel_hf_download(
                repo_id=repo_id,
                filename=fname,
                dest_dir=primary_dl_dir,
                hf_token=hf_token,
                num_connections=8,
            )
            if DRIVE_MOUNTED and drive_has_space and primary_dl_dir != drive_dir:
                threading.Thread(
                    target=_async_backup_to_drive,
                    args=(dl_path, drive_dir, fname),
                    daemon=True,
                ).start()

        MODEL_DOWNLOAD_STATE["progress"] = 0.92
        if target_id == "qwen38_27b_gguf" or _is_custom_gguf_model(target_id):
            MODEL_DOWNLOAD_STATE["message"] = "正在预装并校验 llama.cpp CUDA 高速推理引擎..."
            _ensure_llama_server_binary()
        else:
            MODEL_DOWNLOAD_STATE["message"] = f"正在获取 {spec['name']} 的分词器与 Processor 配置文件..."
            from huggingface_hub import snapshot_download

            if target_id == "minimax_h3":
                from diffsynth.pipelines.minimax_h3_audio_video import ModelConfig
                proc_cfg = ModelConfig(
                    model_id="MiniMaxAI/MiniMax-H3",
                    origin_file_pattern="FL2VA/processor/",
                    download_source="huggingface",
                )
                proc_cfg.download_if_necessary()
            elif target_id == "qwen38_27b":
                snapshot_download(
                    repo_id="Qwen/Qwen3.8-27B",
                    local_dir=primary_dl_dir,
                    ignore_patterns=["*.safetensors", "*.bin", "*.pth"],
                )
            elif target_id == "qwen_image_21":
                from diffsynth.pipelines.qwen_image_21 import ModelConfig
                proc_cfg = ModelConfig(
                    model_id="Qwen/Qwen-Image-2.1",
                    origin_file_pattern="processor/",
                    download_source="huggingface",
                )
                proc_cfg.download_if_necessary()
            elif target_id == "z_image_turbo":
                from diffsynth.pipelines.z_image import ModelConfig
                tok_cfg = ModelConfig(
                    model_id="Tongyi-MAI/Z-Image-Turbo",
                    origin_file_pattern="tokenizer/",
                    download_source="huggingface",
                )
                tok_cfg.download_if_necessary()

        MODEL_DOWNLOAD_STATE["progress"] = 1.0
        status = check_model_files_status(target_id)
        msg = f"{spec['name']} 全部 {status['downloaded_count']}/{status['total_count']} 个权重文件已就绪（共 {status['total_size_gb']:.2f} GB），现在可点击「加载到显存」！"
        MODEL_DOWNLOAD_STATE["message"] = msg
        print(msg)
        return True, msg

    except Exception as e:
        err_msg = f"下载 {spec['name']} 权重出错: {str(e)}"
        MODEL_DOWNLOAD_STATE["message"] = err_msg
        print(err_msg)
        return False, err_msg
    finally:
        with _TELEMETRY_LOCK:
            _CACHED_FILE_STATUS_MAP.clear()
        MODEL_DOWNLOAD_STATE["is_downloading"] = False


def download_minimax_h3_weights_core(hf_token: str = "") -> Tuple[bool, str]:
    return download_model_weights_core("minimax_h3", hf_token)


# ==============================================================================
# 独立功能 2：多模型通用显存加载引擎 (自动卸载旧模型释放显存 + 零拷贝极速加载)
# ==============================================================================
def _resolve_and_stage_model_file(model_id: str, fname: str) -> str:
    """解析模型权重路径；若位于较慢的云盘且 /dev/shm 内存盘充裕，自动预热至内存盘"""
    ram_dir, nvme_dir, drive_dir = _get_model_dirs(model_id)
    existing = _find_existing_model_file(model_id, fname)
    if not existing:
        return os.path.join(drive_dir, fname)

    if ram_dir and not existing.startswith("/dev/shm/"):
        ram_target = os.path.join(ram_dir, fname)
        if os.path.exists(ram_target) and os.path.getsize(ram_target) == os.path.getsize(existing):
            return ram_target
        try:
            os.makedirs(os.path.dirname(ram_target), exist_ok=True)
            free_shm = shutil.disk_usage("/dev/shm").free
            file_sz = os.path.getsize(existing)
            if free_shm > file_sz + (2 * 1024 ** 3):
                t0 = time.time()
                print(f"正在将 {fname} ({file_sz/(1024**3):.2f} GB) 高速预热至 /dev/shm 内存盘...")
                tmp_ram = ram_target + ".staging"
                with open(existing, "rb") as rf, open(tmp_ram, "wb") as wf:
                    shutil.copyfileobj(rf, wf, length=32 * 1024 * 1024)
                os.replace(tmp_ram, ram_target)
                dt = max(0.01, time.time() - t0)
                print(f"{fname} 内存盘预热完成 ({file_sz/(1024**2)/dt:.0f} MB/s)")
                return ram_target
        except Exception as e:
            print(f"内存盘预热跳过 ({e})，直接读取原路径")
    return existing


def _stage_model_files_to_local_nvme(
    model_id: str,
    paths: List[str],
    rel_names: List[str],
    group_label: str,
) -> List[str]:
    """当分片仍需走磁盘映射且原文件位于慢速云盘 FUSE (/content/drive) 时，若本地 NVMe 空间充裕则整组顺序缓存至本地高速 SSD"""
    if not paths or len(paths) != len(rel_names):
        return paths
    _, nvme_dir, _ = _get_model_dirs(model_id)
    if not any(p.startswith("/content/drive/") for p in paths):
        return paths

    try:
        os.makedirs(nvme_dir, exist_ok=True)
        free_nvme_bytes = shutil.disk_usage(nvme_dir).free
        needed_bytes = 0
        stage_plan: List[Tuple[int, str, str, int]] = []
        result_paths = list(paths)

        for idx, (src_p, rel_name) in enumerate(zip(paths, rel_names)):
            if not src_p.startswith("/content/drive/") or not os.path.exists(src_p):
                continue
            src_sz = os.path.getsize(src_p)
            dst_p = os.path.join(nvme_dir, rel_name)
            if os.path.exists(dst_p) and os.path.getsize(dst_p) == src_sz:
                result_paths[idx] = dst_p
                continue
            needed_bytes += src_sz
            stage_plan.append((idx, src_p, dst_p, src_sz))

        if not stage_plan:
            return result_paths

        # 保留至少 22GB 本地磁盘安全余量，避免挤占视频导出与系统空间
        if free_nvme_bytes <= needed_bytes + (22 * 1024 ** 3):
            return result_paths

        total_gb = needed_bytes / (1024 ** 3)
        print(f"⚡ [NVMe 高速缓存] 正在将 {group_label} ({len(stage_plan)} 个分片 / {total_gb:.1f} GB) 从云盘顺序预热至本地 NVMe SSD...")
        t_all = time.time()
        for step_i, (idx, src_p, dst_p, src_sz) in enumerate(stage_plan, start=1):
            os.makedirs(os.path.dirname(dst_p), exist_ok=True)
            MODEL_DOWNLOAD_STATE["message"] = (
                f"正在将 {group_label} 高速缓存至本地 SSD ({step_i}/{len(stage_plan)})，消除后续换入延迟..."
            )
            tmp_dst = dst_p + ".staging"
            with open(src_p, "rb") as rf, open(tmp_dst, "wb") as wf:
                shutil.copyfileobj(rf, wf, length=32 * 1024 * 1024)
            os.replace(tmp_dst, dst_p)
            result_paths[idx] = dst_p
        dt_all = max(0.05, time.time() - t_all)
        print(f"⚡ [NVMe 高速缓存] {group_label} 本地 SSD 预热完成 ({total_gb / dt_all * 1024:.0f} MB/s)")
        return result_paths
    except Exception as e:
        print(f"ℹ️ 本地 NVMe 预热跳过 ({e})，继续读取原路径")
        return paths


def _make_noproxy_opener():
    """本地回环请求 (127.0.0.1 llama-server) 强制绕过环境变量代理"""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _build_llama_ld_env(install_dir: str) -> Dict[str, str]:
    """收集 llama.cpp 动态库目录 + PyTorch 内置 NVIDIA CUDA Runtime/cuBLAS 目录"""
    env = os.environ.copy()
    ld_dirs: List[str] = []
    if os.path.isdir(install_dir):
        for root, _, files in os.walk(install_dir):
            if any(f == "llama-server" or ".so" in f for f in files):
                if root not in ld_dirs:
                    ld_dirs.append(root)

    try:
        import site
        site_roots = list(site.getsitepackages()) + [site.getusersitepackages()]
    except Exception:
        site_roots = []
    site_roots.append(os.path.dirname(os.path.dirname(torch.__file__)))

    for sp in site_roots:
        nv_root = os.path.join(sp, "nvidia")
        if os.path.isdir(nv_root):
            for sub in os.listdir(nv_root):
                lib_d = os.path.join(nv_root, sub, "lib")
                if os.path.isdir(lib_d) and lib_d not in ld_dirs:
                    ld_dirs.append(lib_d)

    for sys_lib in ("/usr/lib64-nvidia", "/usr/local/nvidia/lib64", "/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu"):
        if os.path.isdir(sys_lib) and sys_lib not in ld_dirs:
            ld_dirs.append(sys_lib)

    prev_ld = env.get("LD_LIBRARY_PATH", "")
    if prev_ld:
        ld_dirs.append(prev_ld)
    env["LD_LIBRARY_PATH"] = ":".join(ld_dirs)
    return env


def _download_file_to_path(url: str, dst_path: str, desc: str = ""):
    req = urllib.request.Request(url, headers={"User-Agent": "EasyColab-Engine/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dst_path, "wb") as wf:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(4 * 1024 * 1024)
            if not chunk:
                break
            wf.write(chunk)
            done += len(chunk)
            if desc and total > 0:
                pct = int(done * 100 / total)
                MODEL_DOWNLOAD_STATE["message"] = f"{desc} ({done/(1024**2):.0f}/{total/(1024**2):.0f} MB · {pct}%)"


def _ensure_llama_server_binary() -> Tuple[str, Dict[str, str]]:
    """
    自动安装/校验官方 ggml-org/llama.cpp 预编译 llama-server。
    有 GPU 时用 ubuntu-cuda-12.x 变体（NVIDIA 驱动向下兼容 CUDA 12.x/13.x 全系，避免 CUDA 13.5 PTX 在 Colab 13.0 驱动上报 unsupported toolchain）；
    纯 CPU 实例用 ubuntu-x64 变体，免去 567MB 的 cudart 依赖和 libcuda 缺失导致的加载失败。
    """
    want_cuda = torch.cuda.is_available()
    install_dir = (
        "/content/llama_cpp_cuda"
        if os.path.exists("/content")
        else os.path.join(os.getcwd(), "llama_cpp_cuda")
    )
    os.makedirs(install_dir, exist_ok=True)

    def _find_server_bin() -> Optional[str]:
        for root, _, files in os.walk(install_dir):
            if "llama-server" in files:
                cand = os.path.join(root, "llama-server")
                try:
                    os.chmod(cand, 0o755)
                except Exception:
                    pass
                return cand
        return None

    def _has_cudart12() -> bool:
        if not want_cuda:
            # CPU 版 llama-server 不需要 cudart，有二进制就算齐
            return True
        for _, _, files in os.walk(install_dir):
            if any(f.startswith("libcudart.so.12") or f.startswith("libcublas.so.12") for f in files):
                return True
        return False

    def _verify_server_bin(bin_p: str, run_env: Dict[str, str]) -> Tuple[bool, str]:
        try:
            proc = subprocess.run(
                [bin_p, "--version"],
                env=run_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=15,
            )
            if proc.returncode != 0:
                return False, (proc.stdout or "").strip()
            ver_txt = (proc.stdout or "").strip()
            if torch.cuda.is_available():
                dev_proc = subprocess.run(
                    [bin_p, "--list-devices"],
                    env=run_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=15,
                )
                dev_txt = (dev_proc.stdout or "").strip()
                if dev_proc.returncode != 0 or "CUDA0" not in dev_txt:
                    return False, f"CUDA 设备未挂载 ({dev_txt[:200]})"
            return True, ver_txt
        except Exception as ex:
            return False, str(ex)

    existing_bin = _find_server_bin()
    if existing_bin and _has_cudart12():
        env = _build_llama_ld_env(install_dir)
        ok, _ = _verify_server_bin(existing_bin, env)
        if ok:
            return existing_bin, env

    shutil.rmtree(install_dir, ignore_errors=True)
    os.makedirs(install_dir, exist_ok=True)

    cuda_prefix = "ubuntu-cuda-12."
    cpu_suffix = "bin-ubuntu-x64.tar.gz"
    backend_tag = "CUDA 12" if want_cuda else "CPU"
    fallback_tag = "b12211"
    if want_cuda:
        main_url = f"https://github.com/ggml-org/llama.cpp/releases/download/{fallback_tag}/llama-{fallback_tag}-bin-ubuntu-cuda-12.8-x64.tar.gz"
        cudart_url: Optional[str] = f"https://github.com/ggml-org/llama.cpp/releases/download/{fallback_tag}/cudart-llama-{fallback_tag}-bin-ubuntu-cuda-12.8-x64.tar.gz"
    else:
        main_url = f"https://github.com/ggml-org/llama.cpp/releases/download/{fallback_tag}/llama-{fallback_tag}-bin-ubuntu-x64.tar.gz"
        cudart_url = None

    def _pick_assets(rel: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        found_main, found_rt = None, None
        for asset in rel.get("assets") or []:
            aname = str(asset.get("name") or "")
            aurl = str(asset.get("browser_download_url") or "")
            if not aurl or not aname.endswith("-x64.tar.gz"):
                continue
            if not aname.startswith("llama-"):
                # 只有 CUDA 变体配套 cudart，CPU 变体完全忽略它
                if want_cuda and aname.startswith("cudart-llama-") and cuda_prefix in aname:
                    found_rt = aurl
                continue
            if want_cuda and cuda_prefix in aname:
                found_main = aurl
            elif not want_cuda and aname.endswith(cpu_suffix):
                found_main = aurl
        return found_main, found_rt

    try:
        api_req = urllib.request.Request(
            "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=8",
            headers={"Accept": "application/vnd.github+json", "User-Agent": "EasyColab-Engine/1.0"},
        )
        with urllib.request.urlopen(api_req, timeout=10) as resp:
            releases = json.loads(resp.read().decode("utf-8"))
        for rel in releases:
            tag = str(rel.get("tag_name") or "")
            if not tag.startswith("b"):
                continue
            found_main, found_rt = _pick_assets(rel)
            if found_main:
                main_url = found_main
                if want_cuda and found_rt:
                    cudart_url = found_rt
                break
    except Exception as api_err:
        print(f"[GGUF 引擎] GitHub Releases 查询跳过，使用稳定镜像版本 {fallback_tag} ({api_err})")

    print(f"[GGUF 引擎] 正在下载官方预编译 llama.cpp {backend_tag} 推理后端: {os.path.basename(main_url)} ...")
    tar_path = os.path.join(install_dir, "llama_bin.tar.gz")
    _download_file_to_path(main_url, tar_path, desc=f"正在下载 llama.cpp {backend_tag} 引擎")
    subprocess.check_call(["tar", "-xzf", tar_path, "-C", install_dir])
    try:
        os.remove(tar_path)
    except Exception:
        pass

    if cudart_url:
        print(f"[GGUF 引擎] 正在下载配套 CUDA 12 Runtime 动态库 ({os.path.basename(cudart_url)})...")
        rt_tar = os.path.join(install_dir, "cudart_bin.tar.gz")
        _download_file_to_path(cudart_url, rt_tar, desc="正在下载 CUDA Runtime 动态库")
        subprocess.check_call(["tar", "-xzf", rt_tar, "-C", install_dir])
        try:
            os.remove(rt_tar)
        except Exception:
            pass

    bin_path = _find_server_bin()
    if not bin_path:
        raise RuntimeError(f"解压 llama.cpp {backend_tag} 归档后未找到 llama-server 可执行文件")

    env = _build_llama_ld_env(install_dir)
    ok, ver_out = _verify_server_bin(bin_path, env)
    if not ok:
        raise RuntimeError(f"llama-server {backend_tag} 启动自检失败: {ver_out[:300]}")

    first_line = ver_out.splitlines()[0] if ver_out else "ready"
    print(f"[GGUF 引擎] llama-server {backend_tag} 引擎校验通过 ({first_line})")
    return bin_path, env


def _pick_free_local_port() -> int:
    """在非临时端口段 (18080~18999) 选取空闲本地监听端口，规避 Colab 容器 ephemeral 端口段 (32768+) 抢占冲突。"""
    for cand in range(18080, 18999):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", cand))
                return cand
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _gguf_meta_peek(path: str, max_kv: int = 512) -> Dict[str, Any]:
    """轻量读取 GGUF 头部元数据，取 block_count / context_length / 架构名。只读前 16MB。"""
    out: Dict[str, Any] = {}
    try:
        with open(path, "rb") as f:
            head = f.read(16 * 1024 * 1024)
        if len(head) < 24 or head[:4] != b"GGUF":
            return out
        pos = 4

        def u32():
            nonlocal pos
            v = int.from_bytes(head[pos:pos + 4], "little")
            pos += 4
            return v

        def u64():
            nonlocal pos
            v = int.from_bytes(head[pos:pos + 8], "little")
            pos += 8
            return v

        def s():
            nonlocal pos
            n = u64()
            v = head[pos:pos + n].decode("utf-8", "replace")
            pos += n
            return v

        ver = u32()
        out["gguf_version"] = ver
        u64()  # tensor_count
        n_kv = u64()

        # 各类型的字节宽度，用于跳过不需要的值
        fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        for _ in range(min(n_kv, max_kv)):
            if pos >= len(head):
                break
            key = s()
            vtype = u32()
            if vtype in fixed:
                raw = head[pos:pos + fixed[vtype]]
                pos += fixed[vtype]
                if vtype in (4, 10) and raw:
                    val = int.from_bytes(raw, "little")
                    if key.endswith(".block_count") or key.endswith(".n_layer"):
                        out["n_layers"] = val
                    elif key.endswith(".context_length"):
                        out["context_length"] = val
                    elif key == "general.architecture":
                        pass
            elif vtype == 8:
                val = s()
                if key == "general.architecture":
                    out["architecture"] = val
            elif vtype == 9:
                itype = u32()
                cnt = u64()
                if itype == 8:
                    for _i in range(cnt):
                        s()
                elif itype in fixed:
                    pos += fixed[itype] * cnt
                else:
                    break
            else:
                break
            if out.get("n_layers") and out.get("context_length"):
                break
    except Exception:
        pass
    return out


def _start_qwen_gguf_server(
    gguf_path: str,
    mmproj_path: Optional[str],
    ctx_len: int,
    cur_free_gb: float,
    mtp_path: Optional[str] = None,
    model_id: str = "qwen38_27b_gguf",
    model_label: str = "Qwen3.8-27B-UD-Q4_K_M",
    weight_gb: float = 19.2,
    n_layers: int = 65,
    architecture: str = "",
    gpu_available: bool = True,
) -> Dict[str, Any]:
    """
    启动驻留 GPU 的 llama-server 实例。默认参数对应 Qwen3.8-27B-UD-Q4_K_M，
    但已通用化：任何 GGUF（内置或自定义注册）都通过 model_id / weight_gb / n_layers 驱动，
    自动按显存余量决定 KV 量化与 -ngl 层数，显存不足时逐级降配重试。
    gpu_available=False 时（CPU 实例）走纯 CPU 后端：-ngl 0，不传任何 CUDA 专属参数。
    """
    bin_path, env = _ensure_llama_server_binary()

    # KV Cache 估算：按每层每 token 的 KV 体积等比缩放（65 层 256K q8_0 ≈ 8.8 GB 为标定基准）
    layer_scale = max(1.0, n_layers) / 65.0
    q8_kv_gb = (ctx_len / 262144.0) * 8.8 * layer_scale
    q4_kv_gb = (ctx_len / 262144.0) * 4.6 * layer_scale

    # 权重 + KV 能否全量驻留 GPU（额外预留 1.2 GB 给 CUDA 上下文与计算缓冲）
    need_full = weight_gb + q8_kv_gb + 1.2
    if cur_free_gb >= need_full:
        primary_kv = "q8_0"
        primary_kv_gb = q8_kv_gb
    elif cur_free_gb >= weight_gb + q4_kv_gb + 1.2:
        primary_kv = "q4_0"
        primary_kv_gb = q4_kv_gb
    else:
        # 权重都装不下：用 q4 KV 并做部分层卸载（CPU 跑剩余层）
        primary_kv = "q4_0"
        primary_kv_gb = q4_kv_gb

    has_mtp_file = bool(mtp_path and os.path.exists(mtp_path))

    def _ngl_for(budget_gb: float) -> int:
        """按剩余显存算能放多少层到 GPU（层体积 = 总权重 / 层数）"""
        per_layer = max(0.05, weight_gb / max(1.0, float(n_layers)))
        avail = max(0.0, budget_gb - primary_kv_gb - 1.2)
        return int(min(n_layers, max(0, avail // per_layer)))

    if not gpu_available:
        # CPU 实例：全部层留在内存，KV 优先 q8_0，失败退 f16（部分 llama.cpp 构建的 CPU 后端不支持 K 量化）
        primary_ngl = 0
        primary_mtp = False
        attempts: List[Tuple[str, int, bool]] = [("q8_0", 0, False), ("f16", 0, False)]
    else:
        if cur_free_gb >= weight_gb + primary_kv_gb + 1.2:
            primary_ngl = 999  # llama.cpp 语义：全部层上 GPU
        else:
            # 至少留 4 层在 GPU，否则部分卸载意义有限
            primary_ngl = max(4, min(n_layers, _ngl_for(cur_free_gb)))

        primary_mtp = has_mtp_file and primary_ngl == 999 and (cur_free_gb >= weight_gb + primary_kv_gb + 2.4)
        attempts = [(primary_kv, primary_ngl, primary_mtp)]

    # q8 KV 差一点装不下 MTP 时，降 KV 精度换 MTP 全量驻留
    if has_mtp_file and not primary_mtp and (cur_free_gb >= weight_gb + q4_kv_gb + 2.4):
        attempts.insert(0, ("q4_0", 999, True))

    if primary_ngl == 999:
        if primary_mtp:
            attempts.append((primary_kv, 999, False))
        # 全量驻留失败时退到部分层 + q4 KV
        attempts.append(("q4_0", max(4, min(n_layers, _ngl_for(cur_free_gb))), False))

    opener = _make_noproxy_opener()
    last_err = ""

    for attempt_idx, (kv_type, ngl_val, use_mtp) in enumerate(attempts):
        port = _pick_free_local_port()
        cmd = [
            bin_path,
            "-m", gguf_path,
            "--fit", "off",
            "--host", "127.0.0.1",
            "--port", str(port),
            "-c", str(ctx_len),
            "--kv-unified",
            "--parallel", "1",
            "--cache-type-k", kv_type,
            "--cache-type-v", kv_type,
            "-b", "2048",
            "-ub", "1024",
            "--backend-sampling",
            "--samplers", "top_k;top_p;min_p;temperature",
            "--temp", "1.0",
            "--top-p", "0.95",
            "--top-k", "20",
            "--min-p", "0.0",
            "--repeat-penalty", "1.0",
            "--presence-penalty", "0.0",
            "--jinja",
            "-ngl", str(ngl_val),
        ]
        # flash-attn 在 CPU 与 CUDA 后端都受支持，统一保留
        cmd.extend(["--flash-attn", "on"])
        # reasoning-format 是 Qwen 系模板专用的 think 语法，其他架构传了会被 llama.cpp 报错
        if "qwen" in f"{model_label} {architecture}".lower():
            cmd.extend(["--reasoning-format", "deepseek"])
        if use_mtp and mtp_path:
            cmd.extend([
                "--model-draft", mtp_path,
                "-ngld", "999",
                "--spec-type", "draft-mtp",
                "--spec-draft-n-max", "4",
                "--spec-draft-p-min", "0.45",
                "--cache-type-k-draft", kv_type,
                "--cache-type-v-draft", kv_type,
            ])
        if mmproj_path and os.path.exists(mmproj_path):
            cmd.extend(["--mmproj", mmproj_path, "--image-min-tokens", "1024"])

        fit_desc = "全量 GPU 驻留" if ngl_val >= 99 else f"混合驻留 ({ngl_val}/{n_layers} 层 GPU)"
        mtp_desc = " + MTP 多词元投机解码" if use_mtp else ""
        print(
            f"[GGUF 引擎] 正在启动 {model_label}{mtp_desc} 服务 "
            f"(上下文 {ctx_len // 1024}K={ctx_len}, KV={kv_type}, GPU采样, {fit_desc}, port={port})..."
        )
        MODEL_DOWNLOAD_STATE["progress"] = 0.55
        MODEL_DOWNLOAD_STATE["message"] = (
            f"正在将 {model_label}{mtp_desc} 装载至 GPU ({ctx_len // 1024}K 上下文 · KV {kv_type})..."
        )

        recent_logs: List[str] = []
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def _drain_stdout(p: subprocess.Popen, log_buf: List[str]):
            try:
                assert p.stdout is not None
                for raw_line in p.stdout:
                    ln = raw_line.rstrip("\r\n")
                    if not ln:
                        continue
                    log_buf.append(ln)
                    if len(log_buf) > 100:
                        del log_buf[:25]
            except Exception:
                pass

        threading.Thread(target=_drain_stdout, args=(proc, recent_logs), daemon=True).start()

        health_url = f"http://127.0.0.1:{port}/health"
        deadline = time.time() + 210.0
        ready = False
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                with opener.open(health_url, timeout=2.0) as h_resp:
                    if h_resp.status == 200:
                        ready = True
                        break
            except urllib.error.HTTPError as he:
                # 503 表示 llama-server 正在加载权重到显存
                if he.code == 503:
                    MODEL_DOWNLOAD_STATE["progress"] = min(0.95, round(MODEL_DOWNLOAD_STATE["progress"] + 0.01, 2))
            except Exception:
                pass
            time.sleep(0.5)

        if ready:
            ctx_k_str = f"{ctx_len // 1024}K"
            ngl_tag = "" if ngl_val >= 99 else f" · {ngl_val}/{n_layers}层"
            mtp_tag = "+MTP" if use_mtp else ""
            mode_label = f"{model_label}{mtp_tag} · {ctx_k_str} (KV:{kv_type}{ngl_tag})"
            print(f"[GGUF 引擎] {model_label} 已就绪 ({mode_label})")
            return {
                "backend": "gguf",
                "model_id": model_id,
                "port": port,
                "base_url": f"http://127.0.0.1:{port}",
                "server_proc": proc,
                "context_length": ctx_len,
                "kv_type": kv_type,
                "mtp_enabled": use_mtp,
                "mode_label": mode_label,
                "gguf_path": gguf_path,
                "mmproj_path": mmproj_path,
                "mtp_path": mtp_path if use_mtp else None,
                "model_label": model_label,
                "architecture": architecture,
                "n_layers": n_layers,
                "ngl": ngl_val,
            }

        # 未就绪：清理子进程并记录日志
        try:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=3.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

        tail_txt = "\n".join(recent_logs[-12:])
        last_err = tail_txt or "llama-server 启动超时"
        if attempt_idx + 1 < len(attempts):
            print(f"[GGUF 引擎] 当前配置 (KV={kv_type}, ngl={ngl_val}, MTP={use_mtp}) 显存触顶，自动降级重试...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(0.5)

    raise RuntimeError(f"{model_label} GGUF 服务启动失败: {last_err[-400:]}")


def _sync_legacy_globals():
    """将多模型显存驻留表 LOADED_RUNTIMES 同步至兼容全局变量 PIPELINE / LLM_BUNDLE"""
    global PIPELINE, LLM_BUNDLE, LOADED_MODEL_ID, CURRENT_MODEL_MODE
    LLM_BUNDLE = None
    if _is_llm_model_id(SELECTED_MODEL_ID) and SELECTED_MODEL_ID in LOADED_RUNTIMES:
        LLM_BUNDLE = LOADED_RUNTIMES[SELECTED_MODEL_ID].get("llm_bundle")
    if LLM_BUNDLE is None:
        # 内置 Qwen 优先，其次任意已加载的自定义 GGUF（否则 /v1 网关会误报未加载）
        ordered = list(_QWEN_FAMILY_IDS) + [
            mid for mid in LOADED_RUNTIMES.keys() if mid in CUSTOM_MODEL_SPECS
        ]
        for llm_id in ordered:
            rt = LOADED_RUNTIMES.get(llm_id)
            if rt and rt.get("llm_bundle") is not None:
                LLM_BUNDLE = rt["llm_bundle"]
                break

    sel_rt = LOADED_RUNTIMES.get(SELECTED_MODEL_ID)
    if sel_rt and sel_rt.get("pipeline") is not None:
        PIPELINE = sel_rt["pipeline"]
    else:
        PIPELINE = None
        for rt in reversed(list(LOADED_RUNTIMES.values())):
            if rt.get("pipeline") is not None:
                PIPELINE = rt["pipeline"]
                break

    if SELECTED_MODEL_ID in LOADED_RUNTIMES:
        LOADED_MODEL_ID = SELECTED_MODEL_ID
    elif LOADED_RUNTIMES:
        LOADED_MODEL_ID = next(reversed(LOADED_RUNTIMES))
    else:
        LOADED_MODEL_ID = None

    if LOADED_RUNTIMES:
        CURRENT_MODEL_MODE = " + ".join(rt["mode_label"] for rt in LOADED_RUNTIMES.values())
    else:
        CURRENT_MODEL_MODE = None


def _trim_linux_cpu_heap():
    """在释放超大规模 CPU BF16 权重张量后主动触发 glibc malloc_trim(0)，将物理页立即归还给系统 MemAvailable"""
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass


def _unload_model_locked(model_id: Optional[str] = None) -> List[str]:
    """在已持有 GPU_LOCK 的前提下卸载指定模型（或全部模型），返回被卸载的模型名称列表"""
    global PIPELINE, LLM_BUNDLE, LOADED_MODEL_ID, CURRENT_MODEL_MODE
    removed_names: List[str] = []
    if not model_id or model_id == "all":
        for mid in list(LOADED_RUNTIMES.keys()):
            entry = LOADED_RUNTIMES.pop(mid, None)
            if entry:
                _stop_gguf_server_bundle(entry.get("llm_bundle"))
                removed_names.append(entry.get("name", mid))
                entry.clear()
        _stop_gguf_server_bundle(LLM_BUNDLE)
        PIPELINE = None
        LLM_BUNDLE = None
        LOADED_MODEL_ID = None
        CURRENT_MODEL_MODE = None
    elif model_id in LOADED_RUNTIMES:
        entry = LOADED_RUNTIMES.pop(model_id, None)
        if entry:
            _stop_gguf_server_bundle(entry.get("llm_bundle"))
            removed_names.append(entry.get("name", model_id))
            if PIPELINE is entry.get("pipeline"):
                PIPELINE = None
            if LLM_BUNDLE is entry.get("llm_bundle"):
                LLM_BUNDLE = None
            entry.clear()
        _sync_legacy_globals()

    _trim_linux_cpu_heap()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return removed_names


def _unload_all_gpu_models():
    """彻底释放全部已加载模型的 GPU 显存"""
    _unload_model_locked("all")


def unload_model_from_vram_core(model_id: Optional[str] = None) -> Tuple[bool, str]:
    """供客户端按需独立卸载单个或全部显存驻留模型（立即中断正在进行的推理任务以避免死锁）"""
    touch_activity()
    ABORT_GENERATION_EVENT.set()
    for jid, j in list(JOBS.items()):
        if j.get("status") in ("queued", "running") and (not model_id or model_id == "all" or j.get("model_id") == model_id):
            ev = _JOB_CANCEL_EVENTS.get(jid)
            if ev is not None:
                ev.set()
            j["status"] = "failed"
            j["progress"] = 0.0
            j["message"] = "已终止推理并从显存卸载该模型"
    with CONSOLE_LOCK:
        ACTIVE_TQDM_BARS.clear()
    with GPU_LOCK:
        try:
            removed = _unload_model_locked(model_id)
            free_gb = (torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)) if torch.cuda.is_available() else 0.0
            if not removed:
                msg = "当前指定模型未驻留显存，无需卸载。"
            else:
                remain_cnt = len(LOADED_RUNTIMES)
                remain_str = f"（显存中仍保留 {remain_cnt} 个常驻模型）" if remain_cnt > 0 else "（显存已完全清空）"
                msg = f"已从显存卸载 {', '.join(removed)}，当前可用显存 {free_gb:.1f} GB {remain_str}"
            MODEL_DOWNLOAD_STATE["message"] = msg
            print(f"[显存管理] {msg}")
            return True, msg
        finally:
            ABORT_GENERATION_EVENT.clear()


def _ensure_vram_for_new_model_locked(target_id: str, required_free_gb: float) -> List[str]:
    """
    多模型共驻智能显存守卫 (LRU Eviction)：
    - 若目标模型自身已在显存中，先卸载旧实例以便重载；
    - 若剩余可用显存 >= required_free_gb，保留所有已加载模型实现多模型同时常驻！
    - 仅当剩余显存不足以容纳新模型时，按最早加载顺序 (LRU) 自动腾退旧模型，防止 CUDA OOM。
    """
    evicted: List[str] = []
    if target_id in LOADED_RUNTIMES:
        _unload_model_locked(target_id)

    if not torch.cuda.is_available():
        return evicted

    free_gb = torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)
    while free_gb < required_free_gb and LOADED_RUNTIMES:
        oldest_id = next(iter(LOADED_RUNTIMES))
        names = _unload_model_locked(oldest_id)
        evicted.extend(names)
        free_gb = torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)
    return evicted


def load_model_to_vram_core(
    model_id: Optional[str] = None,
    precision_mode: str = "bf16",
    offload_mode: str = "disk",
    hf_token: str = "",
    context_length: Optional[int] = None,
) -> Tuple[bool, str]:
    global SELECTED_MODEL_ID, QWEN_CONTEXT_LENGTH
    touch_activity()

    if context_length is not None:
        QWEN_CONTEXT_LENGTH = _clamp_context_length(context_length)
    ctx_len = QWEN_CONTEXT_LENGTH

    if MODEL_DOWNLOAD_STATE["is_loading_vram"]:
        return False, "模型正在加载进显存中，请稍候..."

    target_id = model_id if (model_id and model_id in MODEL_CATALOG) else SELECTED_MODEL_ID
    has_gpu = torch.cuda.is_available()
    is_gguf_target = _is_gguf_model_id(target_id)
    if not has_gpu and not is_gguf_target:
        return False, (
            "未检测到 CUDA GPU！该模型必须以 BF16/FP16 常驻显卡，请在 Colab 中切换为 "
            "T4 / L4 / G4 / A100 / H100。GGUF 量化模型可直接在 CPU 实例上运行。"
        )

    SELECTED_MODEL_ID = target_id
    spec = MODEL_CATALOG[target_id]
    file_status = check_model_files_status(target_id)
    auto_switched_notice = ""

    if _is_gguf_model_id(target_id):
        # GGUF 只需主权重；mmproj 是可选的多模态投影器，缺失也允许加载（退化为纯文本）
        main_f, _mm_f = _gguf_spec_paths(target_id)
        if not _find_existing_model_file(target_id, main_f):
            return False, (
                f"{spec['name']} 主权重 {main_f} 尚未下载 "
                f"({file_status['downloaded_count']}/{file_status['total_count']})！请先点击「下载权重」。"
            )
    else:
        if not file_status["all_downloaded"]:
            return False, f"{spec['name']} 权重尚未下载完整 ({file_status['downloaded_count']}/{file_status['total_count']})！请先点击「下载权重」。"

    t_start = time.time()
    MODEL_DOWNLOAD_STATE["is_loading_vram"] = True
    MODEL_DOWNLOAD_STATE["progress"] = 0.12
    MODEL_DOWNLOAD_STATE["message"] = (
        f"正在评估显存余量并装载 {spec['name']} 至 GPU..."
        if has_gpu
        else f"正在以 CPU 模式启动 {spec['name']}（llama.cpp）..."
    )
    with CONSOLE_LOCK:
        ACTIVE_TQDM_BARS.clear()

    with GPU_LOCK:
        try:
            total_vram_gb = (torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3)) if has_gpu else 0.0
            vram_limit = max(8.0, total_vram_gb - 2.5) if has_gpu else 0.0

            # 计算新模型所需最小空闲显存阈值：余量充足时自动与已加载模型共驻显存！
            if not has_gpu:
                # 纯 CPU 实例只有 GGUF（llama.cpp）能跑，不需要显存额度，也不能腾退任何东西
                required_free_gb = 0.0
            elif target_id == "qwen38_27b":
                required_free_gb = 53.0
            elif is_gguf_target:
                # 按该 GGUF 实际权重体积 + KV 预留估算；显存不够时启动器会自动降为部分层卸载
                w_gb = float(spec.get("vram_gb") or spec.get("approx_size_gb") or 20.6)
                g_layer_scale = max(1.0, float(spec.get("n_layers") or 65)) / 65.0
                q4_kv_gb = (ctx_len / 262144.0) * 4.6 * g_layer_scale
                required_free_gb = min(max(4.5, w_gb + q4_kv_gb * 0.5), max(4.0, total_vram_gb - 1.2))
            elif target_id == "minimax_h3":
                # BF16 单阶段峰值约 62.1GB；在 G4 (96GB) 上可与 Qwen3.8-GGUF (~21GB) 同时共驻显存
                required_free_gb = min(64.5, max(24.0, total_vram_gb - 5.0))
            else:
                required_free_gb = 24.0

            if has_gpu and torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3) < required_free_gb and COMFYUI_STATE.get("is_running"):
                free_comfyui_vram_core()
            evicted_names = _ensure_vram_for_new_model_locked(target_id, required_free_gb)
            free_before_gb = (torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)) if has_gpu else 0.0

            vram_config = {
                "offload_dtype": "disk",
                "offload_device": "disk",
                "onload_dtype": "disk",
                "onload_device": "disk",
                "preparing_dtype": torch.bfloat16,
                "preparing_device": "cuda",
                "computation_dtype": torch.bfloat16,
                "computation_device": "cuda",
            }

            new_pipe = None
            new_llm = None
            mode_label = spec["name"]

            if target_id == "minimax_h3":
                from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig

                _trim_linux_cpu_heap()
                total_ram_gb, avail_ram_gb = _get_system_ram_gb()
                cpu_fast_vram_config = {
                    "offload_dtype": torch.bfloat16,
                    "offload_device": "cpu",
                    "onload_dtype": torch.bfloat16,
                    "onload_device": "cpu",
                    "preparing_dtype": torch.bfloat16,
                    "preparing_device": "cuda",
                    "computation_dtype": torch.bfloat16,
                    "computation_device": "cuda",
                }

                te_rel_names = [f"FL2VA/text_encoder/model-{i:05d}-of-00014.safetensors" for i in range(1, 15)]
                dit_rel_names = [f"FL2VA/transformer/model-{i:05d}-of-00013.safetensors" for i in range(1, 14)]

                MODEL_DOWNLOAD_STATE["progress"] = 0.22
                MODEL_DOWNLOAD_STATE["message"] = "正在映射 MiniMax-H3 BF16 多模态编码器权重 (14 分片)..."
                te_paths = [_resolve_and_stage_model_file("minimax_h3", fn) for fn in te_rel_names]

                MODEL_DOWNLOAD_STATE["progress"] = 0.42
                MODEL_DOWNLOAD_STATE["message"] = "正在映射 MiniMax-H3 BF16 33B DiT 主干权重 (13 分片)..."
                dit_paths = [_resolve_and_stage_model_file("minimax_h3", fn) for fn in dit_rel_names]

                MODEL_DOWNLOAD_STATE["progress"] = 0.58
                MODEL_DOWNLOAD_STATE["message"] = "正在映射 3D 视频 VAE 与 32kHz 立体声 VAE 权重..."
                vvae_path = _resolve_and_stage_model_file(
                    "minimax_h3", "FL2VA/video_vae/source/model.safetensors"
                )
                avae_path = _resolve_and_stage_model_file(
                    "minimax_h3", "FL2VA/audio_vae/model.safetensors"
                )

                # 基于实例可用系统内存自适应选择热备策略（严守 >=24GB CPU 内存安全余量，杜绝 OOM）：
                # - Tier 1 (如 G4 176GB RAM, 可用 >=148GB)：text_encoder(62.1GB) + dit(61.7GB) 双主干常驻系统内存，PCIe 极速互换 (~4-6s)！
                # - Tier 2 (可用 88~148GB RAM)：text_encoder(62.1GB) 常驻系统内存 (~4-6s 极速解析提示词)，dit 顺序缓存至本地高速 NVMe！
                # - Tier 3 (低内存机型 <88GB RAM)：保持磁盘按需映射，若本地 NVMe 空间充裕则脱离慢速云盘 FUSE。
                ram_residency_tag = ""
                if avail_ram_gb >= 148.0:
                    te_cfg = cpu_fast_vram_config
                    dit_cfg = cpu_fast_vram_config
                    vae_cfg = cpu_fast_vram_config
                    ram_residency_tag = "双主干内存热备"
                    print(
                        f"⚡ [G4 大内存热备] 检测到充裕系统内存 ({avail_ram_gb:.1f}/{total_ram_gb:.1f} GB)，"
                        "已启用 32B 文本编码器 + 33B DiT 双主干 DDR5 内存常驻热备（阶段切换仅需 ~5 秒）！"
                    )
                    MODEL_DOWNLOAD_STATE["progress"] = 0.68
                    MODEL_DOWNLOAD_STATE["message"] = (
                        f"正在将 MiniMax-H3 预装载至 {total_ram_gb:.0f}GB 高速系统内存热备池（首次载入后换提示词仅需 ~5 秒）..."
                    )
                elif avail_ram_gb >= 88.0:
                    te_cfg = cpu_fast_vram_config
                    dit_paths = _stage_model_files_to_local_nvme(
                        "minimax_h3", dit_paths, dit_rel_names, "MiniMax-H3 33B DiT 主干"
                    )
                    dit_cfg = vram_config
                    vae_cfg = cpu_fast_vram_config
                    ram_residency_tag = "编码器内存热备"
                    print(
                        f"⚡ [混合内存热备] 系统可用内存 {avail_ram_gb:.1f}/{total_ram_gb:.1f} GB，"
                        "已将 32B 文本编码器锁入系统内存热备（换提示词 ~5 秒极速换入），DiT 走本地高速存储！"
                    )
                    MODEL_DOWNLOAD_STATE["progress"] = 0.68
                    MODEL_DOWNLOAD_STATE["message"] = (
                        "正在将 32B 文本编码器装载至系统内存热备池（后续解析新提示词仅需 ~5 秒）..."
                    )
                else:
                    te_paths = _stage_model_files_to_local_nvme(
                        "minimax_h3", te_paths, te_rel_names, "MiniMax-H3 多模态编码器"
                    )
                    dit_paths = _stage_model_files_to_local_nvme(
                        "minimax_h3", dit_paths, dit_rel_names, "MiniMax-H3 33B DiT 主干"
                    )
                    te_cfg = vram_config
                    dit_cfg = vram_config
                    vae_cfg = vram_config
                    MODEL_DOWNLOAD_STATE["progress"] = 0.72
                    MODEL_DOWNLOAD_STATE["message"] = "正在构建 MiniMax-H3 BF16 原生 GPU 计算图与 Processor..."

                new_pipe = MiniMaxH3Pipeline.from_pretrained(
                    torch_dtype=torch.bfloat16,
                    device="cuda",
                    model_configs=[
                        ModelConfig(path=te_paths, **te_cfg),
                        ModelConfig(path=dit_paths, **dit_cfg),
                        ModelConfig(path=vvae_path, **vae_cfg),
                        ModelConfig(path=avae_path, **vae_cfg),
                    ],
                    processor_config=ModelConfig(
                        model_id="MiniMaxAI/MiniMax-H3",
                        origin_file_pattern="FL2VA/processor/",
                        download_source="huggingface",
                    ),
                    vram_limit=vram_limit,
                )
                mode_label = "MiniMax-H3 · BF16 原生"
                residency_tag = _enable_high_vram_optimizations(
                    new_pipe,
                    total_vram_gb=total_vram_gb,
                    preload_vram=True,
                    skip_preload_attrs=("dit",) if free_before_gb >= 76.0 else ("text_encoder", "dit"),
                )
                tags = [t for t in (ram_residency_tag, residency_tag) if t]
                if tags:
                    mode_label = f"{mode_label} · {' / '.join(tags)}"

            elif target_id == "qwen_image_21":
                from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline, ModelConfig
                MODEL_DOWNLOAD_STATE["progress"] = 0.45
                MODEL_DOWNLOAD_STATE["message"] = "正在映射 Qwen-Image-2.1 权重并构建 GPU 管线..."
                dit_paths = [
                    _resolve_and_stage_model_file(target_id, f"transformer/diffusion_pytorch_model-{i:05d}-of-00002.safetensors")
                    for i in range(1, 3)
                ]
                te_paths = [
                    _resolve_and_stage_model_file(target_id, f"text_encoder/model-{i:05d}-of-00004.safetensors")
                    for i in range(1, 5)
                ]
                vae_path = _resolve_and_stage_model_file(target_id, "vae/diffusion_pytorch_model.safetensors")
                new_pipe = QwenImage21Pipeline.from_pretrained(
                    torch_dtype=torch.bfloat16,
                    device="cuda",
                    model_configs=[
                        ModelConfig(path=dit_paths, **vram_config),
                        ModelConfig(path=te_paths, **vram_config),
                        ModelConfig(path=vae_path, **vram_config),
                    ],
                    processor_config=ModelConfig(
                        model_id="Qwen/Qwen-Image-2.1",
                        origin_file_pattern="processor/",
                        download_source="huggingface",
                    ),
                    vram_limit=vram_limit,
                )
                _enable_high_vram_optimizations(new_pipe, total_vram_gb=total_vram_gb, preload_vram=True)
                mode_label = "Qwen-Image-2.1 · BF16"

            elif target_id == "z_image_turbo":
                from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
                MODEL_DOWNLOAD_STATE["progress"] = 0.45
                MODEL_DOWNLOAD_STATE["message"] = "正在映射 Z-Image-Turbo 权重并构建 8 步极速生图管线..."
                dit_paths = [
                    _resolve_and_stage_model_file(target_id, f"transformer/diffusion_pytorch_model-{i:05d}-of-00003.safetensors")
                    for i in range(1, 4)
                ]
                te_paths = [
                    _resolve_and_stage_model_file(target_id, f"text_encoder/model-{i:05d}-of-00003.safetensors")
                    for i in range(1, 4)
                ]
                vae_path = _resolve_and_stage_model_file(target_id, "vae/diffusion_pytorch_model.safetensors")
                new_pipe = ZImagePipeline.from_pretrained(
                    torch_dtype=torch.bfloat16,
                    device="cuda",
                    model_configs=[
                        ModelConfig(path=dit_paths, **vram_config),
                        ModelConfig(path=te_paths, **vram_config),
                        ModelConfig(path=vae_path, **vram_config),
                    ],
                    tokenizer_config=ModelConfig(
                        model_id="Tongyi-MAI/Z-Image-Turbo",
                        origin_file_pattern="tokenizer/",
                        download_source="huggingface",
                    ),
                    vram_limit=vram_limit,
                )
                _enable_high_vram_optimizations(new_pipe, total_vram_gb=total_vram_gb, preload_vram=True)
                mode_label = "Z-Image-Turbo · 8步极速 BF16"

            elif target_id == "qwen38_27b":
                if free_before_gb < 49.0:
                    raise RuntimeError(
                        f"当前 GPU 可用显存 ({free_before_gb:.1f} GB / 总计 {total_vram_gb:.1f} GB) 不足以加载 "
                        f"Qwen3.8-27B BF16 原生权重（需约 52 GB 显存，推荐 A100 80GB）。"
                        f"若在 L4 (24GB) 或中低显存运行，请在模型中心选择独立的「Qwen3.8-27B-GGUF (UD-Q4_K_M)」量化版！"
                    )

                MODEL_DOWNLOAD_STATE["progress"] = 0.30
                MODEL_DOWNLOAD_STATE["message"] = (
                    f"正在装载 Qwen3.8-27B BF16 原生满血权重 (上下文上限 {ctx_len // 1024}K)..."
                )
                first_shard = _find_existing_model_file("qwen38_27b", "model-00001-of-00018.safetensors")
                model_dir = os.path.dirname(first_shard) if first_shard else _get_model_dirs("qwen38_27b")[2]

                # 确保配置文件在同一目录中完备
                if not os.path.exists(os.path.join(model_dir, "config.json")):
                    from huggingface_hub import snapshot_download
                    snapshot_download(
                        repo_id="Qwen/Qwen3.8-27B",
                        local_dir=model_dir,
                        ignore_patterns=["*.safetensors", "*.bin", "*.pth"],
                    )

                def _load_qwen_vl_bf16():
                    _ensure_torchaudio_compatible()
                    _ensure_qwen_fast_kernels()
                    from transformers import AutoProcessor, AutoModelForImageTextToText
                    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
                    mdl = AutoModelForImageTextToText.from_pretrained(
                        model_dir,
                        dtype=torch.bfloat16,
                        attn_implementation="sdpa",
                        device_map="cuda",
                        trust_remote_code=True,
                    )
                    mode_tag = f"Qwen3.8-27B · BF16 原生 · {ctx_len // 1024}K"
                    print("[加速引擎] 已启用 FLA Triton 线性注意力 + Triton 融合 CausalConv1d + PyTorch SDPA！")
                    mdl.eval()
                    _optimize_qwen35_for_fast_inference(mdl, max_context_len=ctx_len)
                    return mdl, processor, mode_tag

                try:
                    mdl, proc, mode_tag = _load_qwen_vl_bf16()
                except (KeyError, ValueError, ImportError) as up_e:
                    print(f"检测到旧版 transformers ({up_e})，正在自动升级最新 transformers...")
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate", "qwen-vl-utils"]
                    )
                    import importlib
                    import transformers
                    importlib.reload(transformers)
                    mdl, proc, mode_tag = _load_qwen_vl_bf16()

                new_llm = {
                    "backend": "transformers",
                    "model": mdl,
                    "processor": proc,
                    "model_id": "qwen38_27b",
                    "context_length": ctx_len,
                }
                mode_label = mode_tag

            elif _is_gguf_model_id(target_id):
                MODEL_DOWNLOAD_STATE["progress"] = 0.25
                main_rel, mmproj_rel = _gguf_spec_paths(target_id)
                MODEL_DOWNLOAD_STATE["message"] = f"正在预热 {os.path.basename(main_rel)} 与视觉投影器..."
                gguf_path = _resolve_and_stage_model_file(target_id, main_rel)
                mmproj_path = _resolve_and_stage_model_file(target_id, mmproj_rel) if mmproj_rel else None
                if mmproj_path and not os.path.exists(mmproj_path):
                    mmproj_path = None

                # 读 GGUF 头部拿真实层数与原生上下文，用于精确算 -ngl 和 KV 显存
                meta = _gguf_meta_peek(gguf_path)
                n_layers = int(meta.get("n_layers") or spec.get("n_layers") or 65)
                native_ctx = int(meta.get("context_length") or spec.get("default_context") or ctx_len)
                eff_ctx = min(ctx_len, native_ctx)

                mtp_path: Optional[str] = None
                if target_id == "qwen38_27b_gguf":
                    # 仅内置 Unsloth Qwen3.8 有配套 MTP 投机解码头
                    mtp_rel = "MTP/mtp-Qwen3.8-27B-Q4_0.gguf"
                    try:
                        if not _find_existing_model_file(target_id, mtp_rel):
                            MODEL_DOWNLOAD_STATE["progress"] = 0.38
                            MODEL_DOWNLOAD_STATE["message"] = "正在自动补齐 Unsloth MTP 投机解码加速头 (1.31 GB)..."
                            print("[GGUF 引擎] 检测到尚未缓存 MTP 投机解码权重，正在 8 路并发极速补齐...")
                            ram_d, nvme_d, drive_d = _get_model_dirs(target_id)
                            dl_dir = ram_d if ram_d else (drive_d if DRIVE_MOUNTED else nvme_d)
                            os.makedirs(dl_dir, exist_ok=True)
                            dl_mtp = fast_parallel_hf_download(
                                repo_id=spec["repo_id"],
                                filename=mtp_rel,
                                dest_dir=dl_dir,
                                hf_token=os.environ.get("HF_TOKEN", ""),
                                num_connections=8,
                            )
                            if DRIVE_MOUNTED and dl_dir != drive_d:
                                threading.Thread(
                                    target=_async_backup_to_drive,
                                    args=(dl_mtp, drive_d, mtp_rel),
                                    daemon=True,
                                ).start()
                        mtp_path = _resolve_and_stage_model_file(target_id, mtp_rel)
                    except Exception as mtp_e:
                        print(f"[GGUF 引擎] MTP 投机权重获取跳过，将以标准模式启动: {mtp_e}")
                        mtp_path = None

                weight_gb = round(os.path.getsize(gguf_path) / (1024 ** 3), 2)
                new_llm = _start_qwen_gguf_server(
                    gguf_path=gguf_path,
                    mmproj_path=mmproj_path,
                    ctx_len=eff_ctx,
                    cur_free_gb=free_before_gb,
                    mtp_path=mtp_path,
                    model_id=target_id,
                    model_label=spec["name"],
                    weight_gb=weight_gb,
                    n_layers=n_layers,
                    architecture=str(meta.get("architecture") or ""),
                    gpu_available=has_gpu,
                )
                mode_label = new_llm["mode_label"]

            if has_gpu:
                free_after_gb = torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)
                vram_delta_gb = round(max(0.5, free_before_gb - free_after_gb), 1)
            else:
                # CPU 模式没有显存概念，用权重文件体积作为占用量展示
                vram_delta_gb = 0.0

            LOADED_RUNTIMES[target_id] = {
                "model_id": target_id,
                "name": spec["name"],
                "modality": spec["modality"],
                "mode_label": mode_label,
                "pipeline": new_pipe,
                "llm_bundle": new_llm,
                "vram_gb": vram_delta_gb,
                "loaded_at": time.time(),
            }
            _sync_legacy_globals()

            elapsed_s = round(time.time() - t_start, 1)
            MODEL_DOWNLOAD_STATE["progress"] = 1.0
            co_count = len(LOADED_RUNTIMES)
            evict_tip = f"（因显存上限已自动腾退 {', '.join(evicted_names)}）" if evicted_names else ""
            co_tip = f" | 当前共 {co_count} 个模型同时常驻显存，切换 0 等待" if co_count > 1 else ""
            if has_gpu:
                msg = f"{spec['name']} 已装载至显存 ({mode_label}，约 {vram_delta_gb}GB，耗时 {elapsed_s}s){co_tip}！{evict_tip}{auto_switched_notice}"
            else:
                msg = f"{spec['name']} 已以 CPU 模式就绪 ({mode_label}，耗时 {elapsed_s}s)，无 GPU 走 llama.cpp 纯 CPU 推理！{auto_switched_notice}"
            MODEL_DOWNLOAD_STATE["message"] = msg
            print(msg)
            return True, msg

        except Exception as e:
            _sync_legacy_globals()
            err = f"加载 {spec['name']} 至显存失败: {str(e)}"
            MODEL_DOWNLOAD_STATE["message"] = err
            print(err)
            return False, err
        finally:
            MODEL_DOWNLOAD_STATE["is_loading_vram"] = False
            touch_activity()


def load_minimax_h3_to_vram_core(
    precision_mode: str = "bf16",
    offload_mode: str = "fast_mmap",
    hf_token: str = "",
) -> Tuple[bool, str]:
    return load_model_to_vram_core("minimax_h3", "bf16", offload_mode, hf_token)


def _enable_high_vram_optimizations(
    pipe,
    total_vram_gb: float,
    preload_vram: bool = True,
    skip_preload_attrs: Tuple[str, ...] = (),
) -> str:
    """
    针对 G4 (96GB) / A100 (80GB) 等大显存实例 (>45GB VRAM) 的深度加速（完美兼容多模型显存共驻与 BF16 超大模型）：
    1. 拦截 DiffSynth 默认在每个阶段切换时强制触发的 module.offload()，基于实时剩余显存智能判断：
       - 对于 MiniMax-H3 BF16 (text_encoder 62.1GB + dit 61.7GB > 96GB)，在 text_encoder 与 dit 互斥切换时精准腾退，
         而在 dit -> video_vae (9.7GB) -> audio_vae (0.56GB) 阶段只要余量 >= 25GB 即全量常驻，连抽换种子 0 重载！
    2. 为 Qwen3-VL 32B PromptEmbedder 挂载零触发 LRU 缓存：命中缓存时连 load_models_to_device(["text_encoder"]) 亦直接跳过，
       保证 61.7GB DiT 主干在 G4 96GB 显存中全程锁死不卸载！
    """
    import types

    is_minimax_bf16 = pipe.__class__.__name__ == "MiniMaxH3Pipeline"

    # 1. 挂载 PromptEmbedder 编码缓存（连抽多条或同 Prompt 换 Seed 时 0ms 瞬时跳过 32B Qwen3-VL 推理及显存换入）
    try:
        for unit in getattr(pipe, "units", []):
            if unit.__class__.__name__ == "MiniMaxH3Unit_PromptEmbedder" and not hasattr(
                unit, "_orig_process"
            ):
                unit._orig_process = unit.process
                unit._orig_onload_names = getattr(unit, "onload_model_names", ("text_encoder",))
                # 置空外层自动 onload，仅在缓存未命中时按需加载 text_encoder，避免同提示词连抽时把常驻 DiT 挤出显存
                unit.onload_model_names = ()
                unit._embed_cache = {}

                def _cached_prompt_process(
                    self_unit,
                    pipe_obj,
                    prompt,
                    keyframes=None,
                    ref_blocks=None,
                    height=None,
                    width=None,
                    text_embedding=None,
                ):
                    if ref_blocks is None and text_embedding is None and prompt is not None:
                        kf_sig = None
                        if keyframes:
                            try:
                                kf_sig = tuple(
                                    (img.size, img.mode, img.tobytes()[:512]) for img in keyframes
                                )
                            except Exception:
                                kf_sig = id(keyframes)
                        cache_key = (prompt, height, width, kf_sig)
                        cached = self_unit._embed_cache.get(cache_key)
                        if cached is not None:
                            print("⚡ [PromptCache] 命中 Qwen3-VL 32B 多模态编码缓存，0ms 瞬时跳过文本编码阶段！")
                            return {
                                "prompt_embeds": cached["prompt_embeds"].to(
                                    pipe_obj.device, pipe_obj.torch_dtype
                                ),
                                "text_token_tags": cached["text_token_tags"].to(
                                    pipe_obj.device, torch.long
                                ),
                            }
                        t_pe = time.time()
                        pipe_obj.load_models_to_device(self_unit._orig_onload_names)
                        res = self_unit._orig_process(
                            pipe_obj,
                            prompt,
                            keyframes=keyframes,
                            ref_blocks=ref_blocks,
                            height=height,
                            width=width,
                            text_embedding=text_embedding,
                        )
                        self_unit._embed_cache.clear()
                        self_unit._embed_cache[cache_key] = {
                            "prompt_embeds": res["prompt_embeds"].detach().clone(),
                            "text_token_tags": res["text_token_tags"].detach().clone(),
                        }
                        print(
                            f"⚡ [PromptEmbedder] 多模态提示词解析完成，总耗时 {time.time() - t_pe:.1f}s"
                            "（已写入缓存，同提示词换种子后续 0ms）"
                        )
                        return res
                    t_pe = time.time()
                    pipe_obj.load_models_to_device(self_unit._orig_onload_names)
                    res = self_unit._orig_process(
                        pipe_obj,
                        prompt,
                        keyframes=keyframes,
                        ref_blocks=ref_blocks,
                        height=height,
                        width=width,
                        text_embedding=text_embedding,
                    )
                    print(f"⚡ [PromptEmbedder] 多模态提示词解析完成，总耗时 {time.time() - t_pe:.1f}s")
                    return res

                unit.process = types.MethodType(_cached_prompt_process, unit)
    except Exception as e:
        print(f"ℹ️ Prompt 缓存挂载跳过: {e}")

    if total_vram_gb < 45.0:
        return ""

    def _is_submodel_fully_prepared_on_cuda(mdl) -> bool:
        if mdl is None:
            return False
        found_stateful = False
        for mod in mdl.modules():
            if hasattr(mod, "state"):
                found_stateful = True
                if getattr(mod, "state", 0) != 2:
                    return False
        if found_stateful:
            return True
        p = next(mdl.parameters(), None)
        return bool(p is not None and p.device.type == "cuda")

    def _offload_submodel_to_host(mdl) -> bool:
        if mdl is None or not getattr(mdl, "vram_management_enabled", False):
            return False
        has_active = any(
            hasattr(mod, "state") and getattr(mod, "state", 0) != 0
            for mod in mdl.modules()
        )
        if not has_active:
            return False
        if hasattr(mdl, "offload"):
            mdl.offload()
        else:
            for mod in mdl.modules():
                if hasattr(mod, "offload"):
                    mod.offload()
        return True

    # 2. 替换 load_models_to_device：智能阶段显存驻留 + 整块 PCIe 极速换入
    def _smart_load_models_to_device(self_pipe, model_names):
        if not getattr(self_pipe, "vram_management_enabled", False):
            return
        target_set = set(model_names or ())
        if not target_set:
            return

        # 快速路径：若请求的所有目标子模块均已 100% 处于 GPU 显存 (state=2)，0ms 直接放行，绝不误卸其他常驻模块
        target_children = [
            (name, mdl)
            for name, mdl in self_pipe.named_children()
            if name in target_set and mdl is not None
        ]
        if target_children and all(_is_submodel_fully_prepared_on_cuda(mdl) for _, mdl in target_children):
            return

        def _get_free_vram_gb() -> float:
            try:
                return torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)
            except Exception:
                return 0.0

        free_gb = _get_free_vram_gb()
        need_heavy_backbone = is_minimax_bf16 and (("dit" in target_set) or ("text_encoder" in target_set))
        req_free_gb = 66.5 if need_heavy_backbone else 24.0

        # 第一级腾退：切换 MiniMax-H3 两大 62GB 主干时，优先仅腾退对侧闲置大主干，让轻量的 video_vae / audio_vae 在 G4 96GB 中全程钉死显存
        if need_heavy_backbone and free_gb < req_free_gb:
            offloaded_heavy = False
            for name, model in self_pipe.named_children():
                if name not in target_set and name in ("dit", "text_encoder") and model is not None:
                    if _offload_submodel_to_host(model):
                        offloaded_heavy = True
            if offloaded_heavy:
                gc.collect()
                torch.cuda.empty_cache()
                free_gb = _get_free_vram_gb()

        # 第二级腾退：仅当腾退大主干后显存仍不足 (例如与 21GB Qwen-GGUF 共驻显存时)，再按需腾退其余非目标子模块
        if free_gb < req_free_gb:
            offloaded_rest = False
            for name, model in self_pipe.named_children():
                if name not in target_set and model is not None:
                    if _offload_submodel_to_host(model):
                        offloaded_rest = True
            if offloaded_rest:
                gc.collect()
                torch.cuda.empty_cache()
                free_gb = _get_free_vram_gb()

        # 目标模块整块换入：显存充裕时直接一次性执行 preparing() 锁入 GPU (state=2)，彻底消灭推理时 800+ 层逐层同步钩子开销
        for name, model in target_children:
            if _is_submodel_fully_prepared_on_cuda(model):
                continue
            if getattr(model, "vram_management_enabled", False):
                t_sw = time.time()
                if hasattr(model, "onload"):
                    model.onload()
                else:
                    for module in model.modules():
                        if hasattr(module, "onload"):
                            module.onload()

                cur_free_gb = _get_free_vram_gb()
                bulk_need_gb = 66.0 if (is_minimax_bf16 and name in ("dit", "text_encoder")) else 22.0
                if cur_free_gb >= bulk_need_gb:
                    for module in model.modules():
                        if hasattr(module, "preparing") and getattr(module, "state", 2) != 2:
                            try:
                                module.preparing()
                            except Exception:
                                pass
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    dt_sw = time.time() - t_sw
                    if dt_sw >= 0.3:
                        print(f"⚡ [极速显存换入] 子模块 {name} 已整块载入 GPU 显存，耗时 {dt_sw:.1f}s")

    pipe.load_models_to_device = types.MethodType(_smart_load_models_to_device, pipe)
    pipe._high_vram_fast_mode = True

    # 3. 预热指定子模块至 GPU 显存 (state=2)，保留 >=22GB 安全激活余量
    if preload_vram:
        children = [
            (name, mdl)
            for name, mdl in pipe.named_children()
            if mdl is not None and name not in skip_preload_attrs
        ]
        for idx, (attr_name, model) in enumerate(children):
            free_gb = torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)
            if free_gb < 22.0:
                break
            prog = round(0.80 + 0.18 * ((idx + 1) / max(1, len(children))), 2)
            MODEL_DOWNLOAD_STATE["progress"] = prog
            MODEL_DOWNLOAD_STATE["message"] = f"[显存常驻加速] 正在将子模块 {attr_name} 预热锁入 GPU 显存..."
            for module in model.modules():
                if hasattr(module, "preparing") and getattr(module, "state", 2) != 2:
                    try:
                        module.preparing()
                    except Exception:
                        pass
        _trim_linux_cpu_heap()
        used_gb = (
            torch.cuda.mem_get_info("cuda")[1] - torch.cuda.mem_get_info("cuda")[0]
        ) / (1024 ** 3)
        print(
            f"[大显存常驻已激活] 管线子模块已预热至显存 (当前总驻留 {used_gb:.1f} GB / {total_vram_gb:.1f} GB)"
        )
        return "大显存常驻"

    return "大显存常驻加速"


def _reclaim_transient_gpu_memory():
    """在连续视频/图像生成前后主动回收 Python 引用与 CUDA 缓存碎片，杜绝第 2 条视频中途 OOM 崩溃"""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _run_pipeline_with_smart_tiling(pipe, call_kwargs: dict):
    """仅当实时空闲显存 >= 46GB 时启用整帧 VAE 解码 (tiled=False)；超长帧序列 (如 15s / 362帧) 触顶时自动开启梯度检查点兜底杜绝 OOM"""
    # 若生成高分辨率超长视频 (如 720P + >=260帧/11~15秒)，采样前若 video_vae 驻留显存且总空闲偏紧，先释放 VAE 留足 33GB+ 激活空间
    num_f = int(call_kwargs.get("num_frames") or 124)
    pix = int(call_kwargs.get("height") or 480) * int(call_kwargs.get("width") or 832)
    if pipe.__class__.__name__ == "MiniMaxH3Pipeline" and num_f >= 260 and pix >= 1280 * 700:
        try:
            vvae = getattr(pipe, "video_vae", None)
            if vvae is not None and hasattr(vvae, "offload"):
                vvae.offload()
                _reclaim_transient_gpu_memory()
            elif vvae is not None:
                for mod in vvae.modules():
                    if hasattr(mod, "offload"):
                        mod.offload()
                _reclaim_transient_gpu_memory()
        except Exception:
            pass

    if getattr(pipe, "_high_vram_fast_mode", False) and "tiled" not in call_kwargs:
        try:
            free_gb = torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)
        except Exception:
            free_gb = 0.0
        if free_gb >= 46.0:
            fast_kwargs = dict(call_kwargs)
            fast_kwargs["tiled"] = False
            try:
                return pipe(**fast_kwargs)
            except torch.cuda.OutOfMemoryError:
                print("⚠️ 整帧 VAE 解码显存触顶，正在清理缓存并自动切换为 Tiled 分块安全解码...")
                _reclaim_transient_gpu_memory()
    safe_kwargs = dict(call_kwargs)
    safe_kwargs.setdefault("tiled", True)
    try:
        return pipe(**safe_kwargs)
    except torch.cuda.OutOfMemoryError:
        if pipe.__class__.__name__ == "MiniMaxH3Pipeline" and not safe_kwargs.get("use_gradient_checkpointing"):
            print("⚠️ 超长序列注意力显存触顶，正在自动释放闲置子模块并启用梯度检查点安全重试...")
            for sub_name in ("video_vae", "audio_vae", "text_encoder"):
                sub_m = getattr(pipe, sub_name, None)
                if sub_m is not None:
                    if hasattr(sub_m, "offload"):
                        sub_m.offload()
                    else:
                        for mod in sub_m.modules():
                            if hasattr(mod, "offload"):
                                mod.offload()
            _reclaim_transient_gpu_memory()
            gc_kwargs = dict(safe_kwargs)
            gc_kwargs["use_gradient_checkpointing"] = True
            return pipe(**gc_kwargs)
        raise


# ==============================================================================
# 视频生成核心函数
# ==============================================================================
RESOLUTION_MAP = {
    "832 × 480 (16:9 横屏标准 - 推荐)": (832, 480),
    "480 × 832 (9:16 竖屏短视频)": (480, 832),
    "1280 × 720 (16:9 720P 高清)": (1280, 720),
    "720 × 1280 (9:16 720P 竖屏高清)": (720, 1280),
    "640 × 640 (1:1 方形)": (640, 640),
}


def generate_video(
    task_mode: str,
    visual_prompt: str,
    soundscape_prompt: str,
    music_prompt: str,
    first_frame_img: Optional[Image.Image],
    last_frame_img: Optional[Image.Image],
    resolution_label: str,
    num_frames: int,
    num_inference_steps: int,
    fps: int,
    seed: int,
    randomize_seed: bool,
    save_dir: str,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[Optional[str], str, int, List[str]]:
    global PIPELINE
    touch_activity()

    if not visual_prompt or not visual_prompt.strip():
        return None, "⚠️ 请输入画面描述提示词 (Prompt)！", seed, get_history_videos(save_dir)

    if PIPELINE is None:
        return (
            None,
            "⚠️ 模型尚未加载到 GPU 显存！请先在左侧点击「1️⃣ 下载模型权重」与「2️⃣ 加载模型至显存」。",
            seed,
            get_history_videos(save_dir),
        )

    if randomize_seed or seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

    width, height = RESOLUTION_MAP.get(resolution_label, (832, 480))
    full_prompt = build_structured_prompt(visual_prompt, soundscape_prompt, music_prompt)

    keyframes = []
    keyframe_indices = []
    if task_mode == "fl2va":
        if first_frame_img is not None:
            keyframes.append(first_frame_img.convert("RGB"))
            keyframe_indices.append(0)
        if last_frame_img is not None:
            keyframes.append(last_frame_img.convert("RGB"))
            keyframe_indices.append(-1)
        if not keyframes:
            return (
                None,
                "⚠️ 在「首尾帧图生视频」模式下，请至少上传一张【首帧图片】或【尾帧图片】！",
                seed,
                get_history_videos(save_dir),
            )

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"minimax_h3_{task_mode}_{timestamp}_seed{seed}.mp4"
    output_path = os.path.join(save_dir, filename)

    start_time = time.time()
    with GPU_LOCK:
        try:
            from diffsynth.utils.data.audio_video import write_video_audio

            if progress is not None:
                progress(0.05, desc="正在准备潜空间与多模态编码...")

            call_kwargs = dict(
                prompt=full_prompt,
                height=height,
                width=width,
                num_frames=int(num_frames),
                num_inference_steps=int(num_inference_steps),
                seed=int(seed),
            )
            if keyframes:
                call_kwargs["keyframes"] = keyframes
                call_kwargs["keyframe_indices"] = keyframe_indices

            _reclaim_transient_gpu_memory()
            video, audio = _run_pipeline_with_smart_tiling(PIPELINE, call_kwargs)

            if progress is not None:
                progress(0.92, desc="正在合成 32kHz 立体声音轨与 MP4 视频...")

            write_video_audio(
                video=video,
                audio=audio,
                output_path=output_path,
                fps=int(fps),
                audio_sample_rate=32000,
            )
            video = None
            audio = None
            _reclaim_transient_gpu_memory()

            elapsed = time.time() - start_time
            status_msg = (
                f"✅ 生成完成！耗时: {elapsed:.1f} 秒 | 分辨率: {width}×{height} | "
                f"帧数: {num_frames} ({num_frames/fps:.1f}s) | Seed: {seed}\n"
                f"📁 已保存至: {output_path}"
            )
            return output_path, status_msg, seed, get_history_videos(save_dir)

        except Exception as e:
            _reclaim_transient_gpu_memory()
            return None, f"❌ 生成过程中出错: {str(e)}", seed, get_history_videos(save_dir)
        finally:
            touch_activity()


def get_history_videos(save_dir: str) -> List[str]:
    if not os.path.exists(save_dir):
        return []
    files = glob.glob(os.path.join(save_dir, "*.mp4"))
    files.sort(key=os.path.getmtime, reverse=True)
    return files[:12]


# ==============================================================================
# ComfyUI 云端节点工作台 + Google Drive 自动软链与双向持久化引擎
# ==============================================================================
COMFY_STANDARD_MODEL_SUBDIRS = [
    "checkpoints",
    "diffusion_models",
    "unet",
    "vae",
    "loras",
    "text_encoders",
    "clip",
    "clip_vision",
    "controlnet",
    "upscale_models",
    "latent_upscale_models",
    "audio_encoders",
    "embeddings",
    "style_models",
    "gguf",
]

_COMFY_PROTECTED_PIP_PKGS = re.compile(
    r"^\s*(torch|torchvision|torchaudio|xformers)($|[<>=!~;\s\[])",
    re.IGNORECASE,
)
_COMFY_BG_SYNC_STARTED = False
_COMFY_ADOPT_CHECKED = False
_COMFY_STATE_FILE = "/tmp/.easycolab_comfy_state.json"
_COMFY_STATUS_CACHE: Dict[str, Any] = {"ts": 0.0, "counts": (0, 0, 0), "outputs": [], "packs": []}

COMFY_TEMPLATE_RAW_BASE = "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/"

# 100% 免审核 (Gated=False) 官方原生节点预置图生视频 / 首尾帧运镜工作流包
COMFY_PRESET_WORKFLOW_PACKS: List[Dict[str, Any]] = [
    {
        "id": "wan22_i2v_flf2v",
        "title": "万相 Wan2.2 14B · 图生视频 + 首尾帧双工作流",
        "badge": "新手首选 · 4步极速",
        "scenario": "含「单图动起来」与「首尾双图丝滑过渡」2 套中文画布，G4 约 25 秒出片，纯官方内置节点零红框报错",
        "approx_size_gb": 35.4,
        "workflows": [
            "01_Wan2.2_极速单图生视频(4步傻瓜版).json",
            "02_Wan2.2_首尾帧丝滑过渡(双图4步极速).json",
        ],
        "models": [
            ("text_encoders", "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", 6.27),
            ("vae", "wan_2.1_vae.safetensors", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/vae/wan_2.1_vae.safetensors", 0.24),
            ("diffusion_models", "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", 13.31),
            ("diffusion_models", "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", 13.31),
            ("loras", "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", 1.14),
            ("loras", "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", 1.14),
        ],
    },
    {
        "id": "minimax_h3_i2v_flf2v",
        "title": "MiniMax-H3 Turbo · 图生有声视频 + 首尾帧双工作流",
        "badge": "自带立体声音效 · 8步加速",
        "scenario": "含「单图生有声视频」与「首尾双图有声过渡」2 套极简画布，一次生成高清动态画面 + 同步环境音效/配乐/对白",
        "approx_size_gb": 39.1,
        "workflows": [
            "03_MiniMax-H3_极速图生有声视频(单图带立体声).json",
            "04_MiniMax-H3_首尾帧有声过渡(双图带立体声).json",
        ],
        "models": [
            ("vae", "minimax_h3_video_vae_int8_convrot.safetensors", "Comfy-Org/MiniMax-H3", "vae/minimax_h3_video_vae_int8_convrot.safetensors", 2.62),
            ("vae", "minimax_h3_audio_vae_fp32.safetensors", "Comfy-Org/MiniMax-H3", "vae/minimax_h3_audio_vae_fp32.safetensors", 0.56),
            ("diffusion_models", "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "Comfy-Org/MiniMax-H3", "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors", 19.53),
            ("text_encoders", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "Comfy-Org/MiniMax-H3", "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", 14.61),
            ("loras", "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", "lightx2v/Minimax-h3-Turbo", "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", 1.82),
        ],
    },
    {
        "id": "ltx23_flf2v",
        "title": "LTX-2.3 (22B) · 首尾帧电影级运镜有声工作流",
        "badge": "电影推拉摇移 · 仅 2 个权重",
        "scenario": "专为首尾关键帧大幅度镜头运动优化（如高空俯拍平滑推镜至低角度特写 + 自动合成配乐音轨）",
        "approx_size_gb": 36.3,
        "workflows": [
            "05_LTX-2.3_首尾帧电影运镜(双图极速带音效).json",
        ],
        "models": [
            ("checkpoints", "ltx-2.3-22b-distilled-fp8.safetensors", "Lightricks/LTX-2.3-fp8", "ltx-2.3-22b-distilled-fp8.safetensors", 27.50),
            ("text_encoders", "gemma_3_12B_it_fp4_mixed.safetensors", "Comfy-Org/ltx-2", "split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors", 8.80),
        ],
    },
    {
        "id": "hunyuan15_i2v_1080p",
        "title": "混元 HunyuanVideo 1.5 · 单图直出 1080P 超分工作流",
        "badge": "720P + 1080P 双路超清",
        "scenario": "已默认激活潜空间超分管线，传 1 张图一键自动渲染 720P 原片与 1920×1080 超高清成片",
        "approx_size_gb": 43.5,
        "workflows": [
            "06_Hunyuan1.5_单图直出1080P超清视频.json",
        ],
        "models": [
            ("clip_vision", "sigclip_vision_patch14_384.safetensors", "Comfy-Org/sigclip_vision_384", "sigclip_vision_patch14_384.safetensors", 0.80),
            ("text_encoders", "qwen_2.5_vl_7b_fp8_scaled.safetensors", "Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", 8.74),
            ("text_encoders", "byt5_small_glyphxl_fp16.safetensors", "Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors", 0.41),
            ("latent_upscale_models", "hunyuanvideo15_latent_upsampler_1080p.safetensors", "Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/latent_upscale_models/hunyuanvideo15_latent_upsampler_1080p.safetensors", 0.45),
            ("diffusion_models", "hunyuanvideo1.5_1080p_sr_distilled_fp16.safetensors", "Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/diffusion_models/hunyuanvideo1.5_1080p_sr_distilled_fp16.safetensors", 15.20),
            ("vae", "hunyuanvideo15_vae_fp16.safetensors", "Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/vae/hunyuanvideo15_vae_fp16.safetensors", 2.35),
            ("diffusion_models", "hunyuanvideo1.5_720p_i2v_fp16.safetensors", "Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/diffusion_models/hunyuanvideo1.5_720p_i2v_fp16.safetensors", 15.51),
        ],
    },
]


def _ensure_comfyui_beginner_assets(comfy_root: str, persist_root: str):
    """
    为新手免折腾准备：
    1. 默认将 ComfyUI 界面语言设为简体中文 (Comfy.Locale = zh)；
    2. 生成 1280x720 预设示例首帧/尾帧图片至 input/ 目录并建立软链，即使未传图直接点「运行」也零报错。
    """
    try:
        settings_dir = os.path.join(persist_root, "user", "default")
        os.makedirs(settings_dir, exist_ok=True)
        settings_path = os.path.join(settings_dir, "comfy.settings.json")
        cfg_obj: Dict[str, Any] = {}
        if os.path.isfile(settings_path):
            try:
                cfg_obj = json.load(open(settings_path, "r", encoding="utf-8"))
            except Exception:
                cfg_obj = {}
        if not cfg_obj.get("Comfy.Locale"):
            cfg_obj["Comfy.Locale"] = "zh"
            with open(settings_path, "w", encoding="utf-8") as sf:
                json.dump(cfg_obj, sf, ensure_ascii=False, indent=2)
    except Exception:
        pass

    try:
        from PIL import ImageDraw
        input_persist = os.path.join(persist_root, "input")
        os.makedirs(input_persist, exist_ok=True)
        if os.path.isdir(comfy_root):
            _safe_symlink_dir(input_persist, os.path.join(comfy_root, "input"))

        start_img_p = os.path.join(input_persist, "easycolab_sample_start.png")
        end_img_p = os.path.join(input_persist, "easycolab_sample_end.png")

        if not os.path.isfile(start_img_p):
            w, h = 1280, 720
            im = Image.new("RGB", (w, h), (16, 24, 48))
            dr = ImageDraw.Draw(im)
            for y in range(h):
                t = y / float(h - 1)
                r = int(18 * (1 - t) + 64 * t)
                g = int(28 * (1 - t) + 108 * t)
                b = int(68 * (1 - t) + 168 * t)
                dr.line([(0, y), (w, y)], fill=(r, g, b))
            # 太阳光晕与海平面剪影（首帧：朝阳初升左侧）
            dr.ellipse([260, 210, 440, 390], fill=(255, 206, 126))
            dr.ellipse([285, 235, 415, 365], fill=(255, 240, 196))
            dr.polygon([(0, 520), (240, 430), (520, 495), (860, 410), (1140, 485), (1280, 450), (1280, 720), (0, 720)], fill=(18, 28, 46))
            dr.rectangle([0, 565, w, h], fill=(12, 22, 38))
            im.save(start_img_p, format="PNG")

        if not os.path.isfile(end_img_p):
            w, h = 1280, 720
            im = Image.new("RGB", (w, h), (32, 18, 52))
            dr = ImageDraw.Draw(im)
            for y in range(h):
                t = y / float(h - 1)
                r = int(42 * (1 - t) + 232 * t)
                g = int(24 * (1 - t) + 118 * t)
                b = int(76 * (1 - t) + 88 * t)
                dr.line([(0, y), (w, y)], fill=(r, g, b))
            # 尾帧：晚霞金日移至右侧高空，测试首尾帧运镜与光影平滑过渡
            dr.ellipse([800, 130, 1000, 330], fill=(255, 168, 92))
            dr.ellipse([830, 160, 970, 300], fill=(255, 232, 178))
            dr.polygon([(0, 540), (210, 460), (490, 510), (820, 395), (1090, 470), (1280, 430), (1280, 720), (0, 720)], fill=(24, 16, 36))
            dr.rectangle([0, 575, w, h], fill=(16, 12, 28))
            im.save(end_img_p, format="PNG")
    except Exception as e:
        print(f"[ComfyUI] 生成默认占位参考图提示: {e}")


def _fetch_official_comfy_template(template_filename: str) -> Dict[str, Any]:
    url = COMFY_TEMPLATE_RAW_BASE + template_filename
    req = urllib.request.Request(url, headers={"User-Agent": "EasyColab/2.0"})
    with urllib.request.urlopen(req, timeout=25) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _make_zh_guide_note(node_id: int, pos: List[float], title: str, steps_md: str) -> Dict[str, Any]:
    return {
        "id": node_id,
        "type": "MarkdownNote",
        "pos": pos,
        "size": [430, 340],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [],
        "outputs": [],
        "title": title,
        "properties": {},
        "widgets_values": [steps_md],
        "widgets_values_named": {"text": steps_md},
        "color": "#1e3a5f",
        "bgcolor": "#13263e",
    }


def _ensure_easycolab_comfy_preset_helper(comfy_root: str, persist_root: str):
    """
    预置轻量前端扩展：每次通过全新 *.trycloudflare.com 域名首次打开 ComfyUI 时，
    由于新域名 localStorage 为空，ComfyUI 默认会加载内置的 Z-Image Turbo 文生图模板，
    并把缺失的模型自动替换成目录里字母序第一个文件（例如把 minimax_h3_audio_vae 塞进图像 VAEDecode 导致维度 24!=32 报错）。
    本扩展会在检测到默认 Z-Image Turbo 占位图时，自动切换为用户已安装的 EasyColab 中文预置工作流。
    """
    if not os.path.isdir(comfy_root):
        return
    try:
        helper_dir = os.path.join(comfy_root, "custom_nodes", "easycolab_preset_helper")
        js_dir = os.path.join(helper_dir, "js")
        os.makedirs(js_dir, exist_ok=True)

        init_py = '''import os
import json
from aiohttp import web
import folder_paths
from server import PromptServer

WEB_DIRECTORY = "./js"
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    if getattr(PromptServer, "instance", None) is not None:
        @PromptServer.instance.routes.get("/easycolab/default_workflow")
        async def _easycolab_get_default_workflow(request):
            try:
                get_ud = getattr(folder_paths, "get_user_directory", None)
                user_dir = get_ud() if callable(get_ud) else os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "user",
                )
                default_dir = os.path.join(user_dir, "default")
                wf_dir = os.path.join(default_dir, "workflows")
                target_fn = ""
                ptr_path = os.path.join(default_dir, "easycolab_active_preset.json")
                if os.path.isfile(ptr_path):
                    try:
                        with open(ptr_path, "r", encoding="utf-8") as pf:
                            ptr = json.load(pf)
                        cand = str(ptr.get("workflow_filename") or "").strip()
                        if cand and os.path.isfile(os.path.join(wf_dir, cand)):
                            target_fn = cand
                    except Exception:
                        pass
                if not target_fn and os.path.isdir(wf_dir):
                    cands = sorted(
                        f for f in os.listdir(wf_dir)
                        if f.lower().endswith(".json") and not f.startswith(".")
                    )
                    if cands:
                        target_fn = cands[0]
                if target_fn:
                    wf_path = os.path.join(wf_dir, target_fn)
                    with open(wf_path, "r", encoding="utf-8") as wf:
                        wf_obj = json.load(wf)
                    return web.json_response({"ok": True, "filename": target_fn, "workflow": wf_obj})
            except Exception as exc:
                return web.json_response({"ok": False, "error": str(exc)})
            return web.json_response({"ok": False})
except Exception:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
'''

        autoload_js = '''import { app } from "../../scripts/app.js";

function isBuiltInZImageStarterGraph(graph) {
    if (!graph || !Array.isArray(graph._nodes)) return false;
    if (graph._nodes.length === 0) return false;
    return graph._nodes.some((n) => {
        const models = n?.properties?.models;
        return Array.isArray(models) && models.some((m) => String(m?.url || "").includes("z_image_turbo"));
    });
}

app.registerExtension({
    name: "EasyColab.PresetWorkflowAutoLoader",
    async setup() {
        if (sessionStorage.getItem("easycolab_preset_autoloaded_v1") === "1") {
            return;
        }
        const tryAutoLoad = async (allowEmpty) => {
            if (sessionStorage.getItem("easycolab_preset_autoloaded_v1") === "1") return true;
            const g = app.graph;
            const isStarter = isBuiltInZImageStarterGraph(g);
            const isEmpty = allowEmpty && g && Array.isArray(g._nodes) && g._nodes.length === 0;
            if (!isStarter && !isEmpty) return false;

            sessionStorage.setItem("easycolab_preset_autoloaded_v1", "1");
            try {
                const resp = await fetch("/easycolab/default_workflow", { cache: "no-store" });
                if (!resp.ok) return true;
                const data = await resp.json();
                if (data && data.ok && data.workflow) {
                    const wfTitle = String(data.filename || "EasyColab_Preset").replace(/\\.json$/i, "");
                    await app.loadGraphData(data.workflow, true, true, wfTitle);
                    console.log("[EasyColab] 已自动载入预置中文工作流:", data.filename);
                }
            } catch (err) {
                console.warn("[EasyColab] 自动载入预置工作流跳过:", err);
            }
            return true;
        };

        setTimeout(() => { tryAutoLoad(false); }, 350);
        setTimeout(() => { tryAutoLoad(true); }, 1100);
    },
});
'''
        with open(os.path.join(helper_dir, "__init__.py"), "w", encoding="utf-8") as f_py:
            f_py.write(init_py)
        with open(os.path.join(js_dir, "autoload.js"), "w", encoding="utf-8") as f_js:
            f_js.write(autoload_js)
    except Exception as e:
        print(f"[ComfyUI] 写入 EasyColab 默认工作流引导扩展提示: {e}")


def _prune_workflow_dangling_links(wf_data: Dict[str, Any]):
    """确保每个节点的 inputs/outputs 链接表与顶层 links 完全一致，彻底杜绝 LiteGraph 断链警告"""
    node_ids = {n.get("id") for n in wf_data.get("nodes", [])}
    valid_links = [
        lk for lk in wf_data.get("links", [])
        if isinstance(lk, list) and len(lk) >= 5 and lk[1] in node_ids and lk[3] in node_ids
    ]
    wf_data["links"] = valid_links
    valid_lids = {lk[0] for lk in valid_links}
    for n in wf_data.get("nodes", []):
        for inp in n.get("inputs", []) or []:
            if inp.get("link") is not None and inp.get("link") not in valid_lids:
                inp["link"] = None
        for out in n.get("outputs", []) or []:
            if isinstance(out.get("links"), list):
                out["links"] = [lid for lid in out["links"] if lid in valid_lids]


def _build_and_install_comfy_preset_workflows(
    persist_root: str,
    pack_ids: Optional[List[str]] = None,
) -> List[str]:
    """拉取官方最新模板并转换为全中文标题、内置三步傻瓜注释、G4 高清参数与零缺图报错的开箱即用工作流"""
    wf_dir = os.path.join(persist_root, "user", "default", "workflows")
    os.makedirs(wf_dir, exist_ok=True)
    target_set = set(pack_ids) if pack_ids else {p["id"] for p in COMFY_PRESET_WORKFLOW_PACKS}
    installed_files: List[str] = []

    # 1. 万相 Wan2.2 14B (单图 I2V + 首尾帧 FLF2V 4步极速版)
    if "wan22_i2v_flf2v" in target_set:
        try:
            d_i2v = _fetch_official_comfy_template("video_wan2_2_14B_i2v.json")
            kept_nodes = []
            for n in d_i2v.get("nodes", []):
                ntype = n.get("type")
                if ntype == "MarkdownNote":
                    continue
                if n.get("id") == 97 and ntype == "LoadImage":
                    n["title"] = "① 点击此处选择起始图片 (Choose file to upload)"
                    n["widgets_values"] = ["easycolab_sample_start.png", "image"]
                    n.pop("widgets_values_named", None)
                elif n.get("id") == 129:
                    n["title"] = "② 填写画面动作与镜头描述 (G4 已预设 1280×720 · 4步极速)"
                    wv = list(n.get("widgets_values") or [])
                    if len(wv) >= 4:
                        wv[0] = "镜头缓慢向前推进，晨曦阳光洒在海面与远山上，云层轻柔流动，水面泛起细腻金光涟漪，电影级光影质感。"
                        wv[1] = 1280
                        wv[2] = 720
                    n["widgets_values"] = wv
                    n.pop("widgets_values_named", None)
                elif ntype == "SaveVideo":
                    n["title"] = "③ 生成结果预览与自动保存 (右键可下载视频)"
                    n["widgets_values"] = ["video/Wan2.2_单图生视频", "auto", "auto"]
                    n.pop("widgets_values_named", None)
                kept_nodes.append(n)
            kept_nodes.append(_make_zh_guide_note(
                501,
                [-520, -420],
                "【EasyColab 三步傻瓜出片指南 · 单图生视频】",
                "## 三步直接出片（无需连线！）\n\n"
                "1. **传图**：点击右侧 **「① 点击此处选择起始图片」** 节点底部的上传按钮，选一张你的图片。\n"
                "2. **写动作**：在中间 **「② 填写画面动作与镜头描述」** 第一个文本框里用中文或英文写希望画面怎么动（如：镜头推进、人物转头微笑、发丝随风飘动）。\n"
                "3. **点运行**：点击网页右上角蓝色的 **「运行 (Run)」** 按钮，G4 显卡约 **25 秒** 即可在右侧直接预览成片！\n\n"
                "> 提示：所有生成的视频都会自动同步至你的 Google Drive，也可回到 EasyColab 客户端一键拉取到 Mac 本地。",
            ))
            d_i2v["nodes"] = kept_nodes
            d_i2v["last_node_id"] = max(int(d_i2v.get("last_node_id") or 200), 505)
            d_i2v.setdefault("extra", {})["ds"] = {"scale": 0.72, "offset": [580, 560]}
            _prune_workflow_dangling_links(d_i2v)
            out_1 = os.path.join(wf_dir, "01_Wan2.2_极速单图生视频(4步傻瓜版).json")
            with open(out_1, "w", encoding="utf-8") as wf:
                json.dump(d_i2v, wf, ensure_ascii=False, indent=2)
            installed_files.append(os.path.basename(out_1))
        except Exception as e:
            print(f"[ComfyUI 工作流生成] Wan2.2 I2V 模板处理异常: {e}")

        try:
            d_flf = _fetch_official_comfy_template("video_wan2_2_14B_flf2v.json")
            # 仅提取官方模板中 4-Step LightX2V LoRA 极速分支连通分量 (原 mode=4)，彻底剔除慢速重复节点并自动激活 (mode=0)
            active_ids = {n["id"] for n in d_flf.get("nodes", []) if n.get("mode") == 4}
            flf_nodes = []
            for n in d_flf.get("nodes", []):
                nid = n.get("id")
                if nid not in active_ids or n.get("type") == "Note":
                    continue
                n["mode"] = 0
                if isinstance(n.get("pos"), list) and len(n["pos"]) >= 2:
                    n["pos"] = [n["pos"][0], n["pos"][1] - 1600]
                if nid == 68 and n.get("type") == "LoadImage":
                    # link 157 -> node 67 slot 5 (start_image)
                    n["title"] = "① 点击上传起始首帧图片 (Start Frame)"
                    n["widgets_values"] = ["easycolab_sample_start.png", "image"]
                    n.pop("widgets_values_named", None)
                elif nid == 62 and n.get("type") == "LoadImage":
                    # link 158 -> node 67 slot 6 (end_image)
                    n["title"] = "② 点击上传结束尾帧图片 (End Frame)"
                    n["widgets_values"] = ["easycolab_sample_end.png", "image"]
                    n.pop("widgets_values_named", None)
                elif nid == 6 and n.get("type") == "CLIPTextEncode":
                    n["title"] = "③ 输入首尾帧过渡动作与运镜描述 (正向提示词)"
                    n["widgets_values"] = [
                        "清晨海面的朝阳缓缓升起并化作绚丽的傍晚漫天晚霞，镜头平稳向右推进，云层与海浪自然流动过渡，光影由冷蓝渐变为温暖金红，画面连贯丝滑。"
                    ]
                    n.pop("widgets_values_named", None)
                elif nid == 7 and n.get("type") == "CLIPTextEncode":
                    n["title"] = "反向提示词 (保持默认即可)"
                elif nid == 67 and n.get("type") == "WanFirstLastFrameToVideo":
                    n["title"] = "④ 视频尺寸与总帧数 (81帧≈5秒 · 1280×720)"
                    n["widgets_values"] = [1280, 720, 81, 1]
                    n.pop("widgets_values_named", None)
                elif nid == 61 and n.get("type") == "SaveVideo":
                    n["title"] = "⑤ 首尾帧过渡成片预览与自动保存"
                    n["widgets_values"] = ["video/Wan2.2_首尾帧过渡", "auto", "auto"]
                    n.pop("widgets_values_named", None)
                flf_nodes.append(n)

            flf_nodes.append(_make_zh_guide_note(
                502,
                [-430, 490],
                "【EasyColab 三步傻瓜出片指南 · 首尾帧过渡】",
                "## 首尾两张图丝滑变形/运镜（4步极速已开启）\n\n"
                "1. **传首尾两张图**：在右侧 **「① 起始首帧图片」** 和 **「② 结束尾帧图片」** 分别上传开头和结尾画面。\n"
                "2. **写过渡过程**：在 **「③ 输入首尾帧过渡动作与运镜描述」** 写清楚从图 1 怎么变化/运镜到图 2。\n"
                "3. **点右上角运行**：无需按任何快捷键，4 步加速 LoRA 已自动配好，约 **25~35 秒** 直接生成丝滑过渡视频！",
            ))
            d_flf["nodes"] = flf_nodes
            d_flf["links"] = [
                lk for lk in d_flf.get("links", [])
                if lk[1] in active_ids and lk[3] in active_ids
            ]
            zh_group_titles = {
                1: "自动模型加载区 (无需改动)",
                2: "第一步：上传起始首帧图片",
                5: "第二步：上传结束尾帧图片",
                3: "第三步：输入画面过渡描述",
                4: "第四步：视频分辨率与帧数 (默认 1280×720)",
            }
            kept_groups = []
            for g in d_flf.get("groups", []):
                gid = g.get("id")
                if gid in zh_group_titles:
                    b = list(g.get("bounding", [0, 0, 400, 400]))
                    if len(b) >= 2:
                        b[1] = b[1] - 1600
                    g["bounding"] = b
                    g["title"] = zh_group_titles[gid]
                    kept_groups.append(g)
            d_flf["groups"] = kept_groups
            d_flf["last_node_id"] = max(int(d_flf.get("last_node_id") or 200), 505)
            d_flf.setdefault("extra", {})["ds"] = {"scale": 0.62, "offset": [480, 120]}
            _prune_workflow_dangling_links(d_flf)
            out_2 = os.path.join(wf_dir, "02_Wan2.2_首尾帧丝滑过渡(双图4步极速).json")
            with open(out_2, "w", encoding="utf-8") as wf:
                json.dump(d_flf, wf, ensure_ascii=False, indent=2)
            installed_files.append(os.path.basename(out_2))
        except Exception as e:
            print(f"[ComfyUI 工作流生成] Wan2.2 FLF2V 模板处理异常: {e}")

    # 2. MiniMax-H3 Turbo (单图有声视频 + 首尾帧有声视频 8步极速双工作流)
    if "minimax_h3_i2v_flf2v" in target_set:
        try:
            import copy
            d_h3_raw = _fetch_official_comfy_template("video_minimax_h3_i2v.json")

            def _customize_h3(is_dual_frame: bool) -> Dict[str, Any]:
                d_h3 = copy.deepcopy(d_h3_raw)
                nodes_out = []
                for n in d_h3.get("nodes", []):
                    nid = n.get("id")
                    ntype = n.get("type")
                    if ntype == "MarkdownNote" or nid in (119, 120):
                        continue
                    if nid == 114 and ntype == "LoadImage":
                        n["title"] = "① 点击上传起始首帧图片 (First Frame)"
                        n["widgets_values"] = ["easycolab_sample_start.png", "image"]
                        n["widgets_values_named"] = {"image": "easycolab_sample_start.png", "upload": "image"}
                        n["pos"] = [-1320, 4820]
                    elif nid == 115 and ntype == "ResolutionSelector":
                        n["title"] = "画幅比例与清晰度 (0.6≈1056×608 · 0.98=1344×768)"
                        n["widgets_values"] = ["16:9 (Widescreen)", 0.6, 32]
                        n["widgets_values_named"] = {
                            "aspect_ratio": "16:9 (Widescreen)",
                            "megapixels": 0.6,
                            "multiple": 32,
                        }
                        n["pos"] = [-1320, 5660] if is_dual_frame else [-1320, 5260]
                    elif nid == 105:
                        n["title"] = (
                            "③ 填写过渡动作 + 音效配乐描述 (8步 Turbo 极速已开启)"
                            if is_dual_frame
                            else "② 填写画面动作 + 音效配乐描述 (8步 Turbo 极速已开启)"
                        )
                        zh_prompt = (
                            "电影级航拍镜头：画面从清晨宁静的蔚蓝海岸平稳推进并过渡至漫天金红晚霞的黄昏海湾，海浪轻拍礁石，光影自然流转。\n"
                            "Audio: 轻柔的海浪拍岸声、温暖的海风吹拂声，伴随舒缓治愈的钢琴与大提琴电影配乐。"
                            if is_dual_frame
                            else "电影级运镜缓慢向前推进，清晨阳光穿透薄雾洒在海面，金光波光粼粼，飞鸟掠过天际。\n"
                                 "Audio: 清脆的海浪声、微风声与远处海鸥鸣叫，搭配温柔空灵的钢琴环境配乐。"
                        )
                        wv = list(n.get("widgets_values") or [])
                        if len(wv) >= 10:
                            wv[0] = zh_prompt
                            wv[9] = True  # 默认启用 8-step Turbo LoRA 极速出片
                        n["widgets_values"] = wv
                        if isinstance(n.get("widgets_values_named"), dict):
                            n["widgets_values_named"]["prompt"] = zh_prompt
                            n["widgets_values_named"]["value"] = True
                    elif nid == 92 and ntype == "SaveVideo":
                        n["title"] = "④ 有声视频预览与自动保存 (请打开播放器音量!)"
                        vid_prefix = "video/MiniMax_H3_首尾帧有声" if is_dual_frame else "video/MiniMax_H3_图生有声"
                        n["widgets_values"] = [vid_prefix, "auto", "auto"]
                        n["widgets_values_named"] = {
                            "filename_prefix": vid_prefix,
                            "format": "auto",
                            "codec": "auto",
                        }
                    nodes_out.append(n)

                links_out = [lk for lk in d_h3.get("links", []) if lk[0] != 228]
                if is_dual_frame:
                    end_node_id = 201
                    new_link_id = 250
                    nodes_out.append({
                        "id": end_node_id,
                        "type": "LoadImage",
                        "pos": [-1320, 5240],
                        "size": [315, 314],
                        "flags": {},
                        "order": 1,
                        "mode": 0,
                        "inputs": [],
                        "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [new_link_id]}, {"name": "MASK", "type": "MASK", "links": None}],
                        "title": "② 点击上传结束尾帧图片 (Last Frame)",
                        "properties": {"Node name for S&R": "LoadImage"},
                        "widgets_values": ["easycolab_sample_end.png", "image"],
                        "widgets_values_named": {"image": "easycolab_sample_end.png", "upload": "image"},
                    })
                    for n in nodes_out:
                        if n.get("id") == 105:
                            for inp in n.get("inputs", []):
                                if inp.get("name") == "last_frame":
                                    inp["link"] = new_link_id
                    links_out.append([new_link_id, end_node_id, 0, 105, 1, "IMAGE"])
                    d_h3["last_link_id"] = max(int(d_h3.get("last_link_id") or 230), new_link_id + 1)

                nodes_out.append(_make_zh_guide_note(
                    503,
                    [-1800, 4820],
                    "【EasyColab 三步出片指南 · MiniMax-H3 原生有声视频】",
                    "## 画面 + 立体声音效/配乐一次生成！\n\n"
                    + (
                        "1. **传首尾两张图**：在 **「① 起始首帧」** 与 **「② 结束尾帧」** 上传起止图片。\n"
                        "2. **写画面 + 声音**：在中间 **「③ 过渡动作 + 音效配乐描述」** 写清画面变化，并在 `Audio:` 后写想要的环境音/台词/音乐。\n"
                        "3. **点运行**：已默认开启 **8 步 Turbo 加速**，生成后记得打开右侧播放器喇叭图标听立体声！"
                        if is_dual_frame
                        else "1. **传首帧图**：在 **「① 起始首帧」** 点击上传你的参考图。\n"
                             "2. **写画面 + 声音**：在中间 **「② 画面动作 + 音效配乐描述」** 描述动作，并在 `Audio:` 后写音效与配乐风格。\n"
                             "3. **点运行**：已默认开启 **8 步 Turbo 加速**，生成后打开右侧视频喇叭即可试听同步立体声！"
                    ),
                ))
                d_h3["nodes"] = nodes_out
                d_h3["links"] = links_out
                d_h3["last_node_id"] = max(int(d_h3.get("last_node_id") or 210), 505)
                d_h3.setdefault("extra", {})["ds"] = {"scale": 0.68, "offset": [1860, -4740]}
                _prune_workflow_dangling_links(d_h3)
                return d_h3

            out_3 = os.path.join(wf_dir, "03_MiniMax-H3_极速图生有声视频(单图带立体声).json")
            with open(out_3, "w", encoding="utf-8") as wf:
                json.dump(_customize_h3(is_dual_frame=False), wf, ensure_ascii=False, indent=2)
            installed_files.append(os.path.basename(out_3))

            out_4 = os.path.join(wf_dir, "04_MiniMax-H3_首尾帧有声过渡(双图带立体声).json")
            with open(out_4, "w", encoding="utf-8") as wf:
                json.dump(_customize_h3(is_dual_frame=True), wf, ensure_ascii=False, indent=2)
            installed_files.append(os.path.basename(out_4))
        except Exception as e:
            print(f"[ComfyUI 工作流生成] MiniMax-H3 模板处理异常: {e}")

    # 3. LTX-2.3 22B Distilled (首尾帧电影运镜 + 音轨)
    if "ltx23_flf2v" in target_set:
        try:
            d_ltx = _fetch_official_comfy_template("video_ltx2_3_flf2v.json")
            ltx_nodes = []
            for n in d_ltx.get("nodes", []):
                nid = n.get("id")
                ntype = n.get("type")
                if ntype == "MarkdownNote":
                    continue
                if nid == 31 and ntype == "LoadImage":
                    n["title"] = "① 点击上传起始首帧图片 (First Frame)"
                    n["widgets_values"] = ["easycolab_sample_start.png", "image"]
                    n.pop("widgets_values_named", None)
                elif nid == 39 and ntype == "LoadImage":
                    n["title"] = "② 点击上传结束尾帧图片 (Last Frame)"
                    n["widgets_values"] = ["easycolab_sample_end.png", "image"]
                    n.pop("widgets_values_named", None)
                elif nid == 129:
                    n["title"] = "③ 首尾帧电影运镜与音轨参数 (LTX-2.3 22B Distilled)"
                elif nid == 68 and ntype == "SaveVideo":
                    n["title"] = "④ 电影运镜成片预览与自动保存"
                    n["widgets_values"] = ["video/LTX2.3_首尾帧运镜", "auto", "auto"]
                    n.pop("widgets_values_named", None)
                ltx_nodes.append(n)

            ltx_nodes.append(_make_zh_guide_note(
                504,
                [-490, 2250],
                "【EasyColab 三步傻瓜出片指南 · LTX-2.3 首尾帧运镜】",
                "## 电影级大幅度推拉摇移运镜（自带音效配乐）\n\n"
                "1. **传首尾帧**：在右侧 **「① 起始首帧」** 与 **「② 结束尾帧」** 上传两张不同机位或状态的画面。\n"
                "2. **写运镜轨迹**：在中间 **「③ 首尾帧电影运镜与音轨参数」** 文本框描述镜头怎么移动（建议英文描述运镜 + `Music:` 描述音乐）。\n"
                "3. **点运行**：采用 Distilled 极速蒸馏权重，一键直出高帧率带音轨视频！",
            ))
            d_ltx["nodes"] = ltx_nodes
            d_ltx["last_node_id"] = max(int(d_ltx.get("last_node_id") or 200), 506)
            d_ltx.setdefault("extra", {})["ds"] = {"scale": 0.66, "offset": [560, -2160]}
            _prune_workflow_dangling_links(d_ltx)
            out_5 = os.path.join(wf_dir, "05_LTX-2.3_首尾帧电影运镜(双图极速带音效).json")
            with open(out_5, "w", encoding="utf-8") as wf:
                json.dump(d_ltx, wf, ensure_ascii=False, indent=2)
            installed_files.append(os.path.basename(out_5))
        except Exception as e:
            print(f"[ComfyUI 工作流生成] LTX-2.3 FLF2V 模板处理异常: {e}")

    # 4. 混元 HunyuanVideo 1.5 (单图 720P -> 1080P 自动潜空间超分)
    if "hunyuan15_i2v_1080p" in target_set:
        try:
            d_hy = _fetch_official_comfy_template("video_hunyuan_video_1.5_720p_i2v.json")
            sr_enable_ids = {108, 111, 109, 113, 114, 117, 118, 131, 132, 133, 134, 135, 137, 138, 139, 140, 141}
            drop_ids = {104, 120, 112, 121, 122, 123, 124, 136}
            hy_nodes = []
            for n in d_hy.get("nodes", []):
                nid = n.get("id")
                if nid in drop_ids:
                    continue
                if nid in sr_enable_ids:
                    n["mode"] = 0
                if nid == 80 and n.get("type") == "LoadImage":
                    n["title"] = "① 点击上传起始首帧图片 (720P → 1080P 超分)"
                    n["widgets_values"] = ["easycolab_sample_start.png", "image"]
                    n.pop("widgets_values_named", None)
                elif nid == 44 and n.get("type") == "CLIPTextEncode":
                    n["title"] = "② 输入画面动作与光影描述 (中英文皆可)"
                    n["widgets_values"] = [
                        "清晨阳光洒在壮阔的海岸山脉上，镜头平稳向前推进，海面波光粼粼，云雾在山谷间缓慢流动，极致写实电影摄影质感。"
                    ]
                    n.pop("widgets_values_named", None)
                elif nid in (127, 131) and n.get("type") == "RandomNoise":
                    wv = list(n.get("widgets_values") or [42, "randomize"])
                    if len(wv) >= 2:
                        wv[1] = "randomize"
                    n["widgets_values"] = wv
                    n.pop("widgets_values_named", None)
                elif nid == 102 and n.get("type") == "SaveVideo":
                    n["title"] = "③ 720P 基础成片预览"
                elif nid == 118 and n.get("type") == "SaveVideo":
                    n["title"] = "④ 1920×1080 超高清成片预览与保存"
                hy_nodes.append(n)

            hy_nodes.append(_make_zh_guide_note(
                506,
                [-1120, 480],
                "【EasyColab 三步傻瓜出片指南 · 混元 1.5 直出 1080P】",
                "## 单图直出 1920×1080 超高清视频（超分管线已激活）\n\n"
                "1. **传图片**：在右侧 **「① 点击上传起始首帧图片」** 选择你的原图。\n"
                "2. **改提示词**：在上方 **「② 输入画面动作与光影描述」** 输入动态描述。\n"
                "3. **点运行**：无需按 `Ctrl+B`，工作流会先生成 720P 动态原片，再自动接力潜空间超分输出 **1920×1080** 超清成片！",
            ))
            d_hy["nodes"] = hy_nodes
            d_hy["links"] = [
                lk for lk in d_hy.get("links", [])
                if lk[1] not in drop_ids and lk[3] not in drop_ids
            ]
            for g in d_hy.get("groups", []):
                if g.get("id") == 3:
                    g["title"] = "第一步：上传起始图片"
                elif g.get("id") == 2:
                    g["title"] = "第二步：输入画面动作描述"
                elif g.get("id") == 8:
                    g["title"] = "1080P 超高清潜空间放大管线 (已默认激活)"
            d_hy["last_node_id"] = max(int(d_hy.get("last_node_id") or 200), 508)
            d_hy.setdefault("extra", {})["ds"] = {"scale": 0.56, "offset": [1180, 120]}
            _prune_workflow_dangling_links(d_hy)
            out_6 = os.path.join(wf_dir, "06_Hunyuan1.5_单图直出1080P超清视频.json")
            with open(out_6, "w", encoding="utf-8") as wf:
                json.dump(d_hy, wf, ensure_ascii=False, indent=2)
            installed_files.append(os.path.basename(out_6))
        except Exception as e:
            print(f"[ComfyUI 工作流生成] Hunyuan 1.5 模板处理异常: {e}")

    if installed_files:
        try:
            with open(os.path.join(wf_dir, ".easycolab_wf_schema_v2"), "w", encoding="utf-8") as sf:
                sf.write("v2")
            ptr_path = os.path.join(persist_root, "user", "default", "easycolab_active_preset.json")
            if (pack_ids and len(pack_ids) == 1) or not os.path.isfile(ptr_path):
                with open(ptr_path, "w", encoding="utf-8") as pf:
                    json.dump({"workflow_filename": installed_files[0]}, pf, ensure_ascii=False, indent=2)
        except Exception:
            pass

    _COMFY_STATUS_CACHE["ts"] = 0.0
    return installed_files


def _get_comfy_preset_packs_status(persist_root: str) -> List[Dict[str, Any]]:
    wf_dir = os.path.join(persist_root, "user", "default", "workflows")
    results: List[Dict[str, Any]] = []
    for pack in COMFY_PRESET_WORKFLOW_PACKS:
        models = pack["models"]
        installed_m = 0
        for sub, fname, _, _, _ in models:
            p = os.path.join(persist_root, "models", sub, fname)
            if os.path.isfile(p) and os.path.getsize(p) > 1024 * 1024:
                installed_m += 1
        wfs = pack["workflows"]
        installed_wf = sum(1 for w in wfs if os.path.isfile(os.path.join(wf_dir, w)))
        ready = (installed_m == len(models)) and (installed_wf == len(wfs))
        results.append({
            "id": pack["id"],
            "title": pack["title"],
            "badge": pack["badge"],
            "scenario": pack["scenario"],
            "approx_size_gb": pack["approx_size_gb"],
            "workflows": wfs,
            "installed_models": installed_m,
            "total_models": len(models),
            "workflows_installed": installed_wf == len(wfs),
            "ready": ready,
        })
    return results


def _resolve_comfyui_persistence_paths() -> Tuple[str, str, str, str, bool]:
    comfy_root = "/content/ComfyUI" if os.path.exists("/content") else os.path.join(os.getcwd(), "ComfyUI")
    if os.path.exists("/content/drive/MyDrive"):
        persist_root = "/content/drive/MyDrive/EasyColab_ComfyUI"
        return comfy_root, persist_root, "drive_symlink", "Google Drive 实时软链直连", True
    if _has_rclone_drive_remote():
        persist_root = "/content/drive_mirror/EasyColab_ComfyUI"
        return comfy_root, persist_root, "rclone_drive_sync", "Google Drive 自动软链持久化", True
    persist_root = (
        "/content/EasyColab_ComfyUI_Data"
        if os.path.exists("/content")
        else os.path.join(os.getcwd(), "EasyColab_ComfyUI_Data")
    )
    return comfy_root, persist_root, "local_only", "实例本地高速存储", False


def _safe_symlink_dir(target_dir: str, link_path: str):
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(os.path.dirname(link_path), exist_ok=True)
    if os.path.islink(link_path):
        cur_target = os.readlink(link_path)
        if os.path.abspath(cur_target) == os.path.abspath(target_dir):
            return
        os.unlink(link_path)
    elif os.path.isdir(link_path):
        for entry in os.listdir(link_path):
            src_item = os.path.join(link_path, entry)
            dst_item = os.path.join(target_dir, entry)
            if not os.path.exists(dst_item):
                try:
                    shutil.move(src_item, dst_item)
                except Exception:
                    pass
        shutil.rmtree(link_path, ignore_errors=True)
    elif os.path.exists(link_path):
        os.remove(link_path)
    os.symlink(target_dir, link_path)


def _scan_comfyui_counts_and_outputs() -> Tuple[Tuple[int, int, int], List[Dict[str, Any]]]:
    now = time.time()
    if (now - _COMFY_STATUS_CACHE["ts"]) < 6.0:
        return _COMFY_STATUS_CACHE["counts"], list(_COMFY_STATUS_CACHE["outputs"])

    comfy_root, persist_root, _, _, _ = _resolve_comfyui_persistence_paths()
    cn_count = 0
    cn_base = os.path.join(comfy_root, "custom_nodes")
    if not os.path.isdir(cn_base):
        cn_base = os.path.join(persist_root, "custom_nodes")
    if os.path.isdir(cn_base):
        try:
            for d in os.listdir(cn_base):
                if d.startswith(".") or d == "__pycache__":
                    continue
                if os.path.isdir(os.path.join(cn_base, d)):
                    cn_count += 1
        except Exception:
            pass

    model_exts = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx"}
    m_count = 0
    m_base = os.path.join(persist_root, "models")
    if os.path.isdir(m_base):
        try:
            for root_d, _, fnames in os.walk(m_base):
                for fn in fnames:
                    if os.path.splitext(fn)[1].lower() in model_exts:
                        m_count += 1
        except Exception:
            pass

    wf_count = 0
    wf_dir = os.path.join(persist_root, "user", "default", "workflows")
    if os.path.isdir(wf_dir):
        try:
            wf_count = sum(1 for fn in os.listdir(wf_dir) if fn.lower().endswith(".json"))
        except Exception:
            pass

    recent_outputs: List[Dict[str, Any]] = []
    out_dir = os.path.join(comfy_root, "output")
    if not os.path.isdir(out_dir):
        out_dir = os.path.join(persist_root, "output")
    valid_media = {".mp4", ".webm", ".mov", ".webp", ".gif", ".png", ".jpg", ".jpeg"}
    if os.path.isdir(out_dir):
        collected = []
        try:
            for root_d, _, fnames in os.walk(out_dir):
                for fn in fnames:
                    if fn.startswith("."):
                        continue
                    ext = os.path.splitext(fn)[1].lower()
                    if ext not in valid_media:
                        continue
                    full_p = os.path.join(root_d, fn)
                    try:
                        st = os.stat(full_p)
                        if st.st_size < 512:
                            continue
                        rel_p = os.path.relpath(full_p, out_dir).replace("\\", "/")
                        mtype = "video" if ext in {".mp4", ".webm", ".mov"} else "image"
                        collected.append((st.st_mtime, {
                            "rel_path": rel_p,
                            "filename": fn,
                            "media_type": mtype,
                            "size_mb": round(st.st_size / (1024 ** 2), 2),
                            "mtime": int(st.st_mtime),
                        }))
                    except Exception:
                        pass
            collected.sort(key=lambda x: x[0], reverse=True)
            recent_outputs = [item for _, item in collected[:16]]
        except Exception:
            pass

    _COMFY_STATUS_CACHE["ts"] = now
    _COMFY_STATUS_CACHE["counts"] = (cn_count, m_count, wf_count)
    _COMFY_STATUS_CACHE["outputs"] = recent_outputs
    return (cn_count, m_count, wf_count), recent_outputs


def _try_adopt_running_comfyui():
    """若 app.py 热更新重启前实例上已有在线 ComfyUI 与 cloudflared 隧道，自动无缝接管，不断开用户已打开的画布"""
    global _COMFY_ADOPT_CHECKED, _COMFY_BG_SYNC_STARTED
    if _COMFY_ADOPT_CHECKED or COMFYUI_STATE.get("is_running") or COMFYUI_STATE.get("is_starting"):
        return
    _COMFY_ADOPT_CHECKED = True

    port = int(COMFYUI_STATE.get("local_port") or 8188)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{port}/system_stats", timeout=1.0) as resp:
            if resp.status != 200:
                return
    except Exception:
        return

    pub_url = ""
    if os.path.isfile(_COMFY_STATE_FILE):
        try:
            st_obj = json.load(open(_COMFY_STATE_FILE, "r", encoding="utf-8"))
            pub_url = str(st_obj.get("public_url") or "").strip()
        except Exception:
            pub_url = ""

    if not pub_url:
        cf_re = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
        for m_port in range(20241, 20247):
            for ep in ("/quicktunnel", "/metrics"):
                try:
                    with opener.open(f"http://127.0.0.1:{m_port}{ep}", timeout=0.6) as mr:
                        body = mr.read().decode("utf-8", errors="ignore")
                    m = cf_re.search(body)
                    if m:
                        pub_url = m.group(0)
                        break
                    m_host = re.search(r'"hostname"\s*:\s*"([a-zA-Z0-9-]+\.trycloudflare\.com)"', body)
                    if m_host:
                        pub_url = "https://" + m_host.group(1)
                        break
                except Exception:
                    pass
            if pub_url:
                break

    if not pub_url:
        try:
            cf_bin = _ensure_cloudflared_binary()
            tunnel_proc = subprocess.Popen(
                [cf_bin, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            COMFYUI_STATE["tunnel_proc"] = tunnel_proc
            cf_re = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
            if tunnel_proc.stdout is not None:
                t_end = time.time() + 12.0
                while time.time() < t_end:
                    ln = tunnel_proc.stdout.readline()
                    if not ln:
                        break
                    m = cf_re.search(ln)
                    if m:
                        pub_url = m.group(0)
                        break
        except Exception:
            pass

    comfy_root, persist_root, _, label, _ = _resolve_comfyui_persistence_paths()
    _ensure_comfyui_beginner_assets(comfy_root, persist_root)
    COMFYUI_STATE["is_running"] = True
    COMFYUI_STATE["public_url"] = pub_url
    COMFYUI_STATE["progress"] = 1.0
    COMFYUI_STATE["status_message"] = f"已无缝接管在线 ComfyUI 画布（{label}）"
    if pub_url:
        try:
            with open(_COMFY_STATE_FILE, "w", encoding="utf-8") as sf:
                json.dump({"public_url": pub_url, "port": port}, sf)
        except Exception:
            pass
    if not _COMFY_BG_SYNC_STARTED:
        _COMFY_BG_SYNC_STARTED = True
        threading.Thread(target=_comfyui_periodic_drive_sync_loop, daemon=True).start()
    print(f"[ComfyUI] 已自动接管实例上正在运行的 ComfyUI 服务: {pub_url or f'127.0.0.1:{port}'}")


def _poll_local_comfyui_runtime_quiet():
    proc = COMFYUI_STATE.get("comfy_proc")
    if proc is not None and proc.poll() is not None:
        COMFYUI_STATE["is_running"] = False
        COMFYUI_STATE["queue_running"] = 0
        COMFYUI_STATE["queue_pending"] = 0
        if not COMFYUI_STATE["is_starting"]:
            COMFYUI_STATE["status_message"] = f"ComfyUI 进程已退出 (code={proc.returncode})，可点击重新启动"
        COMFYUI_STATE["comfy_proc"] = None
        return

    if not COMFYUI_STATE.get("is_running"):
        _try_adopt_running_comfyui()
        if not COMFYUI_STATE.get("is_running"):
            return

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    port = int(COMFYUI_STATE.get("local_port") or 8188)
    try:
        with opener.open(f"http://127.0.0.1:{port}/queue", timeout=1.2) as r:
            q_data = json.loads(r.read().decode("utf-8", errors="replace"))
        q_run = len(q_data.get("queue_running") or [])
        q_pend = len(q_data.get("queue_pending") or [])
        COMFYUI_STATE["queue_running"] = q_run
        COMFYUI_STATE["queue_pending"] = q_pend
        if q_run > 0 or q_pend > 0:
            touch_activity()
    except Exception:
        pass

    try:
        with opener.open(f"http://127.0.0.1:{port}/history?max_items=5", timeout=1.2) as r:
            h_data = json.loads(r.read().decode("utf-8", errors="replace"))
        if isinstance(h_data, dict):
            h_sig = hash(tuple(sorted(h_data.keys())))
            if h_sig != COMFYUI_STATE.get("last_history_count", 0) and COMFYUI_STATE.get("last_history_count", 0) != 0:
                touch_activity()
                _COMFY_STATUS_CACHE["ts"] = 0.0
            COMFYUI_STATE["last_history_count"] = h_sig
    except Exception:
        pass


def get_comfyui_status_dict() -> Dict[str, Any]:
    _, persist_root, mode, label, is_drive = _resolve_comfyui_persistence_paths()
    (cn_count, m_count, wf_count), recent_outputs = _scan_comfyui_counts_and_outputs()
    preset_packs = _get_comfy_preset_packs_status(persist_root)
    COMFYUI_STATE["persistence_mode"] = mode
    COMFYUI_STATE["persistence_label"] = label
    COMFYUI_STATE["drive_persisted"] = is_drive
    COMFYUI_STATE["custom_nodes_count"] = cn_count
    COMFYUI_STATE["models_count"] = m_count
    COMFYUI_STATE["workflows_count"] = wf_count
    return {
        "is_running": bool(COMFYUI_STATE["is_running"]),
        "is_starting": bool(COMFYUI_STATE["is_starting"]),
        "is_syncing_drive": bool(COMFYUI_STATE["is_syncing_drive"]),
        "is_downloading_weight": bool(COMFYUI_STATE["is_downloading_weight"]),
        "installing_pack_id": str(COMFYUI_STATE.get("installing_pack_id") or ""),
        "progress": float(COMFYUI_STATE["progress"]),
        "status_message": str(COMFYUI_STATE["status_message"]),
        "public_url": str(COMFYUI_STATE["public_url"] or ""),
        "local_port": int(COMFYUI_STATE["local_port"]),
        "queue_running": int(COMFYUI_STATE["queue_running"]),
        "queue_pending": int(COMFYUI_STATE["queue_pending"]),
        "persistence_mode": mode,
        "persistence_label": label,
        "drive_persisted": is_drive,
        "persist_root": persist_root,
        "custom_nodes_count": cn_count,
        "models_count": m_count,
        "workflows_count": wf_count,
        "last_sync_ts": int(COMFYUI_STATE["last_sync_ts"]),
        "recent_outputs": recent_outputs,
        "preset_packs": preset_packs,
    }


def free_comfyui_vram_core() -> Tuple[bool, str]:
    if not COMFYUI_STATE.get("is_running"):
        return False, "ComfyUI 当前未运行"
    port = int(COMFYUI_STATE.get("local_port") or 8188)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/free",
            data=json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with opener.open(req, timeout=8.0) as _:
            pass
        _reclaim_transient_gpu_memory()
        msg = "已释放 ComfyUI 驻留模型与显存缓存（画布保持在线）"
        COMFYUI_STATE["status_message"] = msg
        print(f"[ComfyUI] {msg}")
        return True, msg
    except Exception as e:
        return False, f"释放 ComfyUI 显存失败: {e}"


def sync_comfyui_to_drive_core(quiet: bool = False) -> Tuple[bool, str]:
    if COMFYUI_STATE["is_syncing_drive"]:
        return True, "正在后台同步 ComfyUI 数据到 Google Drive..."
    comfy_root, persist_root, mode, label, is_drive = _resolve_comfyui_persistence_paths()
    if not os.path.isdir(comfy_root) and not os.path.isdir(persist_root):
        return False, "ComfyUI 尚未初始化"

    COMFYUI_STATE["is_syncing_drive"] = True
    try:
        os.makedirs(os.path.join(persist_root, "models"), exist_ok=True)
        os.makedirs(os.path.join(persist_root, "custom_nodes"), exist_ok=True)
        os.makedirs(os.path.join(persist_root, "user"), exist_ok=True)
        os.makedirs(os.path.join(persist_root, "output"), exist_ok=True)
        os.makedirs(os.path.join(persist_root, "input"), exist_ok=True)

        # 1. 自动接管自定义节点在 models/ 下新建的非软链目录，迁移至持久化目录并建立软链
        models_dir = os.path.join(comfy_root, "models")
        if os.path.isdir(models_dir):
            for entry in os.listdir(models_dir):
                if entry in ("configs", "__pycache__") or entry.startswith("."):
                    continue
                local_sub = os.path.join(models_dir, entry)
                if os.path.isdir(local_sub) and not os.path.islink(local_sub):
                    persist_sub = os.path.join(persist_root, "models", entry)
                    _safe_symlink_dir(persist_sub, local_sub)

        # 2. 将本地高速 NVMe 上的 custom_nodes 增量镜像同步到持久化目录，并记录 Git 源清单
        cn_local = os.path.join(comfy_root, "custom_nodes")
        cn_persist = os.path.join(persist_root, "custom_nodes")
        manifest: Dict[str, Any] = {}
        manifest_path = os.path.join(persist_root, "custom_nodes_manifest.json")
        if os.path.exists(manifest_path):
            try:
                manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
            except Exception:
                manifest = {}

        if os.path.isdir(cn_local):
            for entry in sorted(os.listdir(cn_local)):
                if entry.startswith(".") or entry == "__pycache__":
                    continue
                src_node = os.path.join(cn_local, entry)
                if not os.path.isdir(src_node):
                    continue
                dst_node = os.path.join(cn_persist, entry)
                if shutil.which("rsync"):
                    subprocess.run(
                        ["rsync", "-a", "--delete", "--exclude=__pycache__", f"{src_node}/", f"{dst_node}/"],
                        check=False,
                    )
                else:
                    shutil.copytree(
                        src_node,
                        dst_node,
                        dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                    )
                git_cfg = os.path.join(src_node, ".git", "config")
                if os.path.exists(git_cfg):
                    try:
                        cfg_txt = open(git_cfg, "r", encoding="utf-8", errors="ignore").read()
                        m_url = re.search(r"url\s*=\s*(\S+)", cfg_txt)
                        if m_url:
                            manifest[entry] = {"git_url": m_url.group(1).strip()}
                    except Exception:
                        pass

            with open(manifest_path, "w", encoding="utf-8") as mf:
                json.dump(manifest, mf, ensure_ascii=False, indent=2)

        # 3. 若处于 rclone 云盘镜像模式，通过 Google Cloud 内网 8 并发同步至 Google Drive
        if mode == "rclone_drive_sync" and _has_rclone_drive_remote():
            if not quiet:
                print("[ComfyUI 云盘同步] 正在通过高速通道将自定义节点、模型与工作流同步至 Google Drive...")
            subprocess.run(
                [
                    "rclone", "copy",
                    persist_root,
                    "easycolab_gdrive:EasyColab_Drive/EasyColab_ComfyUI",
                    "--transfers", "8",
                    "--checkers", "16",
                    "--drive-chunk-size", "64M",
                    "--exclude", "**/__pycache__/**",
                    "--exclude", "**/*.part",
                    "--exclude", "**/*.tmp",
                ],
                check=False,
            )

        COMFYUI_STATE["last_sync_ts"] = time.time()
        _COMFY_STATUS_CACHE["ts"] = 0.0
        (cn_count, m_count, wf_count), _ = _scan_comfyui_counts_and_outputs()
        msg = f"已持久化至 {label}（节点 {cn_count} 个 · 权重 {m_count} 个 · 工作流 {wf_count} 个）"
        if not quiet:
            COMFYUI_STATE["status_message"] = msg
            print(f"[ComfyUI 云盘持久化] {msg}")
        return True, msg
    except Exception as e:
        err = f"ComfyUI 云盘同步异常: {e}"
        if not quiet:
            print(f"[ComfyUI] {err}")
        return False, err
    finally:
        COMFYUI_STATE["is_syncing_drive"] = False


def _comfyui_periodic_drive_sync_loop():
    while True:
        time.sleep(60.0)
        if COMFYUI_STATE.get("is_running") and not COMFYUI_STATE.get("is_starting"):
            if COMFYUI_STATE.get("queue_running", 0) == 0:
                sync_comfyui_to_drive_core(quiet=True)


def _ensure_comfyui_manager_tunnel_config(comfy_root: str, persist_root: str):
    """允许通过 trycloudflare 公网隧道在 ComfyUI-Manager 内直接安装 Git 节点与模型，不受默认 normal 安全锁拦截"""
    cfg_targets = [
        os.path.join(persist_root, "user", "default", "ComfyUI-Manager", "config.ini"),
        os.path.join(persist_root, "user", "__manager", "config.ini"),
        os.path.join(comfy_root, "custom_nodes", "ComfyUI-Manager", "config.ini"),
    ]
    for cfg_p in cfg_targets:
        try:
            os.makedirs(os.path.dirname(cfg_p), exist_ok=True)
            if os.path.exists(cfg_p):
                txt = open(cfg_p, "r", encoding="utf-8", errors="ignore").read()
                if "security_level" in txt:
                    new_txt = re.sub(r"security_level\s*=\s*\S+", "security_level = weak", txt)
                    if new_txt != txt:
                        with open(cfg_p, "w", encoding="utf-8") as wf:
                            wf.write(new_txt)
                    continue
            with open(cfg_p, "w", encoding="utf-8") as wf:
                wf.write("[default]\nsecurity_level = weak\npreview_method = auto\n")
        except Exception:
            pass


def _install_persisted_custom_node_requirements(comfy_root: str):
    """扫描已从 Google Drive 恢复的所有自定义节点 requirements.txt，过滤掉 torch/xformers 后批量补齐依赖"""
    import hashlib
    cn_dir = os.path.join(comfy_root, "custom_nodes")
    if not os.path.isdir(cn_dir):
        return
    collected_reqs: List[str] = []
    for entry in sorted(os.listdir(cn_dir)):
        req_file = os.path.join(cn_dir, entry, "requirements.txt")
        if not os.path.isfile(req_file):
            continue
        try:
            for raw_line in open(req_file, "r", encoding="utf-8", errors="ignore"):
                line = raw_line.split("#", 1)[0].strip()
                if not line or line.startswith("-"):
                    continue
                if _COMFY_PROTECTED_PIP_PKGS.match(line):
                    continue
                if line not in collected_reqs:
                    collected_reqs.append(line)
        except Exception:
            pass

    if not collected_reqs:
        return
    digest = hashlib.sha256("\n".join(collected_reqs).encode("utf-8")).hexdigest()
    stamp_file = "/tmp/.easycolab_comfy_node_reqs.sha256"
    if os.path.exists(stamp_file):
        try:
            if open(stamp_file, "r").read().strip() == digest:
                return
        except Exception:
            pass

    print(f"[ComfyUI] 正在为已持久化的自定义节点自动恢复 {len(collected_reqs)} 项 Python 依赖...")
    tmp_req = "/tmp/easycolab_comfy_merged_reqs.txt"
    with open(tmp_req, "w", encoding="utf-8") as wf:
        wf.write("\n".join(collected_reqs) + "\n")
    rc = subprocess.call([sys.executable, "-m", "pip", "install", "-q", "-r", tmp_req])
    if rc == 0:
        with open(stamp_file, "w") as sf:
            sf.write(digest)


def _ensure_cloudflared_binary() -> str:
    cf_bin = shutil.which("cloudflared") or "/usr/local/bin/cloudflared"
    if os.path.isfile(cf_bin) and os.access(cf_bin, os.X_OK):
        return cf_bin
    dl_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    os.makedirs(os.path.dirname(cf_bin), exist_ok=True)
    urllib.request.urlretrieve(dl_url, cf_bin)
    os.chmod(cf_bin, 0o755)
    return cf_bin


def stop_comfyui_core(sync_before_stop: bool = True) -> Tuple[bool, str]:
    with COMFYUI_LOCK:
        if sync_before_stop and (COMFYUI_STATE.get("is_running") or os.path.isdir("/content/ComfyUI")):
            sync_comfyui_to_drive_core(quiet=True)
        for key in ("tunnel_proc", "comfy_proc"):
            proc = COMFYUI_STATE.get(key)
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=5.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                COMFYUI_STATE[key] = None
        subprocess.run(["pkill", "-f", "ComfyUI/main.py"], check=False)
        subprocess.run(["pkill", "-f", "cloudflared.*8188"], check=False)
        if os.path.exists(_COMFY_STATE_FILE):
            try:
                os.remove(_COMFY_STATE_FILE)
            except Exception:
                pass
        _reclaim_transient_gpu_memory()
        COMFYUI_STATE["is_running"] = False
        COMFYUI_STATE["is_starting"] = False
        COMFYUI_STATE["public_url"] = ""
        COMFYUI_STATE["queue_running"] = 0
        COMFYUI_STATE["queue_pending"] = 0
        COMFYUI_STATE["progress"] = 0.0
        COMFYUI_STATE["status_message"] = "ComfyUI 已停止（节点、模型与工作流已持久化保存至 Google Drive）"
        print("[ComfyUI] 服务与公网隧道已停止，GPU 显存已回收。")
        return True, COMFYUI_STATE["status_message"]


def start_comfyui_core(unload_existing: bool = False, force_restart: bool = False) -> Tuple[bool, str]:
    global _COMFY_BG_SYNC_STARTED
    touch_activity()
    if COMFYUI_STATE["is_starting"]:
        return True, "ComfyUI 正在部署启动中，请稍候..."
    if COMFYUI_STATE["is_running"] and COMFYUI_STATE.get("public_url") and not force_restart:
        return True, f"ComfyUI 已在运行: {COMFYUI_STATE['public_url']}"

    if force_restart and COMFYUI_STATE["is_running"]:
        stop_comfyui_core(sync_before_stop=True)

    COMFYUI_STATE["is_starting"] = True
    COMFYUI_STATE["progress"] = 0.06
    COMFYUI_STATE["status_message"] = "正在初始化 ComfyUI 环境与 Google Drive 持久化目录..."

    try:
        if unload_existing:
            COMFYUI_STATE["status_message"] = "正在释放现有常驻模型显存，为 ComfyUI 腾出完整 96GB 空间..."
            unload_model_from_vram_core("all")

        comfy_root, persist_root, mode, label, is_drive = _resolve_comfyui_persistence_paths()

        # 1. 若为 rclone 云盘同步模式且本实例首次拉起，先从 Google Drive 高速拉取已持久化的节点、模型与工作流
        pull_flag = "/tmp/.easycolab_comfy_drive_pulled"
        if mode == "rclone_drive_sync" and _has_rclone_drive_remote() and not os.path.exists(pull_flag):
            COMFYUI_STATE["progress"] = 0.14
            COMFYUI_STATE["status_message"] = "正在从 Google Drive 恢复历史持久化的 ComfyUI 节点、模型与工作流..."
            print("[ComfyUI] 正在从 Google Drive 同步已保存的节点、模型与工作流...")
            os.makedirs(persist_root, exist_ok=True)
            subprocess.run(
                [
                    "rclone", "copy",
                    "easycolab_gdrive:EasyColab_Drive/EasyColab_ComfyUI",
                    persist_root,
                    "--transfers", "8",
                    "--checkers", "16",
                    "--drive-chunk-size", "64M",
                ],
                check=False,
            )
            with open(pull_flag, "w") as pf:
                pf.write("1")

        # 2. 克隆或检查 ComfyUI 主程序 (置于本地高速 NVMe，保证秒级导入)
        if not os.path.exists(os.path.join(comfy_root, "main.py")):
            COMFYUI_STATE["progress"] = 0.24
            COMFYUI_STATE["status_message"] = "正在克隆官方最新 ComfyUI 核心引擎..."
            print("[ComfyUI] 正在克隆官方最新 ComfyUI 仓库...")
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/comfyanonymous/ComfyUI.git", comfy_root],
                check=True,
            )
            COMFYUI_STATE["progress"] = 0.36
            COMFYUI_STATE["status_message"] = "正在安装 ComfyUI 核心依赖（保留当前 G4 Blackwell CUDA PyTorch）..."
            raw_reqs = open(os.path.join(comfy_root, "requirements.txt"), "r", encoding="utf-8").readlines()
            safe_reqs = [
                ln.strip() for ln in raw_reqs
                if ln.strip() and not ln.strip().startswith("#") and not _COMFY_PROTECTED_PIP_PKGS.match(ln)
            ]
            tmp_core_req = "/tmp/comfy_core_safe_reqs.txt"
            with open(tmp_core_req, "w", encoding="utf-8") as wf:
                wf.write("\n".join(safe_reqs) + "\n")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", tmp_core_req], check=True)
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "gguf", "opencv-python-headless", "imageio-ffmpeg", "gitpython", "toml"],
                check=False,
            )

        # 3. 建立 Google Drive 软链体系：models/* 各子目录、user 工作流配置、output 成片输出全部自动软链至持久化根目录
        COMFYUI_STATE["progress"] = 0.48
        COMFYUI_STATE["status_message"] = f"正在建立 {label} 目录软链 (models / workflows / output)..."
        os.makedirs(os.path.join(persist_root, "models"), exist_ok=True)
        all_model_subdirs = set(COMFY_STANDARD_MODEL_SUBDIRS)
        for existing_sub in os.listdir(os.path.join(persist_root, "models")):
            if not existing_sub.startswith(".") and os.path.isdir(os.path.join(persist_root, "models", existing_sub)):
                all_model_subdirs.add(existing_sub)
        for sub in sorted(all_model_subdirs):
            _safe_symlink_dir(
                os.path.join(persist_root, "models", sub),
                os.path.join(comfy_root, "models", sub),
            )
        _safe_symlink_dir(os.path.join(persist_root, "user"), os.path.join(comfy_root, "user"))
        _safe_symlink_dir(os.path.join(persist_root, "output"), os.path.join(comfy_root, "output"))
        _ensure_comfyui_beginner_assets(comfy_root, persist_root)

        # 若云盘中已有任何预置包的全部权重，但缺少中文工作流文件或工作流版本尚未升级至 v2 (修复 widgets_values_named 兼容)，自动重新生成
        wf_dir_chk = os.path.join(persist_root, "user", "default", "workflows")
        schema_v2_ok = os.path.isfile(os.path.join(wf_dir_chk, ".easycolab_wf_schema_v2"))
        for pk in COMFY_PRESET_WORKFLOW_PACKS:
            if all(
                os.path.isfile(os.path.join(persist_root, "models", sub, fn))
                for sub, fn, _, _, _ in pk["models"]
            ):
                if (not schema_v2_ok) or (not all(os.path.isfile(os.path.join(wf_dir_chk, w)) for w in pk["workflows"])):
                    _build_and_install_comfy_preset_workflows(persist_root, [pk["id"]])

        # 4. 恢复已持久化的自定义节点至本地高速 NVMe，并预装 ComfyUI-Manager 与 VideoHelperSuite 视频核心节点
        COMFYUI_STATE["progress"] = 0.58
        COMFYUI_STATE["status_message"] = "正在恢复已持久化的自定义节点并配置 ComfyUI-Manager..."
        cn_local = os.path.join(comfy_root, "custom_nodes")
        cn_persist = os.path.join(persist_root, "custom_nodes")
        os.makedirs(cn_local, exist_ok=True)
        os.makedirs(cn_persist, exist_ok=True)

        for entry in sorted(os.listdir(cn_persist)):
            if entry.startswith(".") or entry in ("__pycache__", "easycolab_preset_helper"):
                continue
            src_p = os.path.join(cn_persist, entry)
            dst_p = os.path.join(cn_local, entry)
            if os.path.isdir(src_p) and not os.path.exists(dst_p):
                if shutil.which("rsync"):
                    subprocess.run(["rsync", "-a", f"{src_p}/", f"{dst_p}/"], check=False)
                else:
                    shutil.copytree(src_p, dst_p, dirs_exist_ok=True)

        # 若 manifest 中记录了节点但目录缺失，自动按 Git URL 补齐
        manifest_path = os.path.join(persist_root, "custom_nodes_manifest.json")
        if os.path.exists(manifest_path):
            try:
                manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
                for node_name, meta in manifest.items():
                    git_url = (meta or {}).get("git_url", "")
                    dst_p = os.path.join(cn_local, node_name)
                    if git_url and not os.path.exists(dst_p):
                        subprocess.run(["git", "clone", "--depth", "1", git_url, dst_p], check=False)
            except Exception:
                pass

        default_nodes = [
            ("ComfyUI-Manager", "https://github.com/ltdrdata/ComfyUI-Manager.git"),
            ("ComfyUI-VideoHelperSuite", "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"),
        ]
        for node_dir_name, repo_url in default_nodes:
            target_node_dir = os.path.join(cn_local, node_dir_name)
            if not os.path.exists(target_node_dir):
                print(f"[ComfyUI] 正在预装核心插件 {node_dir_name}...")
                subprocess.run(["git", "clone", "--depth", "1", repo_url, target_node_dir], check=False)

        _ensure_easycolab_comfy_preset_helper(comfy_root, persist_root)
        _ensure_comfyui_manager_tunnel_config(comfy_root, persist_root)

        COMFYUI_STATE["progress"] = 0.70
        COMFYUI_STATE["status_message"] = "正在检查并安装已保存自定义节点的 Python 依赖包..."
        _install_persisted_custom_node_requirements(comfy_root)
        sync_comfyui_to_drive_core(quiet=True)

        # 5. 启动 ComfyUI 服务进程 (127.0.0.1:8188) 并实时镜像输出至 EasyColab 日志面板
        COMFYUI_STATE["progress"] = 0.80
        COMFYUI_STATE["status_message"] = "正在拉起 ComfyUI 节点推理引擎 (端口 8188)..."
        subprocess.run(["pkill", "-f", "ComfyUI/main.py"], check=False)
        subprocess.run(["pkill", "-f", "cloudflared.*8188"], check=False)
        time.sleep(0.6)

        port = int(COMFYUI_STATE.get("local_port") or 8188)
        comfy_cmd = [
            sys.executable, "-u", "main.py",
            "--listen", "127.0.0.1",
            "--port", str(port),
            "--enable-cors-header", "*",
            "--preview-method", "auto",
        ]
        comfy_proc = subprocess.Popen(
            comfy_cmd,
            cwd=comfy_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        COMFYUI_STATE["comfy_proc"] = comfy_proc

        def _forward_comfy_logs(proc: subprocess.Popen):
            try:
                if proc.stdout is None:
                    return
                for raw_ln in proc.stdout:
                    ln = raw_ln.rstrip()
                    if ln:
                        print(f"[ComfyUI] {ln}")
            except Exception:
                pass

        threading.Thread(target=_forward_comfy_logs, args=(comfy_proc,), daemon=True).start()

        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        ready = False
        for _ in range(90):
            if comfy_proc.poll() is not None:
                raise RuntimeError(f"ComfyUI 启动失败提前退出 (code={comfy_proc.returncode})，请查看右侧运行日志。")
            try:
                with opener.open(f"http://127.0.0.1:{port}/system_stats", timeout=1.5) as resp:
                    if resp.status == 200:
                        ready = True
                        break
            except Exception:
                pass
            time.sleep(1.0)

        if not ready:
            raise RuntimeError("等待 ComfyUI 本地端口 8188 就绪超时，请查看右侧运行日志。")

        # 6. 启动 Cloudflare WebSocket 高速公网隧道
        COMFYUI_STATE["progress"] = 0.92
        COMFYUI_STATE["status_message"] = "ComfyUI 内核已就绪，正在建立 Cloudflare 高速全屏画布隧道..."
        cf_bin = _ensure_cloudflared_binary()
        tunnel_proc = subprocess.Popen(
            [cf_bin, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        COMFYUI_STATE["tunnel_proc"] = tunnel_proc

        cf_url_box: List[str] = [""]
        cf_re = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

        def _watch_cf_logs(proc: subprocess.Popen):
            try:
                if proc.stdout is None:
                    return
                for raw_ln in proc.stdout:
                    m = cf_re.search(raw_ln)
                    if m and not cf_url_box[0]:
                        cf_url_box[0] = m.group(0)
            except Exception:
                pass

        threading.Thread(target=_watch_cf_logs, args=(tunnel_proc,), daemon=True).start()
        for _ in range(45):
            if cf_url_box[0]:
                break
            time.sleep(0.5)

        if not cf_url_box[0]:
            raise RuntimeError("未能获取 Cloudflare 公网隧道地址，请重试。")

        COMFYUI_STATE["public_url"] = cf_url_box[0]
        COMFYUI_STATE["is_running"] = True
        COMFYUI_STATE["progress"] = 1.0
        try:
            with open(_COMFY_STATE_FILE, "w", encoding="utf-8") as sf:
                json.dump({"public_url": cf_url_box[0], "port": port}, sf)
        except Exception:
            pass
        _COMFY_STATUS_CACHE["ts"] = 0.0
        (cn_count, m_count, _), _ = _scan_comfyui_counts_and_outputs()
        ok_msg = f"ComfyUI 已就绪（{label} · 节点 {cn_count} 个 · 权重 {m_count} 个）"
        COMFYUI_STATE["status_message"] = ok_msg
        print(f"[ComfyUI] 画布公网隧道已开通: {cf_url_box[0]}")

        if not _COMFY_BG_SYNC_STARTED:
            _COMFY_BG_SYNC_STARTED = True
            threading.Thread(target=_comfyui_periodic_drive_sync_loop, daemon=True).start()

        return True, ok_msg
    except Exception as e:
        err_msg = f"ComfyUI 启动出错: {e}"
        COMFYUI_STATE["status_message"] = err_msg
        COMFYUI_STATE["progress"] = 0.0
        print(f"[ComfyUI] {err_msg}")
        return False, err_msg
    finally:
        COMFYUI_STATE["is_starting"] = False
        touch_activity()


def download_comfyui_model_core(
    url_or_repo: str,
    filename: str = "",
    subfolder: str = "diffusion_models",
    hf_token: str = "",
) -> Tuple[bool, str]:
    """8 线程并发下载模型权重直达持久化的 ComfyUI/models/<subfolder>，下次重开 G4 免重复下载"""
    touch_activity()
    raw_src = (url_or_repo or "").strip()
    if not raw_src:
        return False, "请输入 HuggingFace 仓库/直链或模型下载 URL"
    if COMFYUI_STATE["is_downloading_weight"]:
        return False, "当前已有 ComfyUI 权重正在下载中，请稍候"

    clean_sub = re.sub(r"[^a-zA-Z0-9_.-]", "", (subfolder or "diffusion_models").strip()) or "diffusion_models"
    comfy_root, persist_root, mode, label, _ = _resolve_comfyui_persistence_paths()
    dest_dir = os.path.join(persist_root, "models", clean_sub)
    os.makedirs(dest_dir, exist_ok=True)
    if os.path.isdir(comfy_root):
        _safe_symlink_dir(dest_dir, os.path.join(comfy_root, "models", clean_sub))

    COMFYUI_STATE["is_downloading_weight"] = True
    COMFYUI_STATE["progress"] = 0.05
    try:
        token = (hf_token or "").strip() or None
        hf_m = re.match(r"^https?://huggingface\.co/([^/]+/[^/]+)/(?:resolve|blob)/[^/]+/(.+?)(\?.*)?$", raw_src)
        if hf_m:
            repo_id = hf_m.group(1)
            rel_file = hf_m.group(2)
            base_name = os.path.basename(filename.strip() or rel_file)
            COMFYUI_STATE["status_message"] = f"正在 8 线程极速下载 {base_name} 至 models/{clean_sub}..."
            print(f"[ComfyUI 极速下载] {repo_id} -> {rel_file} -> models/{clean_sub}/{base_name}")
            dl_path = fast_parallel_hf_download(repo_id=repo_id, filename=rel_file, dest_dir=dest_dir, hf_token=token)
            final_path = os.path.join(dest_dir, base_name)
            if os.path.abspath(dl_path) != os.path.abspath(final_path):
                shutil.move(dl_path, final_path)
        elif raw_src.startswith("http://") or raw_src.startswith("https://"):
            url_path = urllib.parse.urlparse(raw_src).path
            base_name = os.path.basename(filename.strip() or url_path) or f"model_{int(time.time())}.safetensors"
            final_path = os.path.join(dest_dir, base_name)
            COMFYUI_STATE["status_message"] = f"正在下载 {base_name} 至 models/{clean_sub}..."
            print(f"[ComfyUI 直链下载] {raw_src} -> models/{clean_sub}/{base_name}")
            headers = {"User-Agent": "EasyColab/2.0"}
            if token and "huggingface.co" in raw_src:
                headers["Authorization"] = f"Bearer {token}"
            req = urllib.request.Request(raw_src, headers=headers)
            tmp_p = final_path + ".part"
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp_p, "wb") as wf:
                total_sz = int(resp.headers.get("Content-Length") or 0)
                done_sz = 0
                t0 = time.time()
                last_log = 0.0
                while True:
                    chunk = resp.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    wf.write(chunk)
                    done_sz += len(chunk)
                    now_t = time.time()
                    if total_sz > 0 and (now_t - last_log) >= 0.6:
                        last_log = now_t
                        pct = done_sz / float(total_sz)
                        COMFYUI_STATE["progress"] = round(min(0.98, pct), 3)
                        spd = (done_sz / (1024 ** 2)) / max(0.01, now_t - t0)
                        print(f"{base_name}: {int(pct * 100)}%| {done_sz / (1024**3):.2f}G/{total_sz / (1024**3):.2f}G [{spd:.1f}MB/s]")
            os.replace(tmp_p, final_path)
        else:
            if not filename.strip():
                return False, "填写 HuggingFace 仓库 ID (如 Comfy-Org/Wan_2.1_ComfyUI_repackaged) 时请同时填写文件路径"
            rel_file = filename.strip()
            base_name = os.path.basename(rel_file)
            COMFYUI_STATE["status_message"] = f"正在 8 线程极速下载 {base_name} 至 models/{clean_sub}..."
            dl_path = fast_parallel_hf_download(repo_id=raw_src, filename=rel_file, dest_dir=dest_dir, hf_token=token)
            final_path = os.path.join(dest_dir, base_name)
            if os.path.abspath(dl_path) != os.path.abspath(final_path):
                shutil.move(dl_path, final_path)

        if mode == "rclone_drive_sync":
            sync_comfyui_to_drive_core(quiet=True)
        _COMFY_STATUS_CACHE["ts"] = 0.0
        sz_gb = os.path.getsize(final_path) / (1024 ** 3)
        msg = f"权重 {base_name} ({sz_gb:.2f} GB) 已下载至 models/{clean_sub} 并持久化到 {label}（在画布按 R 键即可刷新选用）"
        COMFYUI_STATE["progress"] = 1.0
        COMFYUI_STATE["status_message"] = msg
        print(f"[ComfyUI] {msg}")
        return True, msg
    except Exception as e:
        err = f"下载 ComfyUI 权重失败: {e}"
        COMFYUI_STATE["status_message"] = err
        print(f"[ComfyUI] {err}")
        return False, err
    finally:
        COMFYUI_STATE["is_downloading_weight"] = False
        touch_activity()


def install_comfyui_preset_pack_core(
    pack_id: str,
    auto_start: bool = True,
    hf_token: str = "",
) -> Tuple[bool, str]:
    """一键配齐预置中文工作流包：8 线程下载缺失模型权重 + 写入中文傻瓜注释工作流 + 生成默认参考图 + 自动拉起/刷新画布"""
    touch_activity()
    target_pack = next((p for p in COMFY_PRESET_WORKFLOW_PACKS if p["id"] == pack_id), None)
    if not target_pack:
        return False, f"未找到预置工作流包: {pack_id}"
    if COMFYUI_STATE["is_downloading_weight"] or COMFYUI_STATE.get("installing_pack_id"):
        return False, "当前已有模型或工作流包正在高速下载配齐中，请稍候..."

    comfy_root, persist_root, mode, label, _ = _resolve_comfyui_persistence_paths()
    models_list = target_pack["models"]
    total_m = len(models_list)

    COMFYUI_STATE["is_downloading_weight"] = True
    COMFYUI_STATE["installing_pack_id"] = pack_id
    COMFYUI_STATE["progress"] = 0.03
    COMFYUI_STATE["status_message"] = f"正在准备配齐「{target_pack['title']}」（共 {total_m} 个官方权重）..."

    try:
        token = (hf_token or "").strip() or ""
        # 若为 rclone 模式且尚未拉取过云盘历史权重，先快速检查恢复，避免重复下载已持久化的权重
        pull_flag = "/tmp/.easycolab_comfy_drive_pulled"
        if mode == "rclone_drive_sync" and _has_rclone_drive_remote() and not os.path.exists(pull_flag):
            COMFYUI_STATE["status_message"] = "正在先从 Google Drive 检查已持久化的历史模型权重..."
            os.makedirs(persist_root, exist_ok=True)
            subprocess.run(
                [
                    "rclone", "copy",
                    "easycolab_gdrive:EasyColab_Drive/EasyColab_ComfyUI",
                    persist_root,
                    "--transfers", "8",
                    "--checkers", "16",
                    "--drive-chunk-size", "64M",
                ],
                check=False,
            )
            with open(pull_flag, "w") as pf:
                pf.write("1")

        _ensure_comfyui_beginner_assets(comfy_root, persist_root)

        for idx, (sub, fname, repo_id, repo_rel_path, approx_gb) in enumerate(models_list, start=1):
            touch_activity()
            dest_dir = os.path.join(persist_root, "models", sub)
            os.makedirs(dest_dir, exist_ok=True)
            if os.path.isdir(comfy_root):
                _safe_symlink_dir(dest_dir, os.path.join(comfy_root, "models", sub))

            final_path = os.path.join(dest_dir, fname)
            base_prog = 0.05 + 0.80 * ((idx - 1) / float(max(1, total_m)))
            COMFYUI_STATE["progress"] = round(base_prog, 3)

            if os.path.isfile(final_path) and os.path.getsize(final_path) > 1024 * 1024:
                print(f"[ComfyUI 预置包] ({idx}/{total_m}) 已存在跳过: models/{sub}/{fname}")
                continue

            COMFYUI_STATE["status_message"] = (
                f"[{idx}/{total_m}] 正在 8 线程极速下载 {fname} (约 {approx_gb:.1f} GB)..."
            )
            print(f"[ComfyUI 预置包] ({idx}/{total_m}) 开始 8 线程下载 {repo_id}/{repo_rel_path} -> models/{sub}/{fname}")
            dl_path = fast_parallel_hf_download(
                repo_id=repo_id,
                filename=repo_rel_path,
                dest_dir=dest_dir,
                hf_token=token,
                num_connections=8,
            )
            if os.path.abspath(dl_path) != os.path.abspath(final_path):
                shutil.move(dl_path, final_path)
                # 清理 fast_parallel_hf_download 因嵌套路径创建的空子目录 (如 split_files/*)
                top_nested = repo_rel_path.split("/", 1)[0]
                nested_dir = os.path.join(dest_dir, top_nested)
                if top_nested and os.path.isdir(nested_dir):
                    try:
                        shutil.rmtree(nested_dir, ignore_errors=True)
                    except Exception:
                        pass

        COMFYUI_STATE["progress"] = 0.88
        COMFYUI_STATE["status_message"] = f"正在生成「{target_pack['title']}」全中文三步傻瓜注释工作流..."
        installed_wfs = _build_and_install_comfy_preset_workflows(persist_root, [pack_id])
        if os.path.isdir(comfy_root):
            _safe_symlink_dir(os.path.join(persist_root, "user"), os.path.join(comfy_root, "user"))
            _safe_symlink_dir(os.path.join(persist_root, "output"), os.path.join(comfy_root, "output"))
            _ensure_comfyui_beginner_assets(comfy_root, persist_root)

        if mode == "rclone_drive_sync":
            COMFYUI_STATE["progress"] = 0.93
            COMFYUI_STATE["status_message"] = "正在将配齐的模型与中文工作流持久化同步至 Google Drive..."
            sync_comfyui_to_drive_core(quiet=True)

        _COMFY_STATUS_CACHE["ts"] = 0.0
        wf_names_str = "、".join(installed_wfs) if installed_wfs else "、".join(target_pack["workflows"])

        # 若 ComfyUI 正在运行，主动轻触 /object_info 刷新模型下拉缓存；若尚未启动且开启了 auto_start，直接自动拉起画布
        if COMFYUI_STATE.get("is_running"):
            try:
                port = int(COMFYUI_STATE.get("local_port") or 8188)
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(f"http://127.0.0.1:{port}/object_info/UNETLoader", timeout=3.0) as _:
                    pass
            except Exception:
                pass
            COMFYUI_STATE["progress"] = 1.0
            ok_msg = f"「{target_pack['title']}」已全部配齐！打开画布左侧「工作流」点击 {wf_names_str} 即可直接传图运行"
            COMFYUI_STATE["status_message"] = ok_msg
            print(f"[ComfyUI 预置包] {ok_msg}")
            return True, ok_msg
    except Exception as e:
        err = f"配齐预置工作流包失败: {e}"
        COMFYUI_STATE["progress"] = 0.0
        COMFYUI_STATE["status_message"] = err
        print(f"[ComfyUI 预置包] {err}")
        return False, err
    finally:
        COMFYUI_STATE["is_downloading_weight"] = False
        COMFYUI_STATE["installing_pack_id"] = ""
        touch_activity()

    # 若尚未拉起 ComfyUI，在释放下载锁后自动拉起云端画布隧道
    if auto_start and not COMFYUI_STATE.get("is_running") and not COMFYUI_STATE.get("is_starting"):
        print("[ComfyUI 预置包] 权重与工作流已就绪，正在自动拉起 ComfyUI 全屏画布...")
        ok_s, msg_s = start_comfyui_core(unload_existing=False, force_restart=False)
        if ok_s:
            final_msg = f"「{target_pack['title']}」已配齐且画布已启动！在左侧「工作流」打开中文模板即可直接出片"
            COMFYUI_STATE["status_message"] = final_msg
            return True, final_msg
        return False, msg_s

    return True, f"「{target_pack['title']}」模型与中文工作流已配齐！"


def disconnect_colab_runtime() -> str:
    try:
        if os.path.isdir("/content/ComfyUI") or COMFYUI_STATE.get("is_running"):
            sync_comfyui_to_drive_core(quiet=True)
    except Exception:
        pass
    try:
        from google.colab import runtime
        runtime.unassign()
        return "🛑 已向 Colab 发送释放运行时指令，实例即将关闭并停止扣除计算单元。"
    except ImportError:
        return "ℹ️ 当前不在 Google Colab 环境中，无需释放云端运行时。"
    except Exception as e:
        return f"⚠️ 释放运行时失败: {e}"


# ==============================================================================
# 供 macOS 原生客户端调用的异步 REST API 接口 (/minimax_api/* · 多模型通用)
# ==============================================================================
class SelectModelRequest(BaseModel):
    model_id: str = "minimax_h3"


class DownloadModelRequest(BaseModel):
    model_id: Optional[str] = None
    hf_token: str = ""


class LoadModelRequest(BaseModel):
    model_id: Optional[str] = None
    precision_mode: str = "bf16"
    offload_mode: str = "disk"
    hf_token: str = ""
    context_length: Optional[int] = None


class ContextConfigRequest(BaseModel):
    context_length: int = 32768


class IdleConfigRequest(BaseModel):
    idle_timeout_minutes: int = 30


class ComfyStartRequest(BaseModel):
    unload_existing: bool = False
    force_restart: bool = False


class ComfyWeightDownloadRequest(BaseModel):
    url_or_repo: str
    filename: str = ""
    subfolder: str = "diffusion_models"
    hf_token: str = ""


class ComfyInstallPackRequest(BaseModel):
    pack_id: str
    auto_start: bool = True
    hf_token: str = ""


class GenerateVideoRequest(BaseModel):
    model_id: Optional[str] = None
    task_mode: str = "t2va"
    visual_prompt: str
    negative_prompt: str = ""
    soundscape_prompt: str = ""
    music_prompt: str = ""
    first_frame_base64: Optional[str] = None
    last_frame_base64: Optional[str] = None
    width: int = 832
    height: int = 480
    num_frames: int = 124
    num_inference_steps: int = 35
    fps: int = 24
    seed: int = 42
    randomize_seed: bool = True
    batch_count: int = 1
    denoising_strength: float = 0.70
    enable_thinking: bool = True
    max_new_tokens: int = 16384
    context_length: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: Optional[float] = None
    repeat_penalty: float = 1.0
    reasoning_effort: str = "medium"
    chat_messages: Optional[List[Dict[str, str]]] = None


def decode_base64_image(b64_str: Optional[str]) -> Optional[Image.Image]:
    if not b64_str:
        return None
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _split_thinking_and_answer(raw_text: str) -> Tuple[str, str]:
    """解析 Qwen3.8 <think>...</think> 推理链与最终回复"""
    text = raw_text or ""
    if "</think>" in text:
        think_part, ans_part = text.split("</think>", 1)
        think_part = think_part.replace("<think>", "").strip()
        return think_part, ans_part.strip()
    if text.lstrip().startswith("<think>"):
        return text.replace("<think>", "", 1).strip(), ""
    return "", text.strip()


def _is_bundle_alive(bundle: Optional[Dict[str, Any]]) -> bool:
    if not bundle:
        return False
    if bundle.get("backend") == "gguf":
        proc = bundle.get("server_proc")
        return proc is not None and proc.poll() is None
    return bundle.get("model") is not None


def _resolve_model_alias(name: Optional[str]) -> Optional[str]:
    """把外部传入的 model 名（repo_id / 展示名 / 文件名 / 面板 id）解析成 MODEL_CATALOG 的 key"""
    if not name:
        return None
    raw = str(name).strip()
    if raw in MODEL_CATALOG:
        return raw
    low = raw.lower()
    # 常见 OpenAI 客户端会传 "repo_id:quant" 或纯 repo_id
    cand = low.split(":")[0]
    for mid, spec in MODEL_CATALOG.items():
        repo = str(spec.get("repo_id") or "").lower()
        if not repo:
            continue
        if cand == repo or low == repo:
            return mid
        gguf_main = str(spec.get("gguf_main") or "").lower()
        if gguf_main and (low == gguf_main or os.path.splitext(os.path.basename(gguf_main))[0] == cand):
            return mid
    for mid, spec in MODEL_CATALOG.items():
        disp = str(spec.get("name") or "").strip().lower()
        if disp and (disp == low or low in disp):
            return mid
    return None


_QWEN_FAMILY_IDS = ("qwen38_27b_gguf", "qwen38_27b")


def _get_qwen_llm_bundle(preferred_model_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    定位已加载的 LLM 运行时 bundle。

    语义：
      - preferred 能解析到具体模型（含自定义 GGUF）时只认它 + 同家族回退（Qwen 的两个变体可互补），
        绝不借用无关的自定义模型 —— 否则会静默串到别的模型上回答。
      - 未指定 preferred 时，先看当前选中模型，再退回任意已加载的 GGUF/LLM。
    """
    candidates: List[str] = []
    if preferred_model_id:
        alias = _resolve_model_alias(preferred_model_id)
        if alias:
            candidates.append(alias)
        low = str(preferred_model_id).lower()
        if not alias or alias in _QWEN_FAMILY_IDS:
            if "gguf" in low or "q4_k" in low or "unsloth" in low:
                candidates.append("qwen38_27b_gguf")
            elif "qwen3.8" in low or "qwen38" in low:
                candidates.extend(_QWEN_FAMILY_IDS)
        if _is_llm_model_id(SELECTED_MODEL_ID) and SELECTED_MODEL_ID not in candidates:
            candidates.append(SELECTED_MODEL_ID)
    else:
        if _is_llm_model_id(SELECTED_MODEL_ID):
            candidates.append(SELECTED_MODEL_ID)
        candidates.extend(_QWEN_FAMILY_IDS)

    for cid in candidates:
        rt = LOADED_RUNTIMES.get(cid)
        if rt:
            b = rt.get("llm_bundle")
            if _is_bundle_alive(b):
                return b
            # GGUF 子进程意外退出时清理陈旧表项
            if b and b.get("backend") == "gguf":
                LOADED_RUNTIMES.pop(cid, None)

    # 未指定模型时的兜底：任意已加载的自定义 GGUF 都可以直接服务
    if not preferred_model_id:
        for mid in list(LOADED_RUNTIMES.keys()):
            if mid in CUSTOM_MODEL_SPECS:
                b = LOADED_RUNTIMES[mid].get("llm_bundle")
                if _is_bundle_alive(b):
                    return b

    if _is_bundle_alive(LLM_BUNDLE):
        return LLM_BUNDLE
    return None


def _count_bundle_tokens(bundle: Optional[Dict[str, Any]], text: str) -> int:
    if not text:
        return 1
    if bundle is not None:
        if bundle.get("backend") == "gguf":
            try:
                opener = _make_noproxy_opener()
                tok_req = urllib.request.Request(
                    f"{bundle['base_url']}/tokenize",
                    data=json.dumps({"content": text}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with opener.open(tok_req, timeout=4.0) as resp:
                    obj = json.loads(resp.read().decode("utf-8"))
                    toks = obj.get("tokens")
                    if isinstance(toks, list):
                        return max(1, len(toks))
            except Exception:
                pass
        elif bundle.get("processor") is not None:
            try:
                return max(1, len(bundle["processor"].tokenizer.encode(text)))
            except Exception:
                pass
    return max(1, len(text) // 3)


def _run_qwen_llm_job(job_id: str, req: GenerateVideoRequest, target_llm_id: Optional[str] = None):
    """运行 Qwen3.8-27B (BF16 原生 或 Unsloth UD-Q4_K_M GGUF) 深度推理对话与多模态视觉理解"""
    global QWEN_CONTEXT_LENGTH
    start_time = time.time()
    if req.context_length is not None:
        QWEN_CONTEXT_LENGTH = _clamp_context_length(req.context_length)

    bundle = _get_qwen_llm_bundle(preferred_model_id=target_llm_id or req.model_id)
    if bundle is None:
        want = target_llm_id or req.model_id or SELECTED_MODEL_ID or "LLM 模型"
        spec = CUSTOM_MODEL_SPECS.get(want) or MODEL_CATALOG.get(want, {})
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["message"] = f"{spec.get('name') or want} 尚未加载至 GPU 显存！请先在「模型清单」点击「加载到显存」。"
        return

    model_label = bundle.get("model_label") or (
        CUSTOM_MODEL_SPECS.get(bundle.get("model_id", ""), {}).get("name")
        or MODEL_CATALOG.get(bundle.get("model_id", ""), {}).get("name")
        or ("GGUF" if bundle.get("backend") == "gguf" else "Qwen3.8-27B")
    )
    first_img = decode_base64_image(req.first_frame_base64)
    last_img = decode_base64_image(req.last_frame_base64)
    attached_images = [im for im in (first_img, last_img) if im is not None]

    messages: List[Dict[str, Any]] = []
    if req.chat_messages:
        for m in req.chat_messages[:-1]:
            role = m.get("role", "user")
            content = m.get("content", "")
            if content:
                messages.append({"role": role, "content": [{"type": "text", "text": content}]})

    user_content: List[Dict[str, Any]] = []
    for im in attached_images:
        user_content.append({"type": "image", "image": im})
    user_content.append({"type": "text", "text": req.visual_prompt})
    messages.append({"role": "user", "content": user_content})

    JOBS[job_id]["media_type"] = "text"
    JOBS[job_id]["progress"] = 0.12
    JOBS[job_id]["message"] = f"{model_label} 正在编码多模态上下文并启动推理..."

    job_cancel_event = _JOB_CANCEL_EVENTS.setdefault(job_id, threading.Event())
    if job_cancel_event.is_set():
        raise GenerationAbortedError("已手动停止当前推理任务")
    ABORT_GENERATION_EVENT.clear()
    bundle_ctx = int(bundle.get("context_length") or QWEN_CONTEXT_LENGTH)
    max_toks = max(64, min(65536, int(req.max_new_tokens or 16384), max(256, bundle_ctx - 256)))
    think_on = bool(req.enable_thinking)
    # 遵循 Unsloth Qwen3.8 官方部署采样表：Thinking (temp=1.0, top_p=0.95, top_k=20, min_p=0.0, presence=0.0)；Non-Thinking (temp=0.7, top_p=0.80, top_k=20, min_p=0.0, presence=1.5)
    default_temp = 1.0 if think_on else 0.7
    default_top_p = 0.95 if think_on else 0.80
    default_presence = 0.0 if think_on else 1.5
    temp = max(0.0, min(2.0, float(req.temperature if req.temperature is not None else default_temp)))
    eff_top_p = max(0.05, min(1.0, float(req.top_p if req.top_p is not None else default_top_p)))
    eff_presence = max(0.0, min(2.0, float(req.presence_penalty if req.presence_penalty is not None else default_presence)))
    eff_top_k = max(0, int(req.top_k if req.top_k is not None else 20))
    eff_min_p = max(0.0, min(1.0, float(req.min_p if req.min_p is not None else 0.0)))
    eff_repeat = max(0.5, min(2.0, float(req.repeat_penalty if req.repeat_penalty is not None else 1.0)))
    eff_effort = str(req.reasoning_effort or "medium").strip().lower()

    streamer, _, start_in_thinking, gen_thread, err_box, cancel_event = _prepare_qwen_streamer_generation(
        hf_messages=messages,
        pil_images=attached_images,
        hf_tools=None,
        enable_thinking=think_on,
        max_new_tokens=max_toks,
        temperature=temp,
        top_p=eff_top_p,
        preferred_model_id=bundle.get("model_id"),
        top_k=eff_top_k,
        min_p=eff_min_p,
        presence_penalty=eff_presence,
        repeat_penalty=eff_repeat,
        reasoning_effort=eff_effort,
    )

    parser = _IncrementalQwenStreamParser(start_in_thinking=start_in_thinking)
    think_parts: List[str] = []
    ans_parts: List[str] = []
    token_chunks = 0
    last_ui_sync = 0.0

    try:
        for piece in streamer:
            if job_cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                cancel_event.set()
                break
            token_chunks += 1
            for kind, val in parser.feed(piece):
                if kind == "thinking":
                    think_parts.append(val)
                elif kind == "text":
                    ans_parts.append(val)
            now_ts = time.time()
            # 按 ~10Hz 节流拼接流式字符串快照，避免 100+ tok/s 高速解码下逐 token O(N^2) 字符串拷贝争抢 GIL
            if token_chunks <= 2 or (now_ts - last_ui_sync) >= 0.10:
                last_ui_sync = now_ts
                think_txt = "".join(think_parts).strip()
                ans_txt = "".join(ans_parts).strip()
                JOBS[job_id]["thinking_output"] = think_txt
                JOBS[job_id]["text_output"] = ans_txt
                JOBS[job_id]["progress"] = min(0.95, round(0.15 + 0.80 * min(1.0, token_chunks / float(max(128.0, max_toks * 0.5))), 3))
                elapsed_now = round(now_ts - start_time, 1)
                JOBS[job_id]["elapsed"] = elapsed_now
                tps = streamer.tok_per_sec
                speed_tag = f" · {tps} tok/s" if tps > 0 else ""
                JOBS[job_id]["message"] = (
                    f"{model_label} 正在深度思考 ({elapsed_now}s{speed_tag})..."
                    if (think_txt and not ans_txt)
                    else f"{model_label} 正在流式输出回复 ({elapsed_now}s{speed_tag})..."
                )
                touch_activity()
    except Exception:
        cancel_event.set()
    finally:
        cancel_event.set()

    tail_events, _ = parser.finish()
    for kind, val in tail_events:
        if kind == "thinking":
            think_parts.append(val)
        elif kind == "text":
            ans_parts.append(val)

    gen_thread.join(timeout=10.0)
    if err_box[0] is not None:
        raise err_box[0]

    think_final = "".join(think_parts).strip()
    ans_final = "".join(ans_parts).strip()
    if req.enable_thinking and not think_final and "</think>" in ans_final:
        think_final, ans_final = _split_thinking_and_answer("<think>\n" + ans_final)

    if job_cancel_event.is_set() or (ABORT_GENERATION_EVENT.is_set() and not (think_final or ans_final)):
        raise GenerationAbortedError("已手动停止当前推理任务")

    elapsed = round(time.time() - start_time, 1)
    final_tps = streamer.tok_per_sec
    gen_toks = streamer.generated_tokens
    tps_summary = f" · 共 {gen_toks} tokens · {final_tps} tok/s" if gen_toks > 0 and final_tps > 0 else ""
    JOBS[job_id]["thinking_output"] = think_final
    JOBS[job_id]["text_output"] = ans_final
    JOBS[job_id]["status"] = "completed"
    JOBS[job_id]["progress"] = 1.0
    JOBS[job_id]["elapsed"] = elapsed
    JOBS[job_id]["message"] = f"{model_label} 推理完成！耗时 {elapsed}s{tps_summary}"
    print(f"[{model_label}] 推理完成 | {gen_toks} tokens | {final_tps} tok/s | 耗时 {elapsed}s")


def _fmt_mmss(sec: float) -> str:
    sec = max(0, int(round(sec)))
    m, s = divmod(sec, 60)
    return f"{m:02d}:{s:02d}"


from contextlib import contextmanager


@contextmanager
def _pipeline_job_execution_guard(
    pipe,
    job_id: str,
    job_cancel_event: threading.Event,
    start_time: float,
    base_prog: float,
    span_prog: float,
    prefix: str,
    model_name: str,
    media_type: str,
):
    """
    为所有 DiffSynth 视频/生图管线注入全链路真实进度与毫秒级 GPU 中断守卫：
    1. 挂载层块级 forward_pre_hook：点击停止推理后在当前 Transformer Block 结束时 (<0.1s) 立即抛出 GenerationAbortedError，
       瞬时释放 GPU_LOCK，彻底杜绝僵尸任务阻塞新任务。
    2. 接管 progress_bar_cmd 与阶段切换 load_models_to_device：
       - 0.04 ~ 0.08: 多模态提示词编码与潜空间初始化
       - 0.08 ~ 0.82 (图像 0.90): DiT 逐步去噪采样（每步实时刷新百分比与剩余时间，单行刷新不堆叠）
       - 0.84 ~ 0.93: 3D 视频 VAE 与 32kHz 立体声 VAE 解码（明确区分采样完成与后处理解码，绝不提前报 100%）
       - 0.94 ~ 0.99: MP4 音轨合成与落盘
    """
    global CURRENT_STAGE_BAR_LABEL
    import types

    dit_end = 0.90 if media_type == "image" else 0.82
    hook_handles = []
    orig_load_fn = getattr(pipe, "load_models_to_device", None)

    def _check_abort():
        if job_cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
            raise GenerationAbortedError("已手动停止当前推理任务")

    def _abort_pre_hook(_mod, _inp):
        if job_cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
            raise GenerationAbortedError("已手动停止当前推理任务")

    # 为管线主干及内部 Transformer Blocks 挂载轻量级中断钩子，使长耗时单步也能在 ~100ms 内立刻中止
    try:
        seen_mods = set()
        for _, child in pipe.named_children():
            if child is None or not isinstance(child, torch.nn.Module):
                continue
            if id(child) not in seen_mods:
                seen_mods.add(id(child))
                hook_handles.append(child.register_forward_pre_hook(_abort_pre_hook))
            for sub_name, sub_mod in child.named_modules():
                if isinstance(sub_mod, torch.nn.ModuleList) and any(
                    k in sub_name for k in ("blocks", "layers")
                ):
                    for blk in sub_mod:
                        if isinstance(blk, torch.nn.Module) and id(blk) not in seen_mods:
                            seen_mods.add(id(blk))
                            hook_handles.append(blk.register_forward_pre_hook(_abort_pre_hook))
    except Exception:
        pass

    if orig_load_fn is not None:
        def _stage_aware_load_models(self_pipe, model_names):
            global CURRENT_STAGE_BAR_LABEL
            _check_abort()
            names_set = set(model_names or ())
            if "text_encoder" in names_set:
                CURRENT_STAGE_BAR_LABEL = "多模态提示词编码"
                if job_id in JOBS:
                    JOBS[job_id]["progress"] = round(base_prog + 0.05 * span_prog, 3)
                    JOBS[job_id]["elapsed"] = round(time.time() - start_time, 1)
                    JOBS[job_id]["message"] = f"{prefix}{model_name} 正在使用多模态编码器解析提示词 (5%)..."
                with CONSOLE_LOCK:
                    ACTIVE_TQDM_BARS["sampling"] = "提示词编码:   5%|#---------| 正在解析画面、运镜与声音提示词..."
            elif "video_vae" in names_set or "vae" in names_set:
                v_frac = 0.92 if media_type == "image" else 0.85
                lbl = "高清图像 VAE 解码" if media_type == "image" else "3D 视频 VAE 解码"
                CURRENT_STAGE_BAR_LABEL = lbl
                if job_id in JOBS:
                    JOBS[job_id]["progress"] = round(base_prog + v_frac * span_prog, 3)
                    JOBS[job_id]["elapsed"] = round(time.time() - start_time, 1)
                    JOBS[job_id]["message"] = f"{prefix}DiT 采样已完成，正在执行{lbl} ({int(v_frac * 100)}%)..."
                with CONSOLE_LOCK:
                    ACTIVE_TQDM_BARS["sampling"] = f"{lbl}: {int(v_frac * 100):3d}%|########--| 正在将潜空间张量解码为高清像素..."
            elif "audio_vae" in names_set:
                CURRENT_STAGE_BAR_LABEL = "32kHz 立体声解码"
                if job_id in JOBS:
                    JOBS[job_id]["progress"] = round(base_prog + 0.92 * span_prog, 3)
                    JOBS[job_id]["elapsed"] = round(time.time() - start_time, 1)
                    JOBS[job_id]["message"] = f"{prefix}视频帧解码完成，正在执行 32kHz 立体声音轨解码 (92%)..."
                with CONSOLE_LOCK:
                    ACTIVE_TQDM_BARS["sampling"] = "32kHz 立体声解码:  92%|#########-| 正在解码双声道环境音与配乐波形..."
            res = orig_load_fn(model_names)
            _check_abort()
            touch_activity()
            return res

        pipe.load_models_to_device = types.MethodType(_stage_aware_load_models, pipe)

    def _job_progress_bar(iterable, *_args, **_kwargs):
        global CURRENT_STAGE_BAR_LABEL
        _check_abort()
        items = list(iterable)
        total = max(1, len(items))
        loop_t0 = time.time()
        CURRENT_STAGE_BAR_LABEL = None

        if job_id in JOBS:
            JOBS[job_id]["progress"] = round(base_prog + 0.08 * span_prog, 3)
            JOBS[job_id]["elapsed"] = round(time.time() - start_time, 1)
            JOBS[job_id]["message"] = f"{prefix}{model_name} 正在执行去噪采样第 1/{total} 步 (8%)..."
        with CONSOLE_LOCK:
            ACTIVE_TQDM_BARS["sampling"] = f"DiT 去噪采样:   0%|----------| 0/{total} [正在执行第 1/{total} 步...]"

        for idx, item in enumerate(items):
            _check_abort()
            yield item
            _check_abort()
            done = idx + 1
            step_frac = done / float(total)
            dt = max(0.01, time.time() - loop_t0)
            avg_s = dt / float(done)
            eta_s = max(0.0, avg_s * (total - done))
            overall_frac = 0.08 + (dit_end - 0.08) * step_frac
            fill = max(0, min(10, int(round(10 * step_frac))))
            touch_activity()

            if done < total:
                if job_id in JOBS:
                    JOBS[job_id]["progress"] = round(base_prog + overall_frac * span_prog, 3)
                    JOBS[job_id]["elapsed"] = round(time.time() - start_time, 1)
                    JOBS[job_id]["message"] = (
                        f"{prefix}{model_name} 正在采样 {done}/{total} 步 "
                        f"(总进度 {int(overall_frac * 100)}% · {avg_s:.1f}s/步 · 采样剩余约 {int(round(eta_s))}s)..."
                    )
                with CONSOLE_LOCK:
                    ACTIVE_TQDM_BARS["sampling"] = (
                        f"DiT 去噪采样: {int(step_frac * 100):3d}%|{'#' * fill}{'-' * (10 - fill)}| "
                        f"{done}/{total} [{_fmt_mmss(dt)}<{_fmt_mmss(eta_s)}, {avg_s:.2f}s/step] · 总进度 {int(overall_frac * 100)}%"
                    )
            else:
                post_frac = dit_end + 0.02
                next_lbl = "高清图像 VAE 解码" if media_type == "image" else "3D 视频与音频 VAE 解码"
                CURRENT_STAGE_BAR_LABEL = next_lbl
                if job_id in JOBS:
                    JOBS[job_id]["progress"] = round(base_prog + post_frac * span_prog, 3)
                    JOBS[job_id]["elapsed"] = round(time.time() - start_time, 1)
                    JOBS[job_id]["message"] = (
                        f"{prefix}DiT 采样已完成 ({total}/{total})，正在进入{next_lbl} ({int(post_frac * 100)}%)..."
                    )
                with CONSOLE_LOCK:
                    ACTIVE_TQDM_BARS["sampling"] = (
                        f"{next_lbl}: {int(post_frac * 100):3d}%|########--| "
                        f"DiT 采样已完成 ({total}/{total}，耗时 {_fmt_mmss(dt)}) -> 正在解码输出..."
                    )
                print(f"[{model_name}] DiT 采样 {total}/{total} 步完成 ({dt:.1f}s)，开始执行{next_lbl}...")

    try:
        yield _job_progress_bar
    finally:
        CURRENT_STAGE_BAR_LABEL = None
        if orig_load_fn is not None:
            pipe.load_models_to_device = orig_load_fn
        for h in hook_handles:
            try:
                h.remove()
            except Exception:
                pass


def _run_async_generation_job(job_id: str, req: GenerateVideoRequest):
    touch_activity()
    job_cancel_event = _JOB_CANCEL_EVENTS.setdefault(job_id, threading.Event())
    active_model_id = (
        req.model_id
        if (req.model_id and req.model_id in MODEL_CATALOG)
        else (SELECTED_MODEL_ID if SELECTED_MODEL_ID in LOADED_RUNTIMES else (LOADED_MODEL_ID or SELECTED_MODEL_ID))
    )
    batch_count = max(1, min(8, int(getattr(req, "batch_count", 1) or 1)))

    JOBS[job_id]["model_id"] = active_model_id
    JOBS[job_id]["status"] = "running"
    JOBS[job_id]["progress"] = 0.01
    JOBS[job_id]["batch_count"] = batch_count
    JOBS[job_id]["completed_count"] = 0
    JOBS[job_id]["completed_videos"] = []
    start_time = time.time()
    JOBS[job_id]["started_at"] = start_time

    try:
        if job_cancel_event.is_set():
            raise GenerationAbortedError("已手动停止当前推理任务")

        # 分支 A：Qwen3.8-27B / Qwen3.8-27B-GGUF 视觉理解与深度推理对话
        if _is_llm_model_id(active_model_id):
            _run_qwen_llm_job(job_id, req, target_llm_id=active_model_id)
            return

        rt = LOADED_RUNTIMES.get(active_model_id)
        active_pipe = rt.get("pipeline") if rt else (PIPELINE if LOADED_MODEL_ID == active_model_id else None)
        if active_pipe is None:
            req_name = MODEL_CATALOG.get(active_model_id, {}).get("name", active_model_id)
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["message"] = f"{req_name} 尚未加载到 GPU 显存！请先在左侧点击「加载到显存」（显存充裕时可与现有模型同时常驻）。"
            return

        spec = MODEL_CATALOG.get(active_model_id, MODEL_CATALOG["minimax_h3"])
        modality = spec["modality"]
        media_type = "image" if modality == "image" else "video"
        JOBS[job_id]["media_type"] = media_type
        JOBS[job_id]["progress"] = 0.02
        JOBS[job_id]["message"] = (
            f"正在准备 {spec['name']} 推理管线 (共 {batch_count} 项)..."
            if batch_count > 1
            else f"正在准备 {spec['name']} 推理管线..."
        )

        base_seed = req.seed
        if req.randomize_seed or base_seed < 0:
            base_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        JOBS[job_id]["seed"] = base_seed

        first_img = decode_base64_image(req.first_frame_base64)
        last_img = decode_base64_image(req.last_frame_base64)

        keyframes = []
        keyframe_indices = []
        if active_model_id == "minimax_h3" and req.task_mode == "fl2va":
            if first_img is not None:
                keyframes.append(first_img)
                keyframe_indices.append(0)
            if last_img is not None:
                keyframes.append(last_img)
                keyframe_indices.append(-1)
            if not keyframes:
                JOBS[job_id]["status"] = "failed"
                JOBS[job_id]["message"] = "首尾帧模式下请至少提供首帧或尾帧图片"
                return
        elif modality == "image" and req.task_mode == "i2i_edit" and first_img is None and last_img is None:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["message"] = "图像编辑/重绘模式下请至少拖入一张输入图片"
            return

        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        last_filename = None
        last_seed = base_seed

        for batch_idx in range(batch_count):
            if job_cancel_event.is_set():
                raise GenerationAbortedError("已手动停止当前推理任务")

            if batch_idx == 0:
                current_seed = base_seed
            elif req.randomize_seed:
                current_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
            else:
                current_seed = base_seed + batch_idx

            last_seed = current_seed
            JOBS[job_id]["seed"] = current_seed
            prefix = f"[{batch_idx + 1}/{batch_count}] " if batch_count > 1 else ""
            base_prog = batch_idx / float(batch_count)
            span_prog = 1.0 / float(batch_count)

            if GPU_LOCK.locked():
                JOBS[job_id]["progress"] = round(base_prog + 0.01 * span_prog, 3)
                JOBS[job_id]["message"] = f"{prefix}正在等待上一任务释放 GPU 显存锁..."

            with GPU_LOCK:
                if job_cancel_event.is_set():
                    raise GenerationAbortedError("已手动停止当前推理任务")
                ABORT_GENERATION_EVENT.clear()

                # 获取 GPU 锁后重置本轮起始计时并清空所有旧进度条，确保新任务严格从 0% 与 0.0s 起步
                sub_start = time.time()
                if batch_idx == 0:
                    start_time = sub_start
                    JOBS[job_id]["started_at"] = start_time
                    JOBS[job_id]["elapsed"] = 0.0
                with CONSOLE_LOCK:
                    ACTIVE_TQDM_BARS.clear()

                JOBS[job_id]["progress"] = round(base_prog + 0.04 * span_prog, 3)
                JOBS[job_id]["message"] = (
                    f"{prefix}{spec['name']} 正在初始化潜空间并准备 {req.num_inference_steps} 步推理 (Seed: {current_seed})..."
                )
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                touch_activity()
                _reclaim_transient_gpu_memory()

                with _pipeline_job_execution_guard(
                    pipe=active_pipe,
                    job_id=job_id,
                    job_cancel_event=job_cancel_event,
                    start_time=start_time,
                    base_prog=base_prog,
                    span_prog=span_prog,
                    prefix=prefix,
                    model_name=spec["name"],
                    media_type=media_type,
                ) as step_progress_bar:
                    if active_model_id == "minimax_h3":
                        from diffsynth.utils.data.audio_video import write_video_audio
                        full_prompt = build_structured_prompt(
                            req.visual_prompt, req.soundscape_prompt, req.music_prompt
                        )
                        filename = f"minimax_h3_{req.task_mode}_{timestamp}_seed{current_seed}.mp4"
                        output_path = os.path.join(DEFAULT_OUTPUT_DIR, filename)
                        call_kwargs = dict(
                            prompt=full_prompt,
                            height=int(req.height),
                            width=int(req.width),
                            num_frames=int(req.num_frames),
                            num_inference_steps=int(req.num_inference_steps),
                            seed=int(current_seed),
                            progress_bar_cmd=step_progress_bar,
                        )
                        if keyframes:
                            call_kwargs["keyframes"] = keyframes
                            call_kwargs["keyframe_indices"] = keyframe_indices
                        video, audio = _run_pipeline_with_smart_tiling(active_pipe, call_kwargs)
                        if job_cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                            raise GenerationAbortedError("已手动停止当前推理任务")
                        JOBS[job_id]["progress"] = round(base_prog + 0.95 * span_prog, 3)
                        JOBS[job_id]["elapsed"] = round(time.time() - start_time, 1)
                        JOBS[job_id]["message"] = f"{prefix}VAE 解码完成，正在合成 32kHz 立体声音轨并封装 MP4 (95%)..."
                        with CONSOLE_LOCK:
                            ACTIVE_TQDM_BARS["sampling"] = "音视频封装:  95%|#########-| 正在编码 H.264 视频与 32kHz 立体声 MP4..."
                        write_video_audio(
                            video=video,
                            audio=audio,
                            output_path=output_path,
                            fps=int(req.fps),
                            audio_sample_rate=32000,
                        )
                        video = None
                        audio = None

                    elif active_model_id == "qwen_image_21":
                        filename = f"qwen_image21_{req.task_mode}_{timestamp}_seed{current_seed}.png"
                        output_path = os.path.join(DEFAULT_OUTPUT_DIR, filename)
                        edit_imgs = [im for im in (first_img, last_img) if im is not None] if req.task_mode == "i2i_edit" else None
                        img_out = active_pipe(
                            prompt=req.visual_prompt,
                            negative_prompt=req.negative_prompt.strip() or " ",
                            edit_image=edit_imgs if edit_imgs else None,
                            height=int(req.height),
                            width=int(req.width),
                            num_inference_steps=int(req.num_inference_steps),
                            seed=int(current_seed),
                            progress_bar_cmd=step_progress_bar,
                        )
                        if job_cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                            raise GenerationAbortedError("已手动停止当前推理任务")
                        JOBS[job_id]["progress"] = round(base_prog + 0.96 * span_prog, 3)
                        JOBS[job_id]["message"] = f"{prefix}正在写入无损 PNG 图像文件 (96%)..."
                        img_out.save(output_path)
                        img_out = None
                        edit_imgs = None

                    elif active_model_id == "z_image_turbo":
                        filename = f"z_image_turbo_{req.task_mode}_{timestamp}_seed{current_seed}.png"
                        output_path = os.path.join(DEFAULT_OUTPUT_DIR, filename)
                        w, h = int(req.width), int(req.height)
                        call_kw = dict(
                            prompt=req.visual_prompt,
                            negative_prompt=req.negative_prompt or "",
                            height=h,
                            width=w,
                            num_inference_steps=int(req.num_inference_steps),
                            seed=int(current_seed),
                            rand_device="cuda",
                            progress_bar_cmd=step_progress_bar,
                        )
                        if req.task_mode == "i2i_edit" and first_img is not None:
                            call_kw["input_image"] = first_img.resize((w, h))
                            call_kw["denoising_strength"] = max(0.15, min(1.0, float(req.denoising_strength or 0.70)))
                        img_out = active_pipe(**call_kw)
                        if job_cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                            raise GenerationAbortedError("已手动停止当前推理任务")
                        JOBS[job_id]["progress"] = round(base_prog + 0.96 * span_prog, 3)
                        JOBS[job_id]["message"] = f"{prefix}正在写入无损 PNG 图像文件 (96%)..."
                        img_out.save(output_path)
                        img_out = None
                        call_kw.clear()

                    else:
                        raise ValueError(f"未知模型 ID: {active_model_id}")

                if not os.path.exists(output_path) or os.path.getsize(output_path) < 256:
                    raise RuntimeError("输出媒体文件未成功生成或写入为空")

                with CONSOLE_LOCK:
                    ACTIVE_TQDM_BARS.clear()
                _reclaim_transient_gpu_memory()

            sub_elapsed = round(time.time() - sub_start, 1)
            last_filename = filename
            kind_cn = "图像" if media_type == "image" else "视频"
            print(f"[{spec['name']}] {kind_cn}生成完成 | 耗时 {sub_elapsed}s | Seed: {current_seed} | 文件: {filename}")
            media_item = {
                "index": batch_idx + 1,
                "seed": current_seed,
                "filename": filename,
                "video_url": f"/minimax_api/video/{filename}",
                "media_type": media_type,
                "elapsed": sub_elapsed,
            }
            JOBS[job_id]["completed_videos"] = list(JOBS[job_id]["completed_videos"]) + [media_item]
            JOBS[job_id]["completed_count"] = batch_idx + 1
            JOBS[job_id]["filename"] = filename
            JOBS[job_id]["video_url"] = f"/minimax_api/video/{filename}"
            JOBS[job_id]["elapsed"] = round(time.time() - start_time, 1)
            if batch_count > 1 and batch_idx + 1 < batch_count:
                JOBS[job_id]["progress"] = round((batch_idx + 1) / float(batch_count), 3)
                JOBS[job_id]["message"] = (
                    f"第 {batch_idx + 1}/{batch_count} 项已完成 ({sub_elapsed}s)，已同步回传 Mac！继续生成第 {batch_idx + 2} 项..."
                )
            touch_activity()

        elapsed = round(time.time() - start_time, 1)
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["progress"] = 1.0
        JOBS[job_id]["elapsed"] = elapsed
        JOBS[job_id]["filename"] = last_filename
        JOBS[job_id]["video_url"] = f"/minimax_api/video/{last_filename}" if last_filename else None
        if batch_count > 1:
            JOBS[job_id]["message"] = (
                f"批量生成 {batch_count} 项全部完成！总耗时 {elapsed}s (平均 {round(elapsed / batch_count, 1)}s/项)"
            )
        else:
            JOBS[job_id]["message"] = f"生成完成！耗时 {elapsed}s (Seed: {last_seed})"

    except GenerationAbortedError as abort_err:
        with CONSOLE_LOCK:
            ACTIVE_TQDM_BARS.clear()
        _reclaim_transient_gpu_memory()
        if job_id in JOBS:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["progress"] = 0.0
            JOBS[job_id]["message"] = str(abort_err) or "已手动停止当前推理任务"
        print(f"[任务控制] 任务 {job_id} 已中断并释放 GPU 显存锁。")
    except Exception as e:
        with CONSOLE_LOCK:
            ACTIVE_TQDM_BARS.clear()
        _reclaim_transient_gpu_memory()
        if job_id in JOBS:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["message"] = f"生成出错: {str(e)}"
    finally:
        _JOB_CANCEL_EVENTS.pop(job_id, None)
        touch_activity()


class ApiKeyConfigRequest(BaseModel):
    api_key: str = ""


class UnloadModelRequest(BaseModel):
    model_id: str = ""


class HFFilesQuery(BaseModel):
    repo_id: str
    hf_token: str = ""


class RegisterCustomModelRequest(BaseModel):
    repo_id: str
    filename: str
    name: str = ""
    mmproj_file: str = ""
    vram_gb: float = 0.0
    approx_size_gb: float = 0.0
    context_length: int = 32768


class RemoveCustomModelRequest(BaseModel):
    model_id: str


# ==============================================================================
# 标准对外网关：OpenAI (/v1/chat/completions, /v1/models, /v1/images/generations)
#            与 Anthropic (/v1/messages, /v1/messages/count_tokens) 双协议兼容引擎
# ==============================================================================
def _verify_external_api_auth(request: Request, protocol: str = "openai") -> Optional[JSONResponse]:
    """同时支持 Authorization: Bearer <key> 与 x-api-key: <key> 鉴权"""
    expected = (API_AUTH_KEY or "").strip()
    if not expected:
        return None
    auth_hdr = (request.headers.get("authorization") or "").strip()
    bearer_tok = auth_hdr[7:].strip() if auth_hdr.lower().startswith("bearer ") else ""
    x_api_tok = (request.headers.get("x-api-key") or "").strip()
    provided = bearer_tok or x_api_tok
    if provided == expected:
        return None
    if protocol == "anthropic":
        return JSONResponse(
            status_code=401,
            content={"type": "error", "error": {"type": "authentication_error", "message": "Invalid API Key"}},
        )
    return JSONResponse(
        status_code=401,
        content={"error": {"message": "Invalid API Key", "type": "invalid_request_error", "code": "invalid_api_key"}},
    )


def _load_pil_from_url_or_b64(src: str) -> Optional[Image.Image]:
    if not src:
        return None
    src = src.strip()
    if src.startswith("http://") or src.startswith("https://"):
        req = urllib.request.Request(src, headers={"User-Agent": "Qwen3.8-API-Gateway/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")
    return decode_base64_image(src)


def _extract_qwen_tool_calls(raw_text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """解析 Qwen3 / Qwen3.8 输出中的 <tool_call>...</tool_call> (兼容 JSON 与 XML 参数格式)"""
    if "<tool_call>" not in raw_text:
        return raw_text, []

    tool_calls: List[Dict[str, Any]] = []
    tc_pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

    for match in tc_pattern.finditer(raw_text):
        inner = match.group(1).strip()
        call_id = f"toolu_{uuid.uuid4().hex[:16]}"
        parsed_name = None
        parsed_args: Dict[str, Any] = {}

        # 格式 1: JSON {"name": "...", "arguments": {...}}
        try:
            obj = json.loads(inner)
            if isinstance(obj, dict) and "name" in obj:
                parsed_name = str(obj["name"])
                raw_args = obj.get("arguments", obj.get("parameters", {}))
                if isinstance(raw_args, str):
                    try:
                        parsed_args = json.loads(raw_args)
                    except Exception:
                        parsed_args = {"raw": raw_args}
                elif isinstance(raw_args, dict):
                    parsed_args = raw_args
        except Exception:
            pass

        # 格式 2: Qwen3.5/3.8 XML <function=fn_name><parameter=k>v</parameter></function>
        if not parsed_name:
            fn_m = re.search(r"<function=([^>\s]+)>\s*(.*?)\s*</function>", inner, re.DOTALL)
            if fn_m:
                parsed_name = fn_m.group(1).strip()
                body = fn_m.group(2)
                for pm in re.finditer(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", body, re.DOTALL):
                    p_key = pm.group(1).strip()
                    p_val_raw = pm.group(2).strip()
                    try:
                        parsed_args[p_key] = json.loads(p_val_raw)
                    except Exception:
                        parsed_args[p_key] = p_val_raw

        if parsed_name:
            tool_calls.append({"id": call_id, "name": parsed_name, "arguments": parsed_args})

    cleaned_text = tc_pattern.sub("", raw_text).strip()
    return cleaned_text, tool_calls


class _IncrementalQwenStreamParser:
    """流式增量拆分 <think>...</think>、正文与 <tool_call>...</tool_call>，杜绝半截标签外泄"""

    def __init__(self, start_in_thinking: bool):
        self.in_thinking = start_in_thinking
        self.in_tool_call = False
        self.buf = ""
        self.tool_raw = ""

    def feed(self, piece: str) -> List[Tuple[str, str]]:
        events: List[Tuple[str, str]] = []
        self.buf += piece

        while self.buf:
            if self.in_tool_call:
                self.tool_raw += self.buf
                self.buf = ""
                break

            if self.in_thinking:
                # 忽略开头的显式 <think>
                if self.buf.lstrip().startswith("<think>"):
                    idx = self.buf.find("<think>")
                    self.buf = self.buf[idx + len("<think>"):]
                    continue
                end_idx = self.buf.find("</think>")
                if end_idx != -1:
                    before = self.buf[:end_idx]
                    if before:
                        events.append(("thinking", before))
                    self.buf = self.buf[end_idx + len("</think>"):].lstrip("\n")
                    self.in_thinking = False
                    continue
                # 保留末尾可能未闭合的 '<' 前缀
                lt = self.buf.rfind("<")
                if lt != -1 and len(self.buf) - lt < 12 and "</think>".startswith(self.buf[lt:]):
                    safe = self.buf[:lt]
                    if safe:
                        events.append(("thinking", safe))
                    self.buf = self.buf[lt:]
                    break
                events.append(("thinking", self.buf))
                self.buf = ""
            else:
                if self.buf.lstrip().startswith("<think>"):
                    idx = self.buf.find("<think>")
                    before = self.buf[:idx]
                    if before:
                        events.append(("text", before))
                    self.buf = self.buf[idx + len("<think>"):]
                    self.in_thinking = True
                    continue

                tc_idx = self.buf.find("<tool_call>")
                if tc_idx != -1:
                    before = self.buf[:tc_idx]
                    if before:
                        events.append(("text", before))
                    self.tool_raw += self.buf[tc_idx:]
                    self.buf = ""
                    self.in_tool_call = True
                    break

                lt = self.buf.rfind("<")
                if lt != -1 and len(self.buf) - lt < 14 and (
                    "<tool_call>".startswith(self.buf[lt:]) or "<think>".startswith(self.buf[lt:])
                ):
                    safe = self.buf[:lt]
                    if safe:
                        events.append(("text", safe))
                    self.buf = self.buf[lt:]
                    break

                events.append(("text", self.buf))
                self.buf = ""

        return events

    def finish(self) -> Tuple[List[Tuple[str, str]], List[Dict[str, Any]]]:
        events: List[Tuple[str, str]] = []
        if self.buf:
            if self.in_tool_call:
                self.tool_raw += self.buf
            elif self.in_thinking:
                events.append(("thinking", self.buf))
            else:
                events.append(("text", self.buf))
            self.buf = ""

        _, tool_calls = _extract_qwen_tool_calls(self.tool_raw) if self.tool_raw else ("", [])
        return events, tool_calls


def _normalize_to_hf_messages_and_images(
    raw_messages: List[Dict[str, Any]],
    system_prompt: Any = None,
    protocol: str = "openai",
) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
    """将 OpenAI 或 Anthropic 消息数组统一转换为 Qwen3.8 Processor 标准多模态结构"""
    hf_messages: List[Dict[str, Any]] = []
    pil_images: List[Image.Image] = []

    if system_prompt:
        if isinstance(system_prompt, str) and system_prompt.strip():
            hf_messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt.strip()}]})
        elif isinstance(system_prompt, list):
            sys_txt = "\n".join(
                b.get("text", "") for b in system_prompt if isinstance(b, dict) and b.get("type") == "text"
            ).strip()
            if sys_txt:
                hf_messages.append({"role": "system", "content": [{"type": "text", "text": sys_txt}]})

    for msg in raw_messages or []:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "tool":
            txt = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            hf_messages.append({"role": "tool", "content": txt})
            continue

        blocks: List[Dict[str, Any]] = []
        if isinstance(content, str):
            blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type", "")
                if ptype == "text":
                    blocks.append({"type": "text", "text": str(part.get("text", ""))})
                elif ptype == "image_url":
                    img_obj = part.get("image_url", {})
                    url_str = img_obj.get("url", "") if isinstance(img_obj, dict) else str(img_obj)
                    im = _load_pil_from_url_or_b64(url_str)
                    if im is not None:
                        pil_images.append(im)
                        blocks.append({"type": "image", "image": im})
                elif ptype == "image":
                    src = part.get("source", {})
                    im = None
                    if isinstance(src, dict):
                        if src.get("type") == "base64":
                            im = decode_base64_image(src.get("data", ""))
                        elif src.get("type") == "url":
                            im = _load_pil_from_url_or_b64(src.get("url", ""))
                    if im is not None:
                        pil_images.append(im)
                        blocks.append({"type": "image", "image": im})
                elif ptype == "tool_use":
                    tc_str = json.dumps({"name": part.get("name", ""), "arguments": part.get("input", {})}, ensure_ascii=False)
                    blocks.append({"type": "text", "text": f"<tool_call>\n{tc_str}\n</tool_call>"})
                elif ptype == "tool_result":
                    tr_content = part.get("content", "")
                    if isinstance(tr_content, list):
                        tr_txt = "\n".join(x.get("text", "") for x in tr_content if isinstance(x, dict))
                    else:
                        tr_txt = str(tr_content)
                    hf_messages.append({"role": "tool", "content": tr_txt})

        # OpenAI assistant historical tool_calls
        if role == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                raw_a = fn.get("arguments", "{}")
                try:
                    a_obj = json.loads(raw_a) if isinstance(raw_a, str) else raw_a
                except Exception:
                    a_obj = {}
                tc_json = json.dumps({"name": fn.get("name", ""), "arguments": a_obj}, ensure_ascii=False)
                blocks.append({"type": "text", "text": f"<tool_call>\n{tc_json}\n</tool_call>"})

        if blocks:
            hf_messages.append({"role": role, "content": blocks})

    return hf_messages, pil_images


def _normalize_hf_tools(raw_tools: Optional[List[Dict[str, Any]]], protocol: str) -> Optional[List[Dict[str, Any]]]:
    if not raw_tools:
        return None
    hf_tools: List[Dict[str, Any]] = []
    for t in raw_tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            hf_tools.append(t)
        elif "name" in t:
            # Anthropic tool schema -> HF OpenAI function tool schema
            hf_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            })
    return hf_tools if hf_tools else None


class _GGUFStreamIterator:
    """与 _TimedStreamer 接口一致的 GGUF 流式迭代器，统一供原生客户端与 OpenAI/Anthropic 网关消费"""

    def __init__(self, est_prompt_tokens: int = 1):
        self._q: "queue.Queue[Optional[str]]" = queue.Queue()
        self.prompt_tokens: int = max(1, int(est_prompt_tokens))
        self.generated_tokens: int = 0
        self.first_token_ts: Optional[float] = None
        self._server_tps: float = 0.0

    def put_text(self, text: str, token_inc: int = 1):
        if not text:
            return
        now = time.time()
        if self.first_token_ts is None:
            self.first_token_ts = now
        if token_inc > 0:
            self.generated_tokens += token_inc
        self._q.put(text)

    def end(self):
        self._q.put(None)

    @property
    def tok_per_sec(self) -> float:
        if self._server_tps > 0:
            return round(self._server_tps, 1)
        if self.first_token_ts is None or self.generated_tokens <= 1:
            return 0.0
        dt = max(0.05, time.time() - self.first_token_ts)
        return round((self.generated_tokens - 1) / dt, 1)

    def __iter__(self):
        return self

    def __next__(self) -> str:
        item = self._q.get()
        if item is None:
            raise StopIteration
        return item


def _pil_to_jpeg_data_url(im: Image.Image) -> str:
    rgb = im.convert("RGB")
    max_side = max(rgb.size)
    if max_side > 2048:
        scale = 2048.0 / float(max_side)
        rgb = rgb.resize(
            (max(64, int(rgb.size[0] * scale)), max(64, int(rgb.size[1] * scale))),
            Image.Resampling.LANCZOS,
        )
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _hf_messages_to_gguf_openai_messages(hf_messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    oai_msgs: List[Dict[str, Any]] = []
    est_chars = 0
    est_imgs = 0

    for msg in hf_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "tool":
            s_txt = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            est_chars += len(s_txt)
            oai_msgs.append({"role": "tool", "content": s_txt, "tool_call_id": msg.get("tool_call_id", "call_0")})
            continue

        if isinstance(content, str):
            est_chars += len(content)
            oai_msgs.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            has_img = any(isinstance(b, dict) and b.get("type") == "image" and b.get("image") is not None for b in content)
            if not has_img:
                txt = "".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
                est_chars += len(txt)
                oai_msgs.append({"role": role, "content": txt})
            else:
                parts: List[Dict[str, Any]] = []
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "image" and b.get("image") is not None:
                        est_imgs += 1
                        parts.append({"type": "image_url", "image_url": {"url": _pil_to_jpeg_data_url(b["image"])}})
                    elif b.get("type") == "text":
                        t_str = str(b.get("text", ""))
                        est_chars += len(t_str)
                        parts.append({"type": "text", "text": t_str})
                oai_msgs.append({"role": role, "content": parts})

    est_tokens = max(1, est_chars // 3 + est_imgs * 512)
    return oai_msgs, est_tokens


def _prepare_gguf_streamer_generation(
    bundle: Dict[str, Any],
    hf_messages: List[Dict[str, Any]],
    hf_tools: Optional[List[Dict[str, Any]]],
    enable_thinking: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int = 20,
    min_p: float = 0.0,
    presence_penalty: Optional[float] = None,
    repeat_penalty: float = 1.0,
    reasoning_effort: Optional[str] = None,
):
    """通过驻留 GPU 的 llama-server 流式执行 Qwen3.8-27B-UD-Q4_K_M 推理 (遵循 Unsloth 官方 GGUF 部署参数规范，支持最高 256K 上下文与视觉输入)"""
    oai_msgs, est_prompt_tokens = _hf_messages_to_gguf_openai_messages(hf_messages)
    bundle_ctx = int(bundle.get("context_length") or QWEN_CONTEXT_LENGTH)
    max_toks = max(16, min(65536, int(max_new_tokens or 16384), max(256, bundle_ctx - min(est_prompt_tokens, bundle_ctx - 256))))

    streamer = _GGUFStreamIterator(est_prompt_tokens=est_prompt_tokens)
    cancel_event = threading.Event()
    err_box: List[Optional[Exception]] = [None]

    eff_presence = (
        float(presence_penalty)
        if presence_penalty is not None
        else (0.0 if enable_thinking else 1.5)
    )
    # enable_thinking / reasoning_effort 是 Qwen 系模板专有变量；其他架构传了会让 jinja 模板渲染失败
    _label_low = f"{bundle.get('model_label') or ''} {bundle.get('architecture') or ''}".lower()
    _is_qwen_tpl = "qwen" in _label_low or bundle.get("model_id") in ("qwen38_27b_gguf", "qwen38_27b")
    tpl_kwargs: Dict[str, Any] = {}
    if _is_qwen_tpl:
        tpl_kwargs["enable_thinking"] = bool(enable_thinking)
        if enable_thinking and reasoning_effort and reasoning_effort not in ("default", "none"):
            tpl_kwargs["reasoning_effort"] = str(reasoning_effort)

    eff_temp = max(0.0, min(2.0, float(temperature)))
    eff_top_p_val = max(0.05, min(1.0, float(top_p)))
    eff_top_k_val = max(0, int(top_k))
    eff_min_p_val = max(0.0, min(1.0, float(min_p)))
    eff_pres_val = max(0.0, min(2.0, eff_presence))
    eff_rep_val = max(0.5, min(2.0, float(repeat_penalty)))

    # 针对 Qwen3.8 超大词表 (248,320) + MTP 多词元投机解码定制采样管线：
    # - 无惩罚项时 (Thinking 默认) 仅保留 GPU 原生采样器，100% 在 GPU 端完成采样，零 PCIe Logits 拷贝；
    # - 启用惩罚项时 (Non-Thinking presence_penalty=1.5) 先在 GPU 执行 top_k 将 248K 词表筛至 20 项，再计算 penalties。
    has_penalties = (eff_pres_val > 1e-5) or (abs(eff_rep_val - 1.0) > 1e-4)
    req_samplers: List[str] = []
    if eff_top_k_val > 0:
        req_samplers.append("top_k")
    if has_penalties:
        req_samplers.append("penalties")
    if eff_top_p_val < 0.9999:
        req_samplers.append("top_p")
    if eff_min_p_val > 1e-6:
        req_samplers.append("min_p")
    req_samplers.append("temperature")

    payload: Dict[str, Any] = {
        "model": bundle.get("model_label") or "Qwen3.8-27B-UD-Q4_K_M",
        "messages": oai_msgs,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_toks,
        "temperature": eff_temp,
        "top_p": eff_top_p_val,
        "top_k": eff_top_k_val,
        "min_p": eff_min_p_val,
        "presence_penalty": eff_pres_val,
        "repeat_penalty": eff_rep_val,
        "samplers": req_samplers,
    }
    if tpl_kwargs:
        payload["chat_template_kwargs"] = tpl_kwargs
    if hf_tools:
        payload["tools"] = hf_tools

    def _locked_gguf_generate():
        if cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
            streamer.end()
            return

        with GPU_LOCK:
            if cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                streamer.end()
                return

            srv_proc = bundle.get("server_proc")
            if srv_proc is not None and srv_proc.poll() is not None:
                err_box[0] = RuntimeError(
                    f"llama-server 子进程已意外退出 (code={srv_proc.returncode})，请在左侧重新点击「加载到显存」。"
                )
                streamer.end()
                return

            opener = _make_noproxy_opener()
            url = f"{bundle['base_url']}/v1/chat/completions"
            req_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            http_req = urllib.request.Request(
                url,
                data=req_bytes,
                headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
                method="POST",
            )

            in_reasoning = False
            tc_acc: Dict[int, Dict[str, str]] = {}
            resp = None
            raw_sock = None
            stop_watch = threading.Event()

            try:
                # 严禁将 raw_sock.settimeout 设为短秒数：Python 3.13 SocketIO 一旦触发单次超时会将 _timeout_occurred 永久置 True，
                # 导致后续 resp.readline() 立即抛出 OSError("cannot read from timed out object")。
                # 此处保持长超时 900s 以覆盖 256K 超长上下文预填充，并由独立守护线程监听中断信号关闭套接字。
                resp = opener.open(http_req, timeout=900.0)
                raw_sock = getattr(getattr(getattr(resp, "fp", None), "raw", None), "_sock", None)

                def _abort_watcher():
                    while not stop_watch.wait(0.25):
                        if cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                            if raw_sock is not None:
                                try:
                                    raw_sock.shutdown(socket.SHUT_RDWR)
                                except Exception:
                                    pass
                            if resp is not None:
                                try:
                                    resp.close()
                                except Exception:
                                    pass
                            break

                threading.Thread(target=_abort_watcher, daemon=True).start()

                while True:
                    if cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                        break
                    raw_line = resp.readline()
                    if not raw_line:
                        break
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                    except Exception:
                        continue

                    if isinstance(chunk.get("error"), dict):
                        raise RuntimeError(str(chunk["error"].get("message") or chunk["error"]))

                    usage = chunk.get("usage")
                    if isinstance(usage, dict):
                        if usage.get("prompt_tokens"):
                            streamer.prompt_tokens = int(usage["prompt_tokens"])
                        if usage.get("completion_tokens"):
                            streamer.generated_tokens = max(streamer.generated_tokens, int(usage["completion_tokens"]))

                    timings = chunk.get("timings")
                    if isinstance(timings, dict) and timings.get("predicted_per_second"):
                        streamer._server_tps = float(timings["predicted_per_second"])

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}

                    r_txt = delta.get("reasoning_content")
                    if r_txt:
                        if not in_reasoning:
                            streamer.put_text("<think>\n" + str(r_txt), token_inc=1)
                            in_reasoning = True
                        else:
                            streamer.put_text(str(r_txt), token_inc=1)

                    c_txt = delta.get("content")
                    if c_txt:
                        if in_reasoning:
                            streamer.put_text("</think>\n" + str(c_txt), token_inc=1)
                            in_reasoning = False
                        else:
                            streamer.put_text(str(c_txt), token_inc=1)

                    d_tcs = delta.get("tool_calls")
                    if isinstance(d_tcs, list):
                        for tc_item in d_tcs:
                            if not isinstance(tc_item, dict):
                                continue
                            idx = int(tc_item.get("index", 0))
                            slot = tc_acc.setdefault(idx, {"name": "", "arguments": ""})
                            fn_obj = tc_item.get("function") or {}
                            if fn_obj.get("name"):
                                slot["name"] += str(fn_obj["name"])
                            if fn_obj.get("arguments"):
                                slot["arguments"] += str(fn_obj["arguments"])

                if in_reasoning:
                    streamer.put_text("</think>\n", token_inc=0)
                    in_reasoning = False

                if tc_acc:
                    for idx in sorted(tc_acc.keys()):
                        slot = tc_acc[idx]
                        if not slot["name"]:
                            continue
                        try:
                            arg_val = json.loads(slot["arguments"]) if slot["arguments"] else {}
                        except Exception:
                            arg_val = {"raw": slot["arguments"]}
                        tc_str = json.dumps({"name": slot["name"], "arguments": arg_val}, ensure_ascii=False)
                        streamer.put_text(f"<tool_call>\n{tc_str}\n</tool_call>", token_inc=0)

            except urllib.error.HTTPError as he:
                err_body = ""
                try:
                    err_body = he.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                if he.code == 400 and ("context" in err_body.lower() or "exceed" in err_body.lower()):
                    err_box[0] = RuntimeError(
                        f"输入长度超出当前 GGUF 上下文窗口 ({bundle_ctx // 1024}K = {bundle_ctx} tokens)！"
                        f"请在左侧调大「上下文长度」（最高 256K）后点击重新加载模型。({err_body[:180]})"
                    )
                else:
                    err_box[0] = RuntimeError(f"GGUF 推理服务返回 HTTP {he.code}: {err_body[:240]}")
            except Exception as ex:
                if not (cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set()):
                    err_box[0] = ex
            finally:
                stop_watch.set()
                if resp is not None:
                    try:
                        resp.close()
                    except Exception:
                        pass
                streamer.end()
                touch_activity()

    t = threading.Thread(target=_locked_gguf_generate, daemon=True)
    t.start()
    return streamer, streamer.prompt_tokens, False, t, err_box, cancel_event


def _prepare_qwen_streamer_generation(
    hf_messages: List[Dict[str, Any]],
    pil_images: List[Image.Image],
    hf_tools: Optional[List[Dict[str, Any]]],
    enable_thinking: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    preferred_model_id: Optional[str] = None,
    top_k: int = 20,
    min_p: float = 0.0,
    presence_penalty: Optional[float] = None,
    repeat_penalty: float = 1.0,
    reasoning_effort: Optional[str] = None,
):
    """构建 Qwen3.8 (GGUF 或 BF16 Transformers) 后台流式推理任务并返回 (streamer, prompt_tokens, start_in_thinking, gen_thread, err_box, cancel_event)"""
    bundle = _get_qwen_llm_bundle(preferred_model_id=preferred_model_id)
    if bundle is not None and bundle.get("backend") == "gguf":
        return _prepare_gguf_streamer_generation(
            bundle=bundle,
            hf_messages=hf_messages,
            hf_tools=hf_tools,
            enable_thinking=enable_thinking,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            reasoning_effort=reasoning_effort,
        )

    model = bundle["model"]
    processor = bundle["processor"]
    bundle_ctx = int(bundle.get("context_length") or QWEN_CONTEXT_LENGTH)

    tpl_kwargs: Dict[str, Any] = dict(
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if enable_thinking and reasoning_effort and reasoning_effort not in ("default", "none"):
        tpl_kwargs["reasoning_effort"] = str(reasoning_effort)
    if hf_tools:
        tpl_kwargs["tools"] = hf_tools

    try:
        prompt_text = processor.apply_chat_template(hf_messages, **tpl_kwargs)
    except TypeError:
        tpl_kwargs.pop("reasoning_effort", None)
        try:
            prompt_text = processor.apply_chat_template(hf_messages, **tpl_kwargs)
        except TypeError:
            tpl_kwargs.pop("enable_thinking", None)
            prompt_text = processor.apply_chat_template(hf_messages, **tpl_kwargs)

    start_in_thinking = enable_thinking and (
        prompt_text.rstrip().endswith("<think>") or "<think>" in prompt_text[-24:]
    ) and not prompt_text.rstrip().endswith("</think>")

    prompt_tokens = max(1, len(processor.tokenizer.encode(prompt_text)))
    streamer = _make_timed_token_streamer(processor.tokenizer, timeout=180.0)

    max_toks = max(16, min(65536, int(max_new_tokens or 16384), max(128, bundle_ctx - prompt_tokens)))
    do_sample = temperature > 0.01
    cancel_event = threading.Event()
    err_box: List[Optional[Exception]] = [None]

    def _locked_generate():
        if cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
            try:
                streamer.end()
            except Exception:
                pass
            return

        with GPU_LOCK:
            if cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                try:
                    streamer.end()
                except Exception:
                    pass
                return

            inputs = None
            try:
                if pil_images:
                    inputs = processor(text=[prompt_text], images=pil_images, padding=True, return_tensors="pt").to(model.device)
                else:
                    inputs = processor(text=[prompt_text], padding=True, return_tensors="pt").to(model.device)

                seq_in_len = int(inputs["input_ids"].shape[-1])
                if seq_in_len >= bundle_ctx:
                    raise RuntimeError(
                        f"输入长度 ({seq_in_len} tokens) 超出当前设定的上下文上限 ({bundle_ctx // 1024}K = {bundle_ctx} tokens)，"
                        f"请在左侧调大「上下文长度」（最高支持 256K）。"
                    )
                eff_max_toks = max(16, min(max_toks, bundle_ctx - seq_in_len))

                gen_kwargs: Dict[str, Any] = dict(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=eff_max_toks,
                    do_sample=do_sample,
                    stopping_criteria=_make_abort_stopping_criteria(cancel_event),
                )
                if do_sample:
                    gen_kwargs["temperature"] = max(0.05, min(2.0, float(temperature)))
                    gen_kwargs["top_p"] = max(0.05, min(1.0, float(top_p)))
                    if int(top_k) > 0:
                        gen_kwargs["top_k"] = int(top_k)

                with torch.inference_mode():
                    model.generate(**gen_kwargs)
            except torch.cuda.OutOfMemoryError:
                err_box[0] = RuntimeError(
                    "GPU 显存不足 (OOM)：BF16 原生精度长上下文峰值显存较高，建议切换使用独立的「Qwen3.8-27B-GGUF (UD-Q4_K_M)」量化模型，或适当调低上下文长度。"
                )
                try:
                    streamer.end()
                except Exception:
                    pass
            except Exception as ex:
                err_box[0] = ex
                try:
                    streamer.end()
                except Exception:
                    pass
            finally:
                if inputs is not None:
                    del inputs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                touch_activity()

    t = threading.Thread(target=_locked_generate, daemon=True)
    t.start()
    return streamer, prompt_tokens, start_in_thinking, t, err_box, cancel_event


def attach_macos_api_routes(app: FastAPI):
    """在 Gradio 底层的 FastAPI 实例上挂载原生 macOS 客户端接口 + OpenAI / Anthropic 双协议 API 网关"""
    global API_AUTH_KEY, API_REQUEST_COUNT

    try:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    except Exception:
        # Gradio 启动后的 FastAPI 实例默认已内置全局 CORS 中间件
        pass

    @app.get("/minimax_api/status")
    def api_status(model_id: Optional[str] = None):
        return get_gpu_status_dict(model_id=model_id)

    @app.post("/minimax_api/select_model")
    def api_select_model(req: SelectModelRequest):
        touch_activity()
        return get_gpu_status_dict(model_id=req.model_id)

    @app.post("/minimax_api/config_idle")
    def api_config_idle(req: IdleConfigRequest):
        global IDLE_TIMEOUT_MINUTES
        IDLE_TIMEOUT_MINUTES = req.idle_timeout_minutes
        touch_activity()
        return get_gpu_status_dict()

    @app.post("/minimax_api/config_context")
    def api_config_context(req: ContextConfigRequest):
        global QWEN_CONTEXT_LENGTH
        QWEN_CONTEXT_LENGTH = _clamp_context_length(req.context_length)
        # BF16 Transformers 运行时支持直接热更新动态上下文上限；GGUF 需重载以重分配 KV 槽位
        bf16_rt = LOADED_RUNTIMES.get("qwen38_27b")
        if bf16_rt and bf16_rt.get("llm_bundle"):
            bf16_rt["llm_bundle"]["context_length"] = QWEN_CONTEXT_LENGTH
            bf16_rt["mode_label"] = f"Qwen3.8-27B · BF16 原生 · {QWEN_CONTEXT_LENGTH // 1024}K"
            _sync_legacy_globals()
        touch_activity()
        print(f"[上下文配置] 目标最大上下文长度已设为 {QWEN_CONTEXT_LENGTH // 1024}K ({QWEN_CONTEXT_LENGTH} tokens)")
        return get_gpu_status_dict(model_id=SELECTED_MODEL_ID)

    @app.post("/minimax_api/config_api_key")
    def api_config_api_key(req: ApiKeyConfigRequest):
        global API_AUTH_KEY
        API_AUTH_KEY = (req.api_key or "").strip()
        touch_activity()
        state_str = "已开启密钥鉴权" if API_AUTH_KEY else "已设为免密开放模式"
        print(f"[API 网关] 对外协议网关 ({state_str})")
        return get_gpu_status_dict()

    # --------------------------------------------------------------------------
    # 自定义 GGUF 模型：任意 HuggingFace 仓库动态注册（不局限于内置目录）
    # --------------------------------------------------------------------------
    @app.get("/minimax_api/hf_gguf_files")
    def api_hf_gguf_files(repo: str, hf_token: str = ""):
        """探测 HF 仓库里的全部 .gguf 文件（含量化级别与体积），用于注册前的选择"""
        try:
            return {"ok": True, **hf_api_list_gguf_files(repo, hf_token)}
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": f"读取 {repo} 失败: {str(e)[:300]}"},
            )

    @app.post("/minimax_api/register_custom_model")
    def api_register_custom_model(req: RegisterCustomModelRequest):
        touch_activity()
        try:
            spec = register_custom_gguf_model(
                repo_id=req.repo_id,
                filename=req.filename,
                name=req.name,
                mmproj_file=req.mmproj_file,
                vram_gb=req.vram_gb,
                approx_size_gb=req.approx_size_gb,
                context_length=req.context_length,
            )
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": str(e)[:300]},
            )
        return {"ok": True, "model": spec, "status": get_gpu_status_dict(model_id=spec["id"])}

    @app.post("/minimax_api/remove_custom_model")
    def api_remove_custom_model(req: RemoveCustomModelRequest):
        touch_activity()
        mid = (req.model_id or "").strip()
        if not _is_custom_gguf_model(mid):
            return JSONResponse(
                status_code=404,
                content={"ok": False, "error": f"{mid} 不是已注册的自定义模型"},
            )
        # 若正在显存中，先卸载
        if mid in LOADED_RUNTIMES:
            try:
                unload_model_from_vram_core(mid)
            except Exception as e:
                print(f"[自定义模型] 卸载 {mid} 时跳过一次错误: {e}")
        removed = remove_custom_gguf_model(mid)
        return {"ok": removed, "model_id": mid}

    @app.get("/minimax_api/custom_models")
    def api_custom_models():
        return {
            "ok": True,
            "custom_models": list(CUSTOM_MODEL_SPECS.values()),
            "count": len(CUSTOM_MODEL_SPECS),
        }

    @app.get("/minimax_api/model_catalog")
    def api_model_catalog():
        """完整模型清单（内置 + 自定义）含下载进度与显存状态，供面板渲染"""
        out: List[Dict[str, Any]] = []
        for mid, spec in MODEL_CATALOG.items():
            try:
                fs = _get_cached_model_files_status(mid)
                downloaded = fs["downloaded_count"]
                total = fs["total_count"]
                size_gb = fs["total_size_gb"]
            except Exception:
                downloaded, total, size_gb = 0, 0, 0.0
            rt = LOADED_RUNTIMES.get(mid)
            out.append({
                "id": mid,
                "name": spec.get("name", mid),
                "repo_id": spec.get("repo_id", ""),
                "modality": spec.get("modality", ""),
                "category": spec.get("category", ""),
                "approx_size_gb": spec.get("approx_size_gb", 0.0),
                "vram_gb": spec.get("vram_gb", 0.0),
                "quant": spec.get("quant", ""),
                "is_custom": bool(spec.get("is_custom")),
                "backend": "gguf" if _is_gguf_model_id(mid) else "transformers",
                "downloaded_count": downloaded,
                "total_count": total,
                "downloaded_size_gb": size_gb,
                "downloaded": bool(total and downloaded == total),
                "loaded": mid in LOADED_RUNTIMES,
                "loaded_mode": (rt or {}).get("mode_label", ""),
                "loaded_context": int(((rt or {}).get("llm_bundle") or {}).get("context_length") or 0),
            })
        out.sort(key=lambda x: (x["is_custom"], x["modality"] != "llm_vlm", x["name"]))
        return {"ok": True, "models": out, "selected_model_id": SELECTED_MODEL_ID}

    # --------------------------------------------------------------------------
    # 1. OpenAI 兼容协议: GET /v1/models, POST /v1/chat/completions, POST /v1/images/generations
    # --------------------------------------------------------------------------
    @app.get("/v1/models")
    @app.get("/models")
    def openai_list_models(request: Request):
        auth_err = _verify_external_api_auth(request, protocol="openai")
        if auth_err:
            return auth_err
        now_ts = int(time.time())
        any_qwen_loaded = ("qwen38_27b" in LOADED_RUNTIMES) or ("qwen38_27b_gguf" in LOADED_RUNTIMES)
        models_data = [
            {"id": "unsloth/Qwen3.8-27B-GGUF", "object": "model", "created": now_ts, "owned_by": "unsloth", "loaded": "qwen38_27b_gguf" in LOADED_RUNTIMES},
            {"id": "Qwen3.8-27B-UD-Q4_K_M", "object": "model", "created": now_ts, "owned_by": "unsloth", "loaded": "qwen38_27b_gguf" in LOADED_RUNTIMES},
            {"id": "Qwen/Qwen3.8-27B", "object": "model", "created": now_ts, "owned_by": "qwen", "loaded": any_qwen_loaded},
            {"id": "qwen3.8-27b", "object": "model", "created": now_ts, "owned_by": "qwen", "loaded": any_qwen_loaded},
            {"id": "Qwen/Qwen-Image-2.1", "object": "model", "created": now_ts, "owned_by": "qwen", "loaded": "qwen_image_21" in LOADED_RUNTIMES},
            {"id": "Tongyi-MAI/Z-Image-Turbo", "object": "model", "created": now_ts, "owned_by": "tongyi", "loaded": "z_image_turbo" in LOADED_RUNTIMES},
            {"id": "MiniMaxAI/MiniMax-H3", "object": "model", "created": now_ts, "owned_by": "minimax", "loaded": "minimax_h3" in LOADED_RUNTIMES},
        ]
        # 自定义 GGUF 模型：同时暴露 repo_id 与本地面板 id 两种别名
        for mid, spec in CUSTOM_MODEL_SPECS.items():
            owner = (spec.get("repo_id") or "").split("/")[0] or "custom"
            loaded = mid in LOADED_RUNTIMES
            models_data.append({
                "id": spec.get("repo_id") or mid,
                "object": "model", "created": now_ts, "owned_by": owner,
                "loaded": loaded, "panel_model_id": mid,
            })
            models_data.append({
                "id": mid, "object": "model", "created": now_ts, "owned_by": owner,
                "loaded": loaded, "panel_model_id": mid,
            })
        return {"object": "list", "data": models_data}

    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    async def openai_chat_completions(request: Request):
        global API_REQUEST_COUNT
        auth_err = _verify_external_api_auth(request, protocol="openai")
        if auth_err:
            return auth_err
        touch_activity()

        body = await request.json()
        req_model = body.get("model") or "Qwen/Qwen3.8-27B"
        bundle = _get_qwen_llm_bundle(preferred_model_id=str(req_model))
        if bundle is None:
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "Qwen3.8-27B / Qwen3.8-27B-GGUF 尚未加载至 Colab GPU 显存，请先在 Mac 客户端点击「加载到显存」。",
                        "type": "server_error",
                        "code": "model_not_loaded",
                    }
                },
            )

        API_REQUEST_COUNT += 1
        stream = bool(body.get("stream", False))
        max_tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or 16384)

        # 支持通过 enable_thinking / chat_template_kwargs / reasoning_effort 控制思考模式与思考深度
        enable_thinking = True
        effort_val: Optional[str] = None
        ctk = body.get("chat_template_kwargs") if isinstance(body.get("chat_template_kwargs"), dict) else {}
        if "enable_thinking" in body:
            enable_thinking = bool(body["enable_thinking"])
        elif "enable_thinking" in ctk:
            enable_thinking = bool(ctk["enable_thinking"])
        raw_eff = str(body.get("reasoning_effort") or ctk.get("reasoning_effort") or "").strip().lower()
        if raw_eff in ("none", "disabled", "off"):
            enable_thinking = False
        elif raw_eff:
            effort_val = raw_eff

        temperature = float(body["temperature"]) if body.get("temperature") is not None else (1.0 if enable_thinking else 0.7)
        top_p = float(body["top_p"]) if body.get("top_p") is not None else (0.95 if enable_thinking else 0.80)
        top_k = int(body["top_k"]) if body.get("top_k") is not None else 20
        min_p = float(body["min_p"]) if body.get("min_p") is not None else 0.0
        presence_penalty = float(body["presence_penalty"]) if body.get("presence_penalty") is not None else (0.0 if enable_thinking else 1.5)
        repeat_penalty = float(body["repeat_penalty"]) if body.get("repeat_penalty") is not None else 1.0

        print(f"[OpenAI API] POST /v1/chat/completions | model={req_model} | backend={bundle.get('backend')} | stream={stream} | thinking={enable_thinking}")

        hf_messages, pil_images = _normalize_to_hf_messages_and_images(body.get("messages", []), protocol="openai")
        hf_tools = _normalize_hf_tools(body.get("tools"), protocol="openai")

        streamer, prompt_tokens, start_in_thinking, gen_thread, err_box, cancel_event = _prepare_qwen_streamer_generation(
            hf_messages=hf_messages,
            pil_images=pil_images,
            hf_tools=hf_tools,
            enable_thinking=enable_thinking,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            preferred_model_id=str(req_model),
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            reasoning_effort=effort_val,
        )

        cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:20]}"
        created_ts = int(time.time())

        if not stream:
            parser = _IncrementalQwenStreamParser(start_in_thinking=start_in_thinking)
            think_parts: List[str] = []
            text_parts: List[str] = []
            out_chunks = 0
            try:
                for piece in streamer:
                    out_chunks += 1
                    for kind, val in parser.feed(piece):
                        if kind == "thinking":
                            think_parts.append(val)
                        elif kind == "text":
                            text_parts.append(val)
            finally:
                cancel_event.set()
            tail_events, tool_calls = parser.finish()
            for kind, val in tail_events:
                if kind == "thinking":
                    think_parts.append(val)
                elif kind == "text":
                    text_parts.append(val)
            gen_thread.join(timeout=5.0)
            if err_box[0] is not None:
                return JSONResponse(status_code=500, content={"error": {"message": str(err_box[0]), "type": "server_error"}})

            full_thinking = "".join(think_parts).strip()
            full_text, extra_tools = _extract_qwen_tool_calls("".join(text_parts).strip())
            all_tools = tool_calls + extra_tools

            msg_obj: Dict[str, Any] = {"role": "assistant", "content": full_text if (full_text or not all_tools) else None}
            if full_thinking:
                msg_obj["reasoning_content"] = full_thinking
            if all_tools:
                msg_obj["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"], ensure_ascii=False)},
                    }
                    for tc in all_tools
                ]

            final_prompt_tokens = int(getattr(streamer, "prompt_tokens", prompt_tokens) or prompt_tokens)
            comp_tokens = int(getattr(streamer, "generated_tokens", 0)) or (
                _count_bundle_tokens(bundle, full_thinking + full_text) if (full_thinking or full_text) else max(1, out_chunks)
            )
            finish_reason = "tool_calls" if all_tools else "stop"

            return {
                "id": cmpl_id,
                "object": "chat.completion",
                "created": created_ts,
                "model": req_model,
                "choices": [{"index": 0, "message": msg_obj, "finish_reason": finish_reason}],
                "usage": {
                    "prompt_tokens": final_prompt_tokens,
                    "completion_tokens": comp_tokens,
                    "total_tokens": final_prompt_tokens + comp_tokens,
                },
            }

        def _openai_sse_stream():
            parser = _IncrementalQwenStreamParser(start_in_thinking=start_in_thinking)
            comp_chars = 0
            try:
                first_chunk = {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req_model,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"

                for piece in streamer:
                    comp_chars += len(piece)
                    for kind, val in parser.feed(piece):
                        if not val:
                            continue
                        delta_payload = {"reasoning_content": val} if kind == "thinking" else {"content": val}
                        chunk_obj = {
                            "id": cmpl_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": req_model,
                            "choices": [{"index": 0, "delta": delta_payload, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk_obj, ensure_ascii=False)}\n\n"

                tail_events, tool_calls = parser.finish()
                for kind, val in tail_events:
                    if not val:
                        continue
                    delta_payload = {"reasoning_content": val} if kind == "thinking" else {"content": val}
                    chunk_obj = {
                        "id": cmpl_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": req_model,
                        "choices": [{"index": 0, "delta": delta_payload, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk_obj, ensure_ascii=False)}\n\n"

                gen_thread.join(timeout=5.0)
                if getattr(streamer, "generated_tokens", 0) > 0:
                    print(f"[OpenAI API] 流式完成 | {streamer.generated_tokens} tokens | {streamer.tok_per_sec} tok/s")
            finally:
                cancel_event.set()

            if tool_calls:
                tc_deltas = [
                    {
                        "index": idx,
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"], ensure_ascii=False)},
                    }
                    for idx, tc in enumerate(tool_calls)
                ]
                tc_chunk = {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req_model,
                    "choices": [{"index": 0, "delta": {"tool_calls": tc_deltas}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(tc_chunk, ensure_ascii=False)}\n\n"

            final_prompt_toks = int(getattr(streamer, "prompt_tokens", prompt_tokens) or prompt_tokens)
            est_out_toks = int(getattr(streamer, "generated_tokens", 0)) or max(1, comp_chars // 3)
            final_chunk = {
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": req_model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if tool_calls else "stop"}],
                "usage": {
                    "prompt_tokens": final_prompt_toks,
                    "completion_tokens": est_out_toks,
                    "total_tokens": final_prompt_toks + est_out_toks,
                },
            }
            yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            _openai_sse_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    @app.post("/v1/images/generations")
    @app.post("/images/generations")
    async def openai_images_generations(request: Request):
        """当显存中驻留了 Qwen-Image-2.1 或 Z-Image-Turbo（可与 Qwen3.8-27B 同时共驻）时，提供标准 OpenAI 生图接口"""
        global API_REQUEST_COUNT
        auth_err = _verify_external_api_auth(request, protocol="openai")
        if auth_err:
            return auth_err
        touch_activity()

        body = await request.json()
        req_model_str = str(body.get("model") or "").lower()
        img_target_id: Optional[str] = None
        if ("z-image" in req_model_str or "z_image" in req_model_str) and "z_image_turbo" in LOADED_RUNTIMES:
            img_target_id = "z_image_turbo"
        elif "qwen" in req_model_str and "qwen_image_21" in LOADED_RUNTIMES:
            img_target_id = "qwen_image_21"
        else:
            for cand in ("z_image_turbo", "qwen_image_21"):
                if cand in LOADED_RUNTIMES:
                    img_target_id = cand
                    break

        if not img_target_id:
            return JSONResponse(
                status_code=503,
                content={"error": {"message": "请先在客户端加载 Qwen-Image-2.1 或 Z-Image-Turbo 至显存（可与 Qwen3.8-27B 同时常驻）。", "type": "server_error"}},
            )

        img_pipe = LOADED_RUNTIMES[img_target_id]["pipeline"]
        API_REQUEST_COUNT += 1
        prompt = body.get("prompt", "")
        size_str = str(body.get("size", "1024x1024")).lower()
        try:
            w, h = [int(x) for x in size_str.split("x", 1)]
        except Exception:
            w, h = 1024, 1024
        n = max(1, min(4, int(body.get("n", 1))))
        steps = int(body.get("steps") or (8 if img_target_id == "z_image_turbo" else 40))

        results = []
        with GPU_LOCK:
            for _ in range(n):
                cur_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
                if img_target_id == "z_image_turbo":
                    img = img_pipe(prompt=prompt, width=w, height=h, num_inference_steps=steps, seed=cur_seed, rand_device="cuda")
                else:
                    img = img_pipe(prompt=prompt, negative_prompt=" ", width=w, height=h, num_inference_steps=steps, seed=cur_seed)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                results.append({"b64_json": base64.b64encode(buf.getvalue()).decode("ascii")})
        return {"created": int(time.time()), "data": results}

    # --------------------------------------------------------------------------
    # 2. Anthropic Messages 协议: POST /v1/messages & POST /v1/messages/count_tokens
    #    完美兼容 Anthropic Python/TS SDK、Claude Code (ANTHROPIC_BASE_URL) 等工具
    # --------------------------------------------------------------------------
    @app.post("/v1/messages/count_tokens")
    @app.post("/messages/count_tokens")
    @app.post("/v1/v1/messages/count_tokens")
    async def anthropic_count_tokens(request: Request):
        auth_err = _verify_external_api_auth(request, protocol="anthropic")
        if auth_err:
            return auth_err
        body = await request.json()
        raw_txt = json.dumps(body.get("messages", []), ensure_ascii=False)
        bundle = _get_qwen_llm_bundle(preferred_model_id=str(body.get("model") or ""))
        toks = _count_bundle_tokens(bundle, raw_txt)
        return {"input_tokens": toks}

    @app.post("/v1/messages")
    @app.post("/messages")
    @app.post("/v1/v1/messages")
    async def anthropic_messages(request: Request):
        global API_REQUEST_COUNT
        auth_err = _verify_external_api_auth(request, protocol="anthropic")
        if auth_err:
            return auth_err
        touch_activity()

        body = await request.json()
        req_model = body.get("model") or "Qwen/Qwen3.8-27B"
        bundle = _get_qwen_llm_bundle(preferred_model_id=str(req_model))
        if bundle is None:
            return JSONResponse(
                status_code=503,
                content={
                    "type": "error",
                    "error": {
                        "type": "overloaded_error",
                        "message": "Qwen3.8-27B / Qwen3.8-27B-GGUF 尚未加载至 Colab GPU 显存，请先在 Mac 客户端点击「加载到显存」。",
                    },
                },
            )

        API_REQUEST_COUNT += 1
        stream = bool(body.get("stream", False))
        max_tokens = int(body.get("max_tokens") or 16384)

        thinking_cfg = body.get("thinking")
        if isinstance(thinking_cfg, dict):
            enable_thinking = thinking_cfg.get("type") != "disabled"
        else:
            enable_thinking = True

        temperature = float(body["temperature"]) if body.get("temperature") is not None else (1.0 if enable_thinking else 0.7)
        top_p = float(body["top_p"]) if body.get("top_p") is not None else (0.95 if enable_thinking else 0.80)
        top_k = int(body["top_k"]) if body.get("top_k") is not None else 20

        print(f"[Anthropic API] POST /v1/messages | model={req_model} | backend={bundle.get('backend')} | stream={stream} | thinking={enable_thinking}")

        hf_messages, pil_images = _normalize_to_hf_messages_and_images(
            body.get("messages", []),
            system_prompt=body.get("system"),
            protocol="anthropic",
        )
        hf_tools = _normalize_hf_tools(body.get("tools"), protocol="anthropic")

        streamer, prompt_tokens, start_in_thinking, gen_thread, err_box, cancel_event = _prepare_qwen_streamer_generation(
            hf_messages=hf_messages,
            pil_images=pil_images,
            hf_tools=hf_tools,
            enable_thinking=enable_thinking,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            preferred_model_id=str(req_model),
            top_k=top_k,
            min_p=0.0,
            presence_penalty=0.0 if enable_thinking else 1.5,
            repeat_penalty=1.0,
        )

        msg_id = f"msg_{uuid.uuid4().hex[:22]}"

        if not stream:
            parser = _IncrementalQwenStreamParser(start_in_thinking=start_in_thinking)
            think_parts: List[str] = []
            text_parts: List[str] = []
            try:
                for piece in streamer:
                    for kind, val in parser.feed(piece):
                        if kind == "thinking":
                            think_parts.append(val)
                        elif kind == "text":
                            text_parts.append(val)
            finally:
                cancel_event.set()
            tail_events, tool_calls = parser.finish()
            for kind, val in tail_events:
                if kind == "thinking":
                    think_parts.append(val)
                elif kind == "text":
                    text_parts.append(val)
            gen_thread.join(timeout=5.0)
            if err_box[0] is not None:
                return JSONResponse(
                    status_code=500,
                    content={"type": "error", "error": {"type": "api_error", "message": str(err_box[0])}},
                )

            full_thinking = "".join(think_parts).strip()
            full_text, extra_tools = _extract_qwen_tool_calls("".join(text_parts).strip())
            all_tools = tool_calls + extra_tools

            content_blocks: List[Dict[str, Any]] = []
            if full_thinking:
                content_blocks.append({"type": "thinking", "thinking": full_thinking, "signature": "qwen38-sig"})
            if full_text or not all_tools:
                content_blocks.append({"type": "text", "text": full_text})
            for tc in all_tools:
                content_blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["arguments"]})

            final_prompt_tokens = int(getattr(streamer, "prompt_tokens", prompt_tokens) or prompt_tokens)
            out_tokens = int(getattr(streamer, "generated_tokens", 0)) or (
                _count_bundle_tokens(bundle, full_thinking + full_text) if (full_thinking or full_text) else 1
            )
            stop_reason = "tool_use" if all_tools else "end_turn"

            return {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": req_model,
                "content": content_blocks,
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": {"input_tokens": final_prompt_tokens, "output_tokens": out_tokens},
            }

        def _anthropic_sse_event(ev_name: str, payload: Dict[str, Any]) -> str:
            return f"event: {ev_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

        def _anthropic_sse_stream():
            parser = _IncrementalQwenStreamParser(start_in_thinking=start_in_thinking)
            yield _anthropic_sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "model": req_model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": prompt_tokens, "output_tokens": 1},
                    },
                },
            )

            next_block_idx = 0
            active_block_type: Optional[str] = None  # "thinking" | "text" | None
            active_block_idx = 0
            comp_chars = 0

            def _ensure_block(target_type: str):
                nonlocal next_block_idx, active_block_type, active_block_idx
                out_strs: List[str] = []
                if active_block_type == target_type:
                    return out_strs
                if active_block_type is not None:
                    if active_block_type == "thinking":
                        out_strs.append(
                            _anthropic_sse_event(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": active_block_idx,
                                    "delta": {"type": "signature_delta", "signature": "qwen38-sig"},
                                },
                            )
                        )
                    out_strs.append(
                        _anthropic_sse_event(
                            "content_block_stop",
                            {"type": "content_block_stop", "index": active_block_idx},
                        )
                    )
                active_block_idx = next_block_idx
                next_block_idx += 1
                active_block_type = target_type
                block_init = (
                    {"type": "thinking", "thinking": ""}
                    if target_type == "thinking"
                    else {"type": "text", "text": ""}
                )
                out_strs.append(
                    _anthropic_sse_event(
                        "content_block_start",
                        {"type": "content_block_start", "index": active_block_idx, "content_block": block_init},
                    )
                )
                return out_strs

            try:
                for piece in streamer:
                    comp_chars += len(piece)
                    for kind, val in parser.feed(piece):
                        if not val:
                            continue
                        for s in _ensure_block(kind):
                            yield s
                        delta_obj = (
                            {"type": "thinking_delta", "thinking": val}
                            if kind == "thinking"
                            else {"type": "text_delta", "text": val}
                        )
                        yield _anthropic_sse_event(
                            "content_block_delta",
                            {"type": "content_block_delta", "index": active_block_idx, "delta": delta_obj},
                        )

                tail_events, tool_calls = parser.finish()
                for kind, val in tail_events:
                    if not val:
                        continue
                    for s in _ensure_block(kind):
                        yield s
                    delta_obj = (
                        {"type": "thinking_delta", "thinking": val}
                        if kind == "thinking"
                        else {"type": "text_delta", "text": val}
                    )
                    yield _anthropic_sse_event(
                        "content_block_delta",
                        {"type": "content_block_delta", "index": active_block_idx, "delta": delta_obj},
                    )

                gen_thread.join(timeout=5.0)
                if getattr(streamer, "generated_tokens", 0) > 0:
                    print(f"[Anthropic API] 流式完成 | {streamer.generated_tokens} tokens | {streamer.tok_per_sec} tok/s")
            finally:
                cancel_event.set()

            if active_block_type is not None:
                if active_block_type == "thinking":
                    yield _anthropic_sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": active_block_idx,
                            "delta": {"type": "signature_delta", "signature": "qwen38-sig"},
                        },
                    )
                yield _anthropic_sse_event(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": active_block_idx},
                )
                active_block_type = None
            elif not tool_calls:
                # 若全程无输出块，保底发送一个空文本块符合 Anthropic 协议规范
                yield _anthropic_sse_event(
                    "content_block_start",
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                )
                yield _anthropic_sse_event("content_block_stop", {"type": "content_block_stop", "index": 0})
                next_block_idx = 1

            for tc in tool_calls:
                t_idx = next_block_idx
                next_block_idx += 1
                yield _anthropic_sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": t_idx,
                        "content_block": {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": {}},
                    },
                )
                yield _anthropic_sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": t_idx,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(tc["arguments"], ensure_ascii=False),
                        },
                    },
                )
                yield _anthropic_sse_event("content_block_stop", {"type": "content_block_stop", "index": t_idx})

            est_out_toks = int(getattr(streamer, "generated_tokens", 0)) or max(1, comp_chars // 3)
            yield _anthropic_sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use" if tool_calls else "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": est_out_toks},
                },
            )
            yield _anthropic_sse_event("message_stop", {"type": "message_stop"})

        return StreamingResponse(
            _anthropic_sse_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    @app.post("/minimax_api/download_model")
    def api_download_model(req: DownloadModelRequest):
        touch_activity()
        target_id = req.model_id if (req.model_id and req.model_id in MODEL_CATALOG) else SELECTED_MODEL_ID
        if not MODEL_DOWNLOAD_STATE["is_downloading"]:
            t = threading.Thread(
                target=download_model_weights_core,
                args=(target_id, req.hf_token),
                daemon=True,
            )
            t.start()
        return get_gpu_status_dict(model_id=target_id)

    @app.post("/minimax_api/load_model")
    def api_load_model(req: LoadModelRequest):
        global QWEN_CONTEXT_LENGTH
        touch_activity()
        if req.context_length is not None:
            QWEN_CONTEXT_LENGTH = _clamp_context_length(req.context_length)
        target_id = req.model_id if (req.model_id and req.model_id in MODEL_CATALOG) else SELECTED_MODEL_ID
        if not MODEL_DOWNLOAD_STATE["is_loading_vram"]:
            t = threading.Thread(
                target=load_model_to_vram_core,
                args=(target_id, req.precision_mode, req.offload_mode, req.hf_token, req.context_length),
                daemon=True,
            )
            t.start()
        return get_gpu_status_dict(model_id=target_id)

    @app.post("/minimax_api/unload_model")
    def api_unload_model(req: UnloadModelRequest):
        touch_activity()
        unload_model_from_vram_core(req.model_id or None)
        return get_gpu_status_dict(model_id=SELECTED_MODEL_ID)

    @app.post("/minimax_api/cancel")
    def api_cancel_generation():
        global CURRENT_STAGE_BAR_LABEL
        touch_activity()
        ABORT_GENERATION_EVENT.set()
        for ev in list(_JOB_CANCEL_EVENTS.values()):
            ev.set()
        for jid, j in list(JOBS.items()):
            if j.get("status") in ("queued", "running"):
                j["status"] = "failed"
                j["progress"] = 0.0
                j["message"] = "已手动停止当前推理任务"
        CURRENT_STAGE_BAR_LABEL = None
        with CONSOLE_LOCK:
            ACTIVE_TQDM_BARS.clear()
        print("[任务控制] 已收到停止指令，正在中断当前 GPU 推理并清空进度缓存...")

        def _clear_abort_after_release():
            with GPU_LOCK:
                ABORT_GENERATION_EVENT.clear()
                with CONSOLE_LOCK:
                    ACTIVE_TQDM_BARS.clear()

        threading.Thread(target=_clear_abort_after_release, daemon=True).start()
        return get_gpu_status_dict(model_id=SELECTED_MODEL_ID)

    @app.post("/minimax_api/generate")
    def api_generate(req: GenerateVideoRequest):
        touch_activity()
        target_model_id = (
            req.model_id
            if (req.model_id and req.model_id in MODEL_CATALOG)
            else SELECTED_MODEL_ID
        )
        if not LOADED_RUNTIMES and PIPELINE is None and LLM_BUNDLE is None:
            return {
                "job_id": "none",
                "status": "failed",
                "progress": 0.0,
                "message": "尚未有任何模型加载至 GPU 显存！请先在左侧点击「下载权重」与「加载到显存」。",
                "seed": req.seed,
                "elapsed": 0.0,
                "filename": None,
                "video_url": None,
            }

        # 若请求的是 LLM/VLM 但当前显存中驻留的是另一变体（如选中了 BF16 但已加载 GGUF，或反之），自动无缝路由至已就绪变体
        if _is_llm_model_id(target_model_id) and target_model_id not in LOADED_RUNTIMES:
            alt_bundle = _get_qwen_llm_bundle(preferred_model_id=target_model_id)
            if alt_bundle and alt_bundle.get("model_id") in LOADED_RUNTIMES:
                target_model_id = str(alt_bundle["model_id"])
                req.model_id = target_model_id

        if target_model_id not in LOADED_RUNTIMES and LOADED_MODEL_ID != target_model_id:
            resident_names = "、".join(rt["name"] for rt in LOADED_RUNTIMES.values()) or "无"
            req_name = MODEL_CATALOG.get(target_model_id, {}).get("name", target_model_id)
            return {
                "job_id": "none",
                "status": "failed",
                "progress": 0.0,
                "message": f"{req_name} 尚未加载到显存（当前已常驻：{resident_names}）。请点击左侧「加载到显存」，显存充裕时可同时常驻无需卸载现有模型。",
                "seed": req.seed,
                "elapsed": 0.0,
                "filename": None,
                "video_url": None,
            }

        # 若此前有残留未结束的任务，先置位其专属中断信号并清空进度条，防止旧任务继续占用 GPU 或污染新任务进度
        for old_jid, old_j in list(JOBS.items()):
            if old_j.get("status") in ("queued", "running"):
                old_ev = _JOB_CANCEL_EVENTS.get(old_jid)
                if old_ev is not None:
                    old_ev.set()
                old_j["status"] = "failed"
                old_j["progress"] = 0.0
                old_j["message"] = "已被新提交任务接替中断"
        with CONSOLE_LOCK:
            ACTIVE_TQDM_BARS.clear()

        is_llm_req = _is_llm_model_id(target_model_id)
        batch_count = 1 if is_llm_req else max(1, min(8, int(getattr(req, "batch_count", 1) or 1)))
        job_id = uuid.uuid4().hex[:12]
        _JOB_CANCEL_EVENTS[job_id] = threading.Event()
        JOBS[job_id] = {
            "job_id": job_id,
            "model_id": target_model_id,
            "status": "queued",
            "progress": 0.0,
            "message": f"任务已提交至 Colab 队列 (共 {batch_count} 项)..." if batch_count > 1 else "任务已提交至 Colab 队列...",
            "seed": req.seed,
            "started_at": time.time(),
            "elapsed": 0.0,
            "filename": None,
            "video_url": None,
            "media_type": "text" if is_llm_req else (
                "image" if MODEL_CATALOG.get(target_model_id, {}).get("modality") == "image" else "video"
            ),
            "text_output": "",
            "thinking_output": "",
            "batch_count": batch_count,
            "completed_count": 0,
            "completed_videos": [],
        }
        t = threading.Thread(target=_run_async_generation_job, args=(job_id, req), daemon=True)
        t.start()
        return JOBS[job_id]

    @app.get("/minimax_api/job/{job_id}")
    def api_job_status(job_id: str):
        if job_id not in JOBS:
            raise HTTPException(status_code=404, detail="Job not found")
        res = dict(JOBS[job_id])
        if res.get("status") in ("queued", "running"):
            touch_activity()
            if res.get("started_at"):
                res["elapsed"] = round(max(0.0, time.time() - float(res["started_at"])), 1)
        res["console_lines"] = get_live_console_Lines()
        res["telemetry"] = get_live_telemetry_snapshot()
        return res

    @app.get("/minimax_api/video/{filename}")
    @app.get("/minimax_api/media/{filename}")
    def api_download_media(filename: str):
        touch_activity()
        safe_name = os.path.basename(filename)
        file_path = os.path.join(DEFAULT_OUTPUT_DIR, safe_name)
        if not os.path.exists(file_path) and DRIVE_MOUNTED:
            for leg_out in LEGACY_DRIVE_OUTPUT_DIRS:
                alt_path = os.path.join(leg_out, safe_name)
                if os.path.exists(alt_path):
                    file_path = alt_path
                    break
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Media file not found")
        ext = os.path.splitext(safe_name)[1].lower()
        mime = (
            "image/png" if ext == ".png"
            else "image/jpeg" if ext in (".jpg", ".jpeg")
            else "image/webp" if ext == ".webp"
            else "video/mp4"
        )
        return FileResponse(file_path, media_type=mime, filename=safe_name)

    @app.post("/minimax_api/disconnect")
    def api_disconnect():
        msg = disconnect_colab_runtime()
        return {"ok": True, "message": msg}

    # --------------------------------------------------------------------------
    # ComfyUI 云端节点工作台与 Google Drive 持久化专用接口 (/minimax_api/comfyui/*)
    # --------------------------------------------------------------------------
    @app.get("/minimax_api/comfyui/status")
    def api_comfyui_status():
        return {"ok": True, "comfyui": get_comfyui_status_dict()}

    @app.post("/minimax_api/comfyui/start")
    def api_comfyui_start(req: ComfyStartRequest):
        touch_activity()
        if not COMFYUI_STATE["is_starting"]:
            threading.Thread(
                target=start_comfyui_core,
                args=(req.unload_existing, req.force_restart),
                daemon=True,
            ).start()
        return get_gpu_status_dict(model_id=SELECTED_MODEL_ID)

    @app.post("/minimax_api/comfyui/stop")
    def api_comfyui_stop():
        touch_activity()
        stop_comfyui_core(sync_before_stop=True)
        return get_gpu_status_dict(model_id=SELECTED_MODEL_ID)

    @app.post("/minimax_api/comfyui/free_vram")
    def api_comfyui_free_vram():
        touch_activity()
        ok, msg = free_comfyui_vram_core()
        res = get_gpu_status_dict(model_id=SELECTED_MODEL_ID)
        res["action_ok"] = ok
        res["action_message"] = msg
        return res

    @app.post("/minimax_api/comfyui/sync_drive")
    def api_comfyui_sync_drive():
        touch_activity()
        ok, msg = sync_comfyui_to_drive_core(quiet=False)
        res = get_gpu_status_dict(model_id=SELECTED_MODEL_ID)
        res["action_ok"] = ok
        res["action_message"] = msg
        return res

    @app.post("/minimax_api/comfyui/download_weight")
    def api_comfyui_download_weight(req: ComfyWeightDownloadRequest):
        touch_activity()
        if not COMFYUI_STATE["is_downloading_weight"]:
            threading.Thread(
                target=download_comfyui_model_core,
                args=(req.url_or_repo, req.filename, req.subfolder, req.hf_token),
                daemon=True,
            ).start()
        return get_gpu_status_dict(model_id=SELECTED_MODEL_ID)

    @app.post("/minimax_api/comfyui/install_pack")
    def api_comfyui_install_pack(req: ComfyInstallPackRequest):
        touch_activity()
        if not COMFYUI_STATE["is_downloading_weight"] and not COMFYUI_STATE.get("installing_pack_id"):
            threading.Thread(
                target=install_comfyui_preset_pack_core,
                args=(req.pack_id, req.auto_start, req.hf_token),
                daemon=True,
            ).start()
        return get_gpu_status_dict(model_id=SELECTED_MODEL_ID)

    @app.get("/minimax_api/comfyui/output_media/{rel_path:path}")
    def api_comfyui_output_media(rel_path: str):
        touch_activity()
        comfy_root, persist_root, _, _, _ = _resolve_comfyui_persistence_paths()
        for base_out in (os.path.join(comfy_root, "output"), os.path.join(persist_root, "output")):
            if not os.path.isdir(base_out):
                continue
            cand = os.path.normpath(os.path.join(base_out, rel_path))
            if not os.path.abspath(cand).startswith(os.path.abspath(base_out)):
                raise HTTPException(status_code=400, detail="Invalid output path")
            if os.path.isfile(cand):
                safe_name = os.path.basename(cand)
                ext = os.path.splitext(safe_name)[1].lower()
                mime = (
                    "image/png" if ext == ".png"
                    else "image/jpeg" if ext in (".jpg", ".jpeg")
                    else "image/webp" if ext == ".webp"
                    else "image/gif" if ext == ".gif"
                    else "video/webm" if ext == ".webm"
                    else "video/mp4"
                )
                return FileResponse(cand, media_type=mime, filename=safe_name)
        raise HTTPException(status_code=404, detail="ComfyUI output file not found")


# ==============================================================================
# 预设提示词模板与 Gradio 界面
# ==============================================================================
PRESET_EXAMPLES = [
    [
        "Masterpiece Makoto Shinkai anime style, 2D cel-shaded animation. A teenage girl in a Japanese sailor school uniform standing at a sunlit railway crossing in late spring. A green commuter train speeds past in front of her, creating a strong gust of wind that blows her long black hair and a flurry of pink cherry blossom petals across the frame. Dramatic lens flare, glowing cumulous clouds in a deep blue sky, vibrant cinematic lighting, smooth 24fps keyframe animation.",
        "Metallic clacking of train wheels on railway tracks, rhythmic dinging of the railroad crossing warning bell, warm spring wind rustling through leaves, a soft gasp from the girl.",
        "Emotional and nostalgic solo piano melody joined by warm acoustic guitar and soaring cinematic strings.",
        "832 × 480 (16:9 横屏标准 - 推荐)",
        124,
        35,
    ],
    [
        "High-budget ufotable anime action sakuga, dynamic fast-paced camera tracking. A silver-haired anime swordsman in a dark haori coat dashes across a shattered stone temple under a blood-red full moon, swinging a katana wreathed in crackling blue lightning and swirling water dragon effects. Sparks and glowing embers fly toward the camera in slow-motion impact frames, intense anime shading, dramatic motion blur.",
        "Sharp metallic sword unsheathing and clashing clang, roaring thunder and crackling electric arcs, heavy stone debris crumbling, fierce battle cry.",
        "Epic dark orchestral battle music with fast taiko drums, intense choir, and distorted electric guitar riffs.",
        "832 × 480 (16:9 横屏标准 - 推荐)",
        124,
        40,
    ],
]

CUSTOM_CSS = """
.header-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
    color: #ffffff;
    padding: 22px 28px;
    border-radius: 14px;
    margin-bottom: 16px;
    border: 1px solid rgba(255, 255, 255, 0.12);
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
}
.header-banner h1 {
    margin: 0 0 6px 0;
    font-size: 1.75rem;
    font-weight: 700;
    color: #f8fafc;
}
.header-banner p {
    margin: 0;
    font-size: 0.95rem;
    color: #cbd5e1;
}
.status-box textarea {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 0.85rem !important;
}
"""


WEBUI_HEADER_ZH = """
<div class="header-banner">
    <h1>⚡ EasyColab · Google Colab 多模态 AI 推理与生成服务端</h1>
    <p>支持 <b>ComfyUI 节点工作台 (云盘软链持久化) · MiniMax-H3 · Qwen3.8-27B (GGUF+MTP / BF16) · Qwen-Image-2.1 · Z-Image-Turbo</b> · 自动关机看门狗</p>
</div>
"""

WEBUI_HEADER_EN = """
<div class="header-banner">
    <h1>⚡ EasyColab · Google Colab Multimodal AI Inference & Generation Server</h1>
    <p>Supports <b>ComfyUI Node Studio (Google Drive Persistence) · MiniMax-H3 · Qwen3.8-27B (GGUF+MTP / BF16) · Qwen-Image-2.1 · Z-Image-Turbo</b> · Idle Auto-Shutdown Watchdog</p>
</div>
"""

RESOLUTION_CHOICES_ZH = [
    ("832 × 480 (16:9 横屏标准 - 推荐)", "832 × 480 (16:9 横屏标准 - 推荐)"),
    ("480 × 832 (9:16 竖屏短视频)", "480 × 832 (9:16 竖屏短视频)"),
    ("1280 × 720 (16:9 720P 高清)", "1280 × 720 (16:9 720P 高清)"),
    ("720 × 1280 (9:16 720P 竖屏高清)", "720 × 1280 (9:16 720P 竖屏高清)"),
    ("640 × 640 (1:1 方形)", "640 × 640 (1:1 方形)"),
]

RESOLUTION_CHOICES_EN = [
    ("832 × 480 (16:9 Landscape · Recommended)", "832 × 480 (16:9 横屏标准 - 推荐)"),
    ("480 × 832 (9:16 Portrait Short)", "480 × 832 (9:16 竖屏短视频)"),
    ("1280 × 720 (16:9 720P HD)", "1280 × 720 (16:9 720P 高清)"),
    ("720 × 1280 (9:16 720P Portrait HD)", "720 × 1280 (9:16 720P 竖屏高清)"),
    ("640 × 640 (1:1 Square)", "640 × 640 (1:1 方形)"),
]


def create_ui() -> gr.Blocks:
    with gr.Blocks(title="EasyColab · Colab 多模型 AI 服务端") as demo:
        ui_lang_state = gr.State(value="zh")

        with gr.Row():
            with gr.Column(scale=9):
                header_html = gr.HTML(WEBUI_HEADER_ZH)
            with gr.Column(scale=1, min_width=130):
                lang_toggle_btn = gr.Button("🌐 English", size="sm", variant="secondary")

        with gr.Row():
            with gr.Column(scale=4):
                with gr.Group():
                    sec1_md = gr.Markdown("### 📦 1. 模型权重管理中心 (先下载检查，再加载显存)")
                    gpu_status_text = gr.Textbox(
                        label="硬件、算力消耗与模型检查报告",
                        value=get_gpu_and_model_summary_text(),
                        lines=9,
                        interactive=False,
                        elem_classes=["status-box"],
                    )
                    refresh_gpu_btn = gr.Button("🔍 检查 / 刷新模型与算力状态", size="sm")

                    precision_Choice = gr.Radio(
                        choices=[
                            ("BF16 原生精度版 (官方满血权重 · G4 96GB 极速推理)", "bf16"),
                        ],
                        value="bf16",
                        label="模型权重精度",
                    )

                    offload_choice = gr.Radio(
                        choices=[
                            ("⚡ 零拷贝极速映射 + GPU 显存常驻 (推荐 · 5秒秒加载)", "disk"),
                            ("CPU 内存全量预载 (耗时较长 · 约3~5分钟)", "cpu"),
                        ],
                        value="disk",
                        label="显存加载与卸载策略 (VRAM Strategy)",
                    )

                    with gr.Accordion("🔑 Hugging Face Token (可选 / Optional)", open=False):
                        hf_token_input = gr.Textbox(
                            label="HF Access Token",
                            placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxx",
                            type="password",
                        )

                    with gr.Row():
                        download_model_btn = gr.Button(
                            "📥 1. 极速下载模型权重",
                            variant="secondary",
                        )
                        load_model_btn = gr.Button(
                            "🚀 2. 加载模型至显存",
                            variant="primary",
                        )

                with gr.Group():
                    sec2_md = gr.Markdown("### ⚙️ 2. 视频与采样参数")
                    resolution_dropdown = gr.Dropdown(
                        choices=RESOLUTION_CHOICES_ZH,
                        value="832 × 480 (16:9 横屏标准 - 推荐)",
                        label="输出分辨率 (Resolution)",
                    )
                    with gr.Row():
                        num_frames_slider = gr.Slider(
                            minimum=90,
                            maximum=362,
                            step=17,
                            value=124,
                            label="生成总帧数 (90~362帧 ≈ 4~15秒, 17n+5 对齐)",
                        )
                        fps_slider = gr.Slider(
                            minimum=16,
                            maximum=30,
                            step=1,
                            value=24,
                            label="帧率 (FPS)",
                        )
                    steps_slider = gr.Slider(
                        minimum=10,
                        maximum=60,
                        step=5,
                        value=35,
                        label="采样步数 (Inference Steps，推荐 30~50)",
                    )
                    with gr.Row():
                        seed_input = gr.Number(
                            value=42,
                            precision=0,
                            label="随机种子 (Seed)",
                        )
                        randomize_seed_chk = gr.Checkbox(
                            value=True,
                            label="🎲 每次随机种子",
                        )

                    save_dir_input = gr.Textbox(
                        value=DEFAULT_OUTPUT_DIR,
                        label="📂 视频自动保存路径",
                    )

                with gr.Accordion("💡 Google Colab 算力省额度工具 / Compute Quota Saver", open=True):
                    disconnect_btn = gr.Button("🛑 停止任务并释放 Colab 运行时 (省计算单元)", variant="stop")

            with gr.Column(scale=6):
                task_mode_state = gr.State(value="t2va")

                with gr.Tabs() as mode_tabs:
                    with gr.TabItem("✨ 文生音视频 / T2VA", id="t2va") as tab_t2va:
                        t2va_desc_md = gr.Markdown("输入画面描述与声音细节，同步生成高清视频与 **32kHz 立体声音轨**。")

                    with gr.TabItem("🖼️ 首尾帧生音视频 / FL2VA", id="fl2va") as tab_fl2va:
                        with gr.Row():
                            first_frame_input = gr.Image(
                                label="📌 首帧图片 (First Frame - index 0)",
                                type="pil",
                                height=220,
                            )
                            last_frame_input = gr.Image(
                                label="🏁 尾帧图片 (Last Frame - index -1，可选)",
                                type="pil",
                                height=220,
                            )

                with gr.Group():
                    visual_prompt_input = gr.Textbox(
                        label="🎥 画面主体与镜头描述 (Visual & Camera Prompt)",
                        placeholder="描述画面主体、动作、运镜方式、光影风格...",
                        lines=4,
                    )
                    with gr.Row():
                        soundscape_prompt_input = gr.Textbox(
                            label="🔊 环境音效与人声描述 (Soundscape & Foley - 可选)",
                            placeholder="例如：雨滴敲击声、脚步踩水声、温柔的女声独白...",
                            lines=2,
                        )
                        music_prompt_input = gr.Textbox(
                            label="🎵 背景配乐风格 (Background Music - 可选)",
                            placeholder="例如：低沉的赛博朋克合成器电子乐、恢弘管弦乐...",
                            lines=2,
                        )

                generate_btn = gr.Button(
                    "🎬 开始生成音视频 (Generate Video + Stereo Audio)",
                    variant="primary",
                    size="lg",
                )

                with gr.Group():
                    output_video = gr.Video(
                        label="🎞️ 生成结果预览 (请开启播放器声音体验原生立体声)",
                        height=420,
                        autoplay=True,
                    )
                    run_status_box = gr.Textbox(
                        label="运行日志与状态",
                        value="请先在左侧点击「1. 极速下载模型权重」并「2. 加载模型至显存」...",
                        interactive=False,
                        lines=2,
                        elem_classes=["status-box"],
                    )

                with gr.Accordion("📚 提示词一键体验模板 / Preset Prompt Templates", open=True):
                    gr.Examples(
                        examples=PRESET_EXAMPLES,
                        inputs=[
                            visual_prompt_input,
                            soundscape_prompt_input,
                            music_prompt_input,
                            resolution_dropdown,
                            num_frames_slider,
                            steps_slider,
                        ],
                    )

                with gr.Accordion("🗂️ 本次会话历史生成记录 / Session History", open=False):
                    history_gallery = gr.Gallery(
                        label="已生成的视频文件",
                        value=get_history_videos(DEFAULT_OUTPUT_DIR),
                        columns=3,
                        height=240,
                    )
                    refresh_history_btn = gr.Button("🔄 刷新历史记录", size="sm")

        def _toggle_webui_lang(cur_lang: str, cur_prec: str, cur_offload: str, cur_res: str, cur_status: str):
            new_lang = "en" if cur_lang == "zh" else "zh"
            is_en = new_lang == "en"
            default_zh_status = "请先在左侧点击「1. 极速下载模型权重」并「2. 加载模型至显存」..."
            default_en_status = "Click '1. Download Weights' then '2. Load to VRAM' on the left to get started..."
            if is_en and cur_status == default_zh_status:
                next_status = default_en_status
            elif not is_en and cur_status == default_en_status:
                next_status = default_zh_status
            else:
                next_status = cur_status

            return (
                new_lang,
                WEBUI_HEADER_EN if is_en else WEBUI_HEADER_ZH,
                "🌐 简体中文" if is_en else "🌐 English",
                "### 📦 1. Model Weights Center (Download First, Then Load to VRAM)" if is_en else "### 📦 1. 模型权重管理中心 (先下载检查，再加载显存)",
                gr.update(label="Hardware, CU Usage & Model Inspection Report" if is_en else "硬件、算力消耗与模型检查报告"),
                "🔍 Check / Refresh GPU & Model Status" if is_en else "🔍 检查 / 刷新模型与算力状态",
                gr.update(
                    choices=[
                        ("Native BF16 Full Precision (Recommended for G4 96GB)" if is_en else "BF16 原生精度版 (官方满血权重 · G4 96GB 极速推理)", "bf16"),
                    ],
                    value=cur_prec or "bf16",
                    label="Model Precision" if is_en else "模型权重精度",
                ),
                gr.update(
                    choices=[
                        ("⚡ Zero-Copy Fast Mapping + GPU VRAM Resident (Recommended · ~5s)" if is_en else "⚡ 零拷贝极速映射 + GPU 显存常驻 (推荐 · 5秒秒加载)", "disk"),
                        ("Full CPU RAM Preload (Slower · ~3-5 min)" if is_en else "CPU 内存全量预载 (耗时较长 · 约3~5分钟)", "cpu"),
                    ],
                    value=cur_offload or "disk",
                    label="VRAM Loading Strategy" if is_en else "显存加载与卸载策略 (VRAM Strategy)",
                ),
                "📥 1. Download Model Weights" if is_en else "📥 1. 极速下载模型权重",
                "🚀 2. Load Model to VRAM" if is_en else "🚀 2. 加载模型至显存",
                "### ⚙️ 2. Video & Sampling Parameters" if is_en else "### ⚙️ 2. 视频与采样参数",
                gr.update(
                    choices=RESOLUTION_CHOICES_EN if is_en else RESOLUTION_CHOICES_ZH,
                    value=cur_res or "832 × 480 (16:9 横屏标准 - 推荐)",
                    label="Output Resolution" if is_en else "输出分辨率 (Resolution)",
                ),
                gr.update(label="Total Frames (90~362 frames ≈ 4~15s, 17n+5 aligned)" if is_en else "生成总帧数 (90~362帧 ≈ 4~15秒, 17n+5 对齐)"),
                gr.update(label="Frame Rate (FPS)" if is_en else "帧率 (FPS)"),
                gr.update(label="Inference Steps (30~50 recommended)" if is_en else "采样步数 (Inference Steps，推荐 30~50)"),
                gr.update(label="Seed" if is_en else "随机种子 (Seed)"),
                gr.update(label="🎲 Randomize Seed Every Run" if is_en else "🎲 每次随机种子"),
                gr.update(label="📂 Output Save Directory" if is_en else "📂 视频自动保存路径"),
                "🛑 Stop Jobs & Release Colab Runtime (Save CU)" if is_en else "🛑 停止任务并释放 Colab 运行时 (省计算单元)",
                "Describe visual scenes and soundscapes to generate HD video with **32kHz stereo audio**." if is_en else "输入画面描述与声音细节，同步生成高清视频与 **32kHz 立体声音轨**。",
                gr.update(label="📌 First Frame Image (index 0)" if is_en else "📌 首帧图片 (First Frame - index 0)"),
                gr.update(label="🏁 Last Frame Image (index -1, Optional)" if is_en else "🏁 尾帧图片 (Last Frame - index -1，可选)"),
                gr.update(
                    label="🎥 Visual & Camera Prompt" if is_en else "🎥 画面主体与镜头描述 (Visual & Camera Prompt)",
                    placeholder="Describe subject, motion, camera movement, lighting, and visual style..." if is_en else "描述画面主体、动作、运镜方式、光影风格...",
                ),
                gr.update(
                    label="🔊 Soundscape & Dialogue (Optional)" if is_en else "🔊 环境音效与人声描述 (Soundscape & Foley - 可选)",
                    placeholder="e.g., Raindrops hitting pavement, footsteps in puddles, soft spoken dialogue..." if is_en else "例如：雨滴敲击声、脚步踩水声、温柔的女声独白...",
                ),
                gr.update(
                    label="🎵 Background Music Style (Optional)" if is_en else "🎵 背景配乐风格 (Background Music - 可选)",
                    placeholder="e.g., Moody cyberpunk synthwave, soaring orchestral strings..." if is_en else "例如：低沉的赛博朋克合成器电子乐、恢弘管弦乐...",
                ),
                "🎬 Generate Video + Stereo Audio" if is_en else "🎬 开始生成音视频 (Generate Video + Stereo Audio)",
                gr.update(label="🎞️ Output Preview (Unmute player for native stereo audio)" if is_en else "🎞️ 生成结果预览 (请开启播放器声音体验原生立体声)"),
                gr.update(label="Run Status & Logs" if is_en else "运行日志与状态", value=next_status),
                gr.update(label="Generated Videos" if is_en else "已生成的视频文件"),
                "🔄 Refresh History" if is_en else "🔄 刷新历史记录",
            )

        lang_toggle_btn.click(
            fn=_toggle_webui_lang,
            inputs=[ui_lang_state, precision_Choice, offload_choice, resolution_dropdown, run_status_box],
            outputs=[
                ui_lang_state,
                header_html,
                lang_toggle_btn,
                sec1_md,
                gpu_status_text,
                refresh_gpu_btn,
                precision_Choice,
                offload_choice,
                download_model_btn,
                load_model_btn,
                sec2_md,
                resolution_dropdown,
                num_frames_slider,
                fps_slider,
                steps_slider,
                seed_input,
                randomize_seed_chk,
                save_dir_input,
                disconnect_btn,
                t2va_desc_md,
                first_frame_input,
                last_frame_input,
                visual_prompt_input,
                soundscape_prompt_input,
                music_prompt_input,
                generate_btn,
                output_video,
                run_status_box,
                history_gallery,
                refresh_history_btn,
            ],
        )

        tab_t2va.select(fn=lambda: "t2va", inputs=None, outputs=task_mode_state)
        tab_fl2va.select(fn=lambda: "fl2va", inputs=None, outputs=task_mode_state)

        refresh_gpu_btn.click(fn=get_gpu_and_model_summary_text, inputs=None, outputs=gpu_status_text)

        download_model_btn.click(
            fn=lambda tok: download_minimax_h3_weights_core(tok)[1],
            inputs=[hf_token_input],
            outputs=run_status_box,
        ).then(fn=get_gpu_and_model_summary_text, inputs=None, outputs=gpu_status_text)

        load_model_btn.click(
            fn=lambda prec, off, tok: load_minimax_h3_to_vram_core(prec, off, tok)[1],
            inputs=[precision_Choice, offload_choice, hf_token_input],
            outputs=run_status_box,
        ).then(fn=get_gpu_and_model_summary_text, inputs=None, outputs=gpu_status_text)

        generate_btn.click(
            fn=generate_video,
            inputs=[
                task_mode_state,
                visual_prompt_input,
                soundscape_prompt_input,
                music_prompt_input,
                first_frame_input,
                last_frame_input,
                resolution_dropdown,
                num_frames_slider,
                steps_slider,
                fps_slider,
                seed_input,
                randomize_seed_chk,
                save_dir_input,
            ],
            outputs=[output_video, run_status_box, seed_input, history_gallery],
        ).then(fn=get_gpu_and_model_summary_text, inputs=None, outputs=gpu_status_text)

        refresh_history_btn.click(
            fn=get_history_videos,
            inputs=save_dir_input,
            outputs=history_gallery,
        )

        disconnect_btn.click(
            fn=disconnect_colab_runtime,
            inputs=None,
            outputs=run_status_box,
        )

    return demo


if __name__ == "__main__":
    _load_custom_gguf_models()
    ui = create_ui()
    ui.queue()
    app, local_url, share_url = ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        inline=False,
        prevent_thread_lock=True,
        css=CUSTOM_CSS,
    )
    attach_macos_api_routes(app)
    print("\n" + "=" * 72)
    print("🍎 【macOS 原生 App 连接地址】请复制下方链接粘贴到 Mac 客户端：")
    print(f"👉  {share_url or local_url}")
    print("=" * 72 + "\n")
    ui.block_thread()
