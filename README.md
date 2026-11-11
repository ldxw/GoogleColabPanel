# Colab Panel

服务器 Docker 一键部署的 **Google Colab Web 管理面板**。底层能力来自 Google 官方开源 [`google-colab-cli`](https://pypi.org/project/google-colab-cli/)（Apache-2.0），面板把它包装成 Web UI + REST API。

不需要浏览器连服务器、不需要 VNC：OAuth 走远程复制粘贴流程，实例创建 / 代码执行 / 文件传输 / 模型部署全在网页里点。

```bash
git clone https://github.com/host1syan/GoogleColabPanel.git
cd GoogleColabPanel && docker compose up -d --build
# 浏览器打开 http://<服务器IP>:8080
```

## 功能

- **账号绑定**：OAuth 远程复制粘贴流程（headless 友好），显示邮箱 / CU 余额 / 每小时消耗 / 活跃实例数
- **会话管理**：新建 GPU 实例（T4 / L4 / A100 / H100 / G4 / CPU，可选大内存）、列表、状态、释放单个 / 全部（含云端孤儿实例）
- **代码执行**：向远程实例内核发送 Python 代码并回显输出
- **网页接入**：一键生成接入该 VM 的 Colab 网页链接（浏览器直接打开）
- **文件**：上传 / 下载实例内文件
- **其他**：重启内核、挂载 Google Drive、远程 pip 安装
- **模型工作台**：一键把 EasyColab 服务端部署到实例，管理 5 个内置模型的权重下载 / 加载 / 卸载，调上下文长度与 API Key，起停 ComfyUI，并通过面板的 `/v1/*` 直接提供 OpenAI 与 Anthropic 兼容网关
- **任意 HF 量化模型**：输入任意 HuggingFace `repo_id`，探测仓库里全部 `.gguf` 变体（含量化级别与体积），选一个注册后与内置模型同权——下载、加载、对话、走 `/v1` 网关。**GGUF 在纯 CPU 实例上也能跑**
- **MCP 服务器**：内置 `/mcp`（streamable-http）把全部能力暴露成 25 个 MCP 工具，AI 客户端配个 URL + 面板令牌即可接管面板

## 模型工作台

「模型工作台」tab 把 `../easycolab/app.py`（Gradio WebUI + FastAPI 网关）上传到实例并后台拉起，从日志里抓 `*.gradio.live` 隧道地址。远端 app.py 的 FastAPI 路由会被隧道原样暴露，所以面板不需要直连 VM 端口。

| 模型 | 权重 | 显存需求 |
|---|---|---|
| MiniMax-H3 (33B) | 134.1 GB | 64.5 GB |
| Qwen3.8-27B (BF16) | 51.8 GB | 53.0 GB |
| Qwen3.8-27B-GGUF | 18.7 GB | 20.8 GB |
| Qwen-Image-2.1 | 30.9 GB | 24.0 GB |
| Z-Image-Turbo | 30.6 GB | 24.0 GB |

流程：选实例 → 一键部署 → 下载权重 → 加载 → 面板里在线生成，或用生成的 Base URL 调 `/v1/chat/completions`、`/v1/images/generations`。

### 拉取任意 HuggingFace 量化模型

内置目录里大多是"满血"权重（T4 的 16 GB 显存装不下）。模型工作台第三张卡片「自定义模型 · 从 HuggingFace 拉取任意 GGUF」解决这件事：

1. 填 `repo_id`（形如 `Qwen/Qwen2.5-3B-Instruct-GGUF`），gated 仓库再填 HF Token
2. 点「探测」→ 面板列出仓库里全部 `.gguf` 文件、量化级别（Q2_K / Q4_K_M / Q8_0 …）与体积，多模态仓库还会列出 `mmproj-*`
3. 选一个量化变体、选上下文长度（8K–128K），名字留空则自动按仓库名 + 量化级别生成
4. 点「注册」→ 模型进入清单，之后和内置模型完全一样：下载权重 → 加载 → 对话 / `/v1`

远端 `app.py` 里是 `hf_api_list_gguf_files()` + `register_custom_gguf_model()`，注册表落 `/content/custom_models.json`，重启后自动恢复。GGUF 走 llama.cpp 的 `llama-server`（自动下载对应后端：有 GPU 用 CUDA 12 版 + cudart，纯 CPU 用 23 MB 的 CPU 版），加载时读 GGUF 头部拿真实层数与原生上下文来算 `-ngl`：

- **有 GPU**：按显存余量决定 KV 量化（q8_0/q4_0）与卸载层数；装得下就全量驻留，装不下自动降级重试
- **纯 CPU 实例**：`-ngl 0` 全部层留在内存，KV 用 q8_0（失败退 f16）。3B-Q4_K_M 在 12.7 GB RAM 的 CPU 实例上加载 3.6 秒，单轮对话 ~9 秒

面板「在线生成」卡片直接走 `/api/easycolab/generate` + `/api/easycolab/job/{id}` 轮询：文生图、文生视频、对话/视觉理解三种模式，带进度条、停止按钮，完成后图片/视频内联渲染。`/api/easycolab/cancel` 对应远端的停止指令。

```bash
curl http://<面板>:8080/v1/models -H "Authorization: Bearer $PANEL_TOKEN"
curl http://<面板>:8080/v1/chat/completions?name=<实例名> \
  -H "Authorization: Bearer $PANEL_TOKEN" -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"hi"}]}'
```

`name` 可省略，但仅在恰好一个实例已部署服务时自动选中。

模型工作台需要 EasyColab 服务端源码在容器内 `/opt/easycolab/app.py`。源码随仓库分发在 `./easycolab/app.py`（已合入自定义 GGUF 支持），compose 默认就挂它，克隆下来即可用。想换别的副本时把 `EASYCOLAB_SRC_DIR` 指向宿主机绝对路径——**注意**：如果 Docker daemon 在远程或路径解析基准与 compose 客户端不同，相对路径会被 daemon 解析到别处，导致挂载成空目录。

## 部署

```bash
git clone https://github.com/host1syan/GoogleColabPanel.git
cd GoogleColabPanel
docker compose up -d --build
```

浏览器打开 `http://<服务器IP>:8080`，输入 API_TOKEN 进面板（口令见下）。

挂载路径默认相对 compose 文件（`./data` 与 `./easycolab`）。需要换位置时用环境变量覆盖，或写一个 `docker-compose.override.yml`（已被 `.gitignore` 忽略，适合放站点专属配置如外部网络）：

```yaml
services:
  colab-panel:
    networks: [my-net]
networks:
  my-net:
    external: true
```

```bash
PANEL_DATA_DIR=/srv/colab-data EASYCOLAB_SRC_DIR=/srv/easycolab docker compose up -d
```

**API_TOKEN**：默认无需设置——面板自动生成随机口令并保存到数据卷，容器重启不变。查看：

```bash
docker exec colab-panel cat /data/.config/colab-cli/panel_api_token
```

也可在 docker-compose.yml 的 environment 里显式指定自己的口令（显式设置优先于自动生成）。

## 使用流程

1. **绑定账号**：点「获取授权链接」→ 在任意设备的浏览器打开该链接登录 Google → 把授权页显示的授权码粘贴回面板提交
2. **建实例**：选 GPU 型号 →「新建实例」（分配约需 1~3 分钟）
3. **管理**：会话列表点「接管」→ 右侧可执行代码 / 生成网页链接 / 传文件 / 挂 Drive / 释放
4. **用完释放**：点「释放此实例」或「释放全部」，CU 立即停止计费

## MCP 接入（给 AI 客户端用）

面板内置一个 MCP 服务器，和 Web UI 同进程同端口，端点 `POST /mcp`（streamable-http）。
工具内部用 `httpx` 的 `ASGITransport` 直接打面板自己的 ASGI app，不开额外端口，反代 / 改绑定地址都不影响。
MCP 端点用**独立口令**，与面板 `API_TOKEN` 互不通用：拿到 MCP 口令的人拿不到面板，反之亦然。默认开启（`MCP_ENABLED=1`）。

**1. 取 MCP 口令和地址**

```bash
docker exec colab-panel cat /data/.config/colab-cli/mcp_api_key
```

该口令优先来自环境变量 `MCP_API_KEY`；未设置时首次启动自动生成 32 字符随机值并落盘（`0600`），容器重启不变。
面板自己的口令仍在 `/data/.config/colab-cli/panel_api_token`，只用于登 Web UI，**不能**用来连 `/mcp`。

端点就是面板地址后面加 `/mcp`：反代后是 `https://<你的域名>/mcp`，直连是 `http://<服务器IP>:8080/mcp`。
MCP 口令两种传法：请求头 `Authorization: Bearer <MCP_KEY>`，或查询串 `?token=<MCP_KEY>`（给不能自定义头的客户端）。

**2. 配到客户端**

Claude Code：

```bash
claude mcp add --transport http colab-panel \
  https://<你的域名>/mcp \
  --header "Authorization: Bearer <MCP_KEY>"
```

只想对当前目录生效时加 `--scope project`；用 `claude mcp list` 确认状态是 connected。

通用 JSON 配置（Claude 桌面版、Cursor 等支持远程 MCP 的客户端）：

```json
{
  "mcpServers": {
    "colab-panel": {
      "type": "http",
      "url": "https://<你的域名>/mcp",
      "headers": { "Authorization": "Bearer <MCP_KEY>" }
    }
  }
}
```

只支持 stdio 的旧版客户端，用 `mcp-remote` 桥接：

```json
{
  "mcpServers": {
    "colab-panel": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "https://<你的域名>/mcp",
               "--header", "Authorization: Bearer <MCP_KEY>"]
    }
  }
}
```

**3. 先自检再接入**

```bash
curl -sS https://<你的域名>/mcp \
  -H "Authorization: Bearer $MCP_KEY" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize",
       "params":{"protocolVersion":"2025-06-18","capabilities":{},
                 "clientInfo":{"name":"curl","version":"1"}}}'
```

返回里出现 `"serverInfo":{"name":"colab-panel"}` 即通。`401` 口令不对；`404` MCP 被关（`MCP_ENABLED=0`）；`421` Host 被 `MCP_ALLOWED_HOSTS` 拒绝。

**4. 首次接入流程**

1. `panel_status` 看登录态与 CU 余额
2. 没绑 Google 就 `panel_auth_url` → 把链接交给人登录 → `panel_auth_code` 提交授权码（这一步 agent 无法独立完成）
3. 临时跑代码用 `panel_run`：按需建实例、执行、跑完释放；默认 CPU，要显卡显式传 `gpu="t4"` / `l4` / `a100` / `h100` / `g4`
4. 实例按小时消耗 CU，用完调 `panel_stop` / `panel_stop_all`

**25 个工具**，分 7 组：

| 组 | 工具 | 说明 |
|---|---|---|
| 账号 | `panel_status` `panel_usage` `panel_auth_url` `panel_auth_code` | 登录态 / CU 余额 / OAuth 授权 |
| 实例 | `panel_sessions` `panel_create` `panel_stop` `panel_stop_all` `panel_ready` | 建、列、探活、释放 |
| 执行 | `panel_exec` `panel_exec_async` `panel_job` `panel_run` `panel_install` | 内核执行 / 后台任务 / 一次性跑 / pip |
| 文件 | `panel_upload` `panel_download` `panel_ls` `panel_rm` | 上传下载 / 列目录 / 删除 |
| 模型 | `panel_models` `panel_deploy` `panel_model` `panel_generate` | 模型目录 / 部署 / 下载加载 / 生成 |
| 网关 | `panel_chat` `panel_image` | `/v1` 对话与文生图 |
| 逃生口 | `panel_api` | 任意端点直调 |

约定：

- `panel_run` 默认 `gpu="cpu"`；要显卡传 `gpu="t4"` / `l4` / `a100` / `h100` / `g4`。`keep=false` 时跑完即释放。
- `panel_stop` / `panel_stop_all` / `panel_rm`、以及 `panel_api` 里的 DELETE / PUT 都需要 `confirm=true`；不带时只返回将影响的对象，不执行。
- `panel_download` 默认把文件落到容器内 `MCP_DOWNLOAD_DIR`（默认 `/data/mcp-downloads/`）并返回路径；`inline=true` 且不超过 2 MB 时才返回 base64。
- `panel_generate` 的 `prompt` 会映射到远端的必填字段 `visual_prompt`；`task_mode=chat` 时自动包成 `chat_messages`。

**环境变量**：

| 变量 | 默认 | 说明 |
|---|---|---|
| `MCP_ENABLED` | `1`（开启） | 设 `0` / `false` / `no` / `off` 关闭整个 `/mcp` 端点（关闭后访问返回 404） |
| `MCP_API_KEY` | 空（自动生成） | MCP 端点专属口令；不设时自动生成并写入 `/data/.config/colab-cli/mcp_api_key` |
| `MCP_ALLOWED_HOSTS` | 空（不校验 Host） | 逗号分隔的 Host 白名单，如 `colab.example.com`；设置后会拒绝其它 Host（421） |
| `MCP_DOWNLOAD_DIR` | `/data/mcp-downloads` | `panel_download` 的面板侧落盘目录 |

`docker-compose.yml` 里 `environment` 是映射，默认已写 `MCP_ENABLED: "1"` 与 `MCP_DOWNLOAD_DIR`；`MCP_API_KEY`、`MCP_ALLOWED_HOSTS`、`API_TOKEN` 是注释行，取消注释即可启用——不要再写成 `environment: []`，那会与键值行冲突导致 compose 解析失败。

不设 `MCP_ALLOWED_HOSTS` 时 DNS rebinding 校验关闭（否则内网 IP、反代域名都会被拒），此时口令是唯一门槛。
**`/mcp` 与面板同样请勿裸暴露公网**，建议放在反代 + 额外认证之后，或设置 `MCP_ALLOWED_HOSTS` 收窄来源。

## 数据持久化

Google OAuth 凭证与会话记录都在容器 `/data`：

```
/data/.config/colab-cli/token.json          # OAuth token（自动刷新）
/data/.config/colab-cli/sessions.json       # 本地会话表
/data/.config/colab-cli/panel_api_token     # 自动生成的面板口令（未显式设置 API_TOKEN 时）
/data/.config/colab-cli/mcp_api_key        # MCP 端点独立口令（未显式设置 MCP_API_KEY 时）
/data/.config/colab-cli/panel_easycolab.json # 各实例的 gradio.live 隧道地址
/data/mcp-downloads/                        # panel_download 落盘的文件（MCP_DOWNLOAD_DIR 可改）
```

`data/` 与 `.env` 已在 `.gitignore` 中——里面是 OAuth token 和面板口令，不要提交。

## 镜像发布（GitHub Actions）

工作流 `.github/workflows/docker-publish.yml`，只在**手动触发**或**推送 tag** 时运行，普通 push / PR 不构建。

**手动触发**：Actions → Build and Publish Image → Run workflow，页面上勾选要推送的库：

| 输入 | 默认 | 说明 |
|---|---|---|
| `push_ghcr` | 勾选 | 推送到 `ghcr.io/<owner>/googlecolabpanel` |
| `push_dockerhub` | 不勾 | 推送到 Docker Hub |
| `push_aliyun` | 不勾 | 推送到阿里云 ACR |
| `image_tag` | `latest` | 默认标签，留空即 `latest` |
| `platforms` | `linux/amd64,linux/arm64` | 可只选单架构 |

**tag 触发**：推 `v1.2.0` 这类 tag 时自动构建，只推 ghcr，默认标签取 tag 名。

每次构建产生**两个标签**：默认标签（`latest` / 指定的名字 / tag 名）+ UTC 时间标签 `YYYYMMDD-HHMMSS`。多架构由一次 buildx 构建合并成 manifest，`docker pull` 时自动按平台选择。

需要的仓库配置（Settings → Secrets and variables → Actions）：

```
Secrets:
  DOCKERHUB_USERNAME / DOCKERHUB_TOKEN     # 勾选 Docker Hub 时必填
  ALIYUN_USERNAME / ALIYUN_PASSWORD        # 勾选阿里云时必填
Variables:
  ALIYUN_REGISTRY   如 registry.cn-hangzhou.aliyuncs.com
  ALIYUN_NAMESPACE  如 ldxw
```

ghcr 用内置 `GITHUB_TOKEN`，无需额外配置。勾选了但没有对应凭据时，工作流在构建前直接报错，不会静默跳过。

用发布好的镜像部署：

```bash
docker run -d --name colab-panel -p 8080:8080 -v ./data:/data \
  ghcr.io/ldxw/googlecolabpanel:latest
```

## 已知限制

1. **面板无用户体系**，仅 API_TOKEN 一道门。自动生成的口令强度足够，但请勿裸暴露公网，建议反代 + 额外认证
2. **需要 Colab 付费 CU**：免费账号 GPU 分配会被 Colab 拒绝（可建 CPU 会话验证流程）
3. **ToS 风险**：headless 非网页方式使用 Colab 属灰色地带，参考上游声明——仅在自有付费账号上合理自用，账号风险自行评估
4. 新建实例名默认随机；`colab sessions` 输出为人类可读文本，面板按行解析，Colab 后端格式变化时以「CLI 原始输出」区为准
5. **gradio.live 隧道是临时的**（官方 best effort，最长约 1 周）。实例重启或隧道失效后重新点「一键部署」即可，面板会记录新地址
6. **模型要匹配实例显存**：MiniMax-H3 需 64.5 GB（只有 G4/H100 级实例够），Qwen3.8-27B BF16 需 53 GB。CPU 实例只能跑 GGUF（llama.cpp）；其他模型面板会直接回 409 说明原因
7. **GGUF 在 CPU 上慢**：3B-Q4_K_M 单轮 ~9 秒；7B 约 3–4 倍耗时，27B 不建议纯 CPU 跑
8. 首次部署若实例里没有 DiffSynth-Studio 会额外克隆并安装，耗时 3~10 分钟；Colab 默认镜像通常已带
