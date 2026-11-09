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
- **模型工作台**：一键把 EasyColab 服务端部署到实例，管理 5 个模型的权重下载 / 加载显存 / 卸载，调上下文长度与 API Key，起停 ComfyUI，并通过面板的 `/v1/*` 直接提供 OpenAI 与 Anthropic 兼容网关

## 模型工作台

「模型工作台」tab 把 `../easycolab/app.py`（Gradio WebUI + FastAPI 网关）上传到实例并后台拉起，从日志里抓 `*.gradio.live` 隧道地址。远端 app.py 的 FastAPI 路由会被隧道原样暴露，所以面板不需要直连 VM 端口。

| 模型 | 权重 | 显存需求 |
|---|---|---|
| MiniMax-H3 (33B) | 134.1 GB | 64.5 GB |
| Qwen3.8-27B (BF16) | 51.8 GB | 53.0 GB |
| Qwen3.8-27B-GGUF | 18.7 GB | 20.8 GB |
| Qwen-Image-2.1 | 30.9 GB | 24.0 GB |
| Z-Image-Turbo | 30.6 GB | 24.0 GB |

流程：选实例 → 一键部署 → 下载权重 → 加载显存 → 面板里在线生成，或用生成的 Base URL 调 `/v1/chat/completions`、`/v1/images/generations`。

面板「在线生成」卡片直接走 `/api/easycolab/generate` + `/api/easycolab/job/{id}` 轮询：文生图、文生视频、对话/视觉理解三种模式，带进度条、停止按钮，完成后图片/视频内联渲染。`/api/easycolab/cancel` 对应远端的停止指令。

```bash
curl http://<面板>:8080/v1/models -H "Authorization: Bearer $PANEL_TOKEN"
curl http://<面板>:8080/v1/chat/completions?name=<实例名> \
  -H "Authorization: Bearer $PANEL_TOKEN" -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"hi"}]}'
```

`name` 可省略，但仅在恰好一个实例已部署服务时自动选中。

部署依赖 EasyColab 源码在容器内 `/opt/easycolab/app.py`（compose 默认挂载 `../easycolab`）。**注意**：如果你的 Docker daemon 在远程或路径解析基准与 compose 客户端不同，相对路径会被 daemon 解析到别处，导致挂载成空目录——这种情况把 `EASYCOLAB_SRC_DIR` 设成宿主机绝对路径。

## 部署

```bash
git clone https://github.com/host1syan/GoogleColabPanel.git
cd GoogleColabPanel
docker compose up -d --build
```

浏览器打开 `http://<服务器IP>:8080`，输入 API_TOKEN 进面板（口令见下）。

挂载路径默认相对 compose 文件（`./data` 与 `../easycolab`）。需要换位置时用环境变量覆盖，或写一个 `docker-compose.override.yml`（已被 `.gitignore` 忽略，适合放站点专属配置如外部网络）：

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

## 数据持久化

Google OAuth 凭证与会话记录都在容器 `/data`：

```
/data/.config/colab-cli/token.json          # OAuth token（自动刷新）
/data/.config/colab-cli/sessions.json       # 本地会话表
/data/.config/colab-cli/panel_api_token     # 自动生成的面板口令（未显式设置 API_TOKEN 时）
/data/.config/colab-cli/panel_easycolab.json # 各实例的 gradio.live 隧道地址
```

`data/` 与 `.env` 已在 `.gitignore` 中——里面是 OAuth token 和面板口令，不要提交。

## 已知限制

1. **面板无用户体系**，仅 API_TOKEN 一道门。自动生成的口令强度足够，但请勿裸暴露公网，建议反代 + 额外认证
2. **需要 Colab 付费 CU**：免费账号 GPU 分配会被 Colab 拒绝（可建 CPU 会话验证流程）
3. **ToS 风险**：headless 非网页方式使用 Colab 属灰色地带，参考上游声明——仅在自有付费账号上合理自用，账号风险自行评估
4. 新建实例名默认随机；`colab sessions` 输出为人类可读文本，面板按行解析，Colab 后端格式变化时以「CLI 原始输出」区为准
5. **gradio.live 隧道是临时的**（官方 best effort，最长约 1 周）。实例重启或隧道失效后重新点「一键部署」即可，面板会记录新地址
6. **模型要匹配实例显存**：MiniMax-H3 需 64.5 GB（只有 G4/H100 级实例够），Qwen3.8-27B BF16 需 53 GB。CPU 实例可以部署服务端但不能加载模型，面板会直接回 409 说明原因
7. 首次部署若实例里没有 DiffSynth-Studio 会额外克隆并安装，耗时 3~10 分钟；Colab 默认镜像通常已带
