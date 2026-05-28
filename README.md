---
title: Automated Literature Review Generation Platform
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 自动化文献综述生成平台

输入一个研究主题，系统通过 8 个 AI Agent 自动完成文献检索、分类、信息提取、关系分析和综述撰写，最终输出结构化 JSON 或可编译的 LaTeX 文档。

---

👉 在线体验：https://huggingface.co/spaces/genius-lc/Automated_Literature_Review_Generation_Platform

---

## 技术栈

| 组件       | 技术                                             |
| ---------- | ------------------------------------------------ |
| 语言模型   | DeepSeek V4 Flash（OpenAI 兼容 SDK）             |
| Agent 编排 | LangGraph（StateGraph + 条件边 + 流式更新）      |
| API 服务   | FastAPI（REST + SSE 流式）                       |
| 前端界面   | Streamlit（Agent 流水线可视化）                  |
| 向量存储   | ChromaDB（论文嵌入与相似度检索）                 |
| 引用图谱   | NetworkX（DiGraph 关系分析）                     |
| 学术数据源 | arXiv OAI-PMH / OpenAlex API / CrossRef REST API |
| 文档导出   | LaTeX（`\section` / `\subsection` 递归生成）     |

## 架构

```
用户输入主题
    │
    ▼
QueryExpansionAgent  →  PaperDiscoveryAgent  →  CategorizationAgent
    │                       │                       │
    └─────── 8 个 Agent 通过 LangGraph 编排 ───────┘
                        │
                        ▼
    ExtractionAgent  →  ComparisonAgent  →  SynthesisAgent
    │                       │                       │
    └──────────── ReferenceAgent → QualityAgent ────┘
                                        │
                                  质量不合格 → 重新合成
                                  质量合格 → 输出综述
```

### Agent 职责

| Agent                   | 输入        | 输出                                             |
| ----------------------- | ----------- | ------------------------------------------------ |
| **QueryExpansionAgent** | 用户主题    | 多组扩展检索词（子查询 + 同义词）                |
| **PaperDiscoveryAgent** | 检索词      | 多库并行搜索 + LLM 相关性评分（阈值 0.3）        |
| **CategorizationAgent** | 论文列表    | 方法论 / 时间线 / 范式 / 影响力分类              |
| **ExtractionAgent**     | 论文        | 研究问题、方法、数据集、关键结果、局限性         |
| **ComparisonAgent**     | 论文对      | 5 类关系：继承/对比/矛盾/扩展/并行               |
| **SynthesisAgent**      | 以上全部    | 带章节结构的综述正文（Markdown → ReviewSection） |
| **ReferenceAgent**      | 综述 + 论文 | APA / IEEE 格式引用列表                          |
| **QualityAgent**        | 综述 + 论文 | 事实一致性检查 + 结构完整性检查                  |

质量控制节点为条件边：若检测到严重问题（`"critical"` 关键词），自动回退到合成阶段重新生成。

## 快速开始

### 1. 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. 配置

在项目根目录创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

> DeepSeek API 密钥从 [platform.deepseek.com](https://platform.deepseek.com) 获取。
>
> 可选配置项（`.env`）：
>
> | 变量                | 默认值                     | 说明       |
> | ------------------- | -------------------------- | ---------- |
> | `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API 端点   |
> | `LLM_MODEL`         | `deepseek-v4-flash`        | 模型名称   |
> | `LOG_LEVEL`         | `INFO`                     | 日志级别   |
> | `MAX_PAPERS`        | `50`                       | 最大论文数 |

### 3. 启动 API 服务

```bash
source .venv/bin/activate
# 开发模式（--reload：修改代码自动重启，但会中断正在进行的流式请求）
uvicorn src.api.main:app --reload --timeout-keep-alive 600

# 生产模式（推荐：稳定处理长时间流式请求）
uvicorn src.api.main:app --host 0.0.0.0 --timeout-keep-alive 600 --limit-concurrency 10
```

> **注意**：`--reload` 会在文件变化时重启服务，可能中断正在运行的长任务（如流式综述生成）。
> 生成过程通常需 1-3 分钟，请勿在生成期间修改代码文件。

API 默认在 `http://localhost:8000` 运行。

### 4. 启动前端界面（可选）

```bash
streamlit run src/frontend/app.py
```

### 5. 调用

```bash
# 普通调用 —— 返回完整综述 JSON
curl -X POST "http://localhost:8000/api/review" \
  -H "Content-Type: application/json" \
  -d '{"topic": "Transformer 在时间序列预测中的应用"}'

# 流式调用 —— 实时查看每个 agent 的执行状态
curl -X POST "http://localhost:8000/api/review/stream" \
  -H "Content-Type: application/json" \
  -d '{"topic": "Transformer 在时间序列预测中的应用"}'

# 导出 LaTeX 文件
curl -X POST "http://localhost:8000/api/review/latex" \
  -H "Content-Type: application/json" \
  -d '{"topic": "Transformer 在时间序列预测中的应用"}' \
  -o review.tex
```

## API 接口

| 方法   | 路径                 | 说明                            |
| ------ | -------------------- | ------------------------------- |
| `GET`  | `/health`            | 健康检查                        |
| `POST` | `/api/review`        | 完整综述生成，返回 JSON         |
| `POST` | `/api/review/stream` | SSE 流式接口，逐 agent 推送状态 |
| `POST` | `/api/review/latex`  | LaTeX 格式导出                  |

### SSE 事件格式

流式接口推送的每行数据格式为 `data: {json}\n\n`，事件类型：

| agent 字段              | 说明                                                  |
| ----------------------- | ----------------------------------------------------- |
| `start`                 | 开始生成                                              |
| `expand_queries`        | 检索词扩展完成                                        |
| `discover_papers`       | 论文发现完成                                          |
| `categorize_papers`     | 论文分类完成                                          |
| `extract_information`   | 信息提取完成                                          |
| `analyze_relationships` | 关系分析完成                                          |
| `synthesize_review`     | 综述撰写完成                                          |
| `format_references`     | 引用格式化完成                                        |
| `quality_check`         | 质量控制完成                                          |
| `result`                | 完整综述结果（含 review / quality_report / progress） |

每个事件包含 `agent`、`status`、`summary` 字段，前端据此更新 Agent 卡片状态。

## 日志与持久化

| 内容              | 路径                                      | 格式     |
| ----------------- | ----------------------------------------- | -------- |
| 运行日志          | `logs/api_YYYYMMDD_HHMMSS.log`            | 文本日志 |
| 生成综述（JSON）  | `logs/reviews/YYYYMMDD_HHMMSS_topic.json` | JSON     |
| 生成综述（LaTeX） | `logs/reviews/YYYYMMDD_HHMMSS_topic.tex`  | LaTeX    |

三种调用方式（REST / 流式 / LaTeX）均会自动保存生成的综述到 `logs/reviews/` 目录。

## Agent 工作流可视化

打开前端后，生成综述过程中 8 个 Agent 状态实时更新。

## 项目结构

```
Automated_Literature_Review_Generation_Platform/
├── .env                    # 本地配置（已 gitignore，不上传）
├── .env.example            # 配置模板（占位符，可上传）
├── pyproject.toml          # 项目元数据与依赖
├── README.md               # 本文档
│
├── src/
│   ├── config/
│   │   └── settings.py                 # Pydantic BaseSettings 配置管理
│   │
│   ├── models/
│   │   ├── schemas.py                  # 数据模型（Paper, LiteratureReview 等 8 个 Pydantic 模型）
│   │   └── state.py                    # LangGraph TypedDict 状态定义
│   │
│   ├── services/
│   │   ├── llm_client.py               # DeepSeek LLM 客户端（tenacity 重试）
│   │   ├── academic_apis.py            # arXiv / OpenAlex / CrossRef 搜索 + 去重
│   │   ├── vector_store.py             # ChromaDB 向量存储（cosine 检索）
│   │   ├── graph_store.py              # NetworkX 引用关系有向图
│   │   ├── latex_export.py             # LaTeX 源码递归生成
│   │   └── logging_setup.py            # 日志配置（文件 + 终端 + 第三方日志抑制）
│   │
│   ├── agents/
│   │   ├── query_expansion.py          # 主题 → 多组扩展检索词
│   │   ├── paper_discovery.py          # 多库并行检索 + 相关性评分过滤
│   │   ├── categorization.py           # 方法论 / 时间线 / 研究范式分类
│   │   ├── extraction.py               # 论文信息提取（问题、方法、数据、结果）
│   │   ├── comparison.py               # 论文间 5 类关系分析（O(n²) 组合）
│   │   ├── synthesis.py                # 多源数据 → 综述正文（Markdown 解析）
│   │   ├── reference.py                # APA / IEEE 引用格式化
│   │   └── quality.py                  # 事实一致性 + 结构完整性检查
│   │
│   ├── orchestrator/
│   │   └── workflow.py                 # LangGraph StateGraph 编排（9 节点 + 条件边）
│   │
│   ├── api/
│   │   └── main.py                     # FastAPI 应用（4 个端点）
│   │
│   └── frontend/
│       └── app.py                      # Streamlit 界面（Agent 流水线 + 结果展示）
│
├── tests/
│   ├── conftest.py                     # pytest 夹具（mock LLM 客户端）
│   ├── test_agents/
│   ├── test_services/
│   └── test_orchestrator/
│
└── docs/
    └── superpowers/
        ├── plans/                      # 实现计划文档
        └── skills/                     # Superpowers 技能定义
```

## 注意事项

- `.env` 文件包含真实的 API 密钥，已加入 `.gitignore`，不会提交到 Git。配置模板请使用 `.env.example`（已使用占位符替换密钥）。
- arXiv 等学术 API 返回 502 时，系统会自动重试（最多 3 次，指数退避）。
- 生成过程通常需要 1-5 分钟，取决于主题范围和返回的论文数量。
- 所有第三方 HTTP 客户端（httpx、chromadb、openai）的日志默认设为 `WARNING` 级别，减少干扰。
