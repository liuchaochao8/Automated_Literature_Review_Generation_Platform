# 自动化文献综述生成平台

输入一个研究主题，系统自动完成文献检索、分类、信息提取、关系分析和综述撰写。

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

## 快速开始

### 1. 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. 配置

在 `.env` 中设置 API key（已预填）：

```env
DEEPSEEK_API_KEY=sk-a22cf174fa5f4d4dad608324f7c3a1a9
```

### 3. 启动 API

```bash
source .venv/bin/activate
uvicorn src.api.main:app --reload
```

### 4. 启动前端（可选）

```bash
streamlit run src/frontend/app.py
```

### 5. 调用

```bash
# 普通调用
curl -X POST "http://localhost:8000/api/review" \
  -H "Content-Type: application/json" \
  -d '{"topic": "Transformer 在时间序列预测中的应用"}'

# 流式调用（实时查看 agent 执行状态）
curl -X POST "http://localhost:8000/api/review/stream" \
  -H "Content-Type: application/json" \
  -d '{"topic": "Transformer 在时间序列预测中的应用"}'

# 导出 LaTeX
curl -X POST "http://localhost:8000/api/review/latex" \
  -H "Content-Type: application/json" \
  -d '{"topic": "Transformer 在时间序列预测中的应用"}' \
  -o review.tex
```

## 日志

每次运行自动保存日志到 `logs/` 目录，文件名格式：`api_20260526_143022.log`

## 输出格式

| 格式 | 说明 | 获取方式 |
|------|------|---------|
| JSON | 综述结构化数据（章节、内容、引用） | `POST /api/review` |
| LaTeX | 可编译的 `.tex` 源码 | `POST /api/review/latex` |
| 流式 SSE | 实时 agent 状态推送 | `POST /api/review/stream` |

## Agent 工作流可视化

打开前端（`streamlit run src/frontend/app.py`）后，生成综述过程中会实时显示 8 个 Agent 的流水线状态：

```
🔍检索词扩展 → 📄论文发现 → 📂论文分类 → ✂️信息提取 → 🔗关系分析 → ✍️综述撰写 → 📝引用格式化 → ✅质量控制
```

每个 Agent 卡片会实时更新：⏸️等待中 → ⏳执行中（显示摘要）→ ✅已完成

## 项目结构

```
src/
├── config/settings.py           # 配置管理
├── models/
│   ├── schemas.py               # 数据模型（Paper、LiteratureReview 等）
│   └── state.py                 # LangGraph 状态定义
├── services/
│   ├── llm_client.py            # DeepSeek LLM 客户端
│   ├── academic_apis.py         # arXiv、Semantic Scholar、CrossRef 搜索
│   ├── vector_store.py          # ChromaDB 向量存储
│   ├── graph_store.py           # NetworkX 引用关系图谱
│   ├── latex_export.py          # LaTeX 源码生成
│   └── logging_setup.py         # 日志配置（文件 + 终端）
├── agents/
│   ├── query_expansion.py       # 主题→检索词扩展
│   ├── paper_discovery.py       # 多库论文检索与相关性过滤
│   ├── categorization.py        # 论文分类（方法、时间线、范式）
│   ├── extraction.py            # 信息提取（问题、方法、数据集、结果）
│   ├── comparison.py            # 论文间关系分析
│   ├── synthesis.py             # 综述撰写
│   ├── reference.py             # 引用格式化（APA/IEEE）
│   └── quality.py               # 事实一致性与结构检查
├── orchestrator/workflow.py     # LangGraph 工作流编排
├── api/main.py                  # FastAPI 服务（REST + SSE 流式）
├── orchestrator/workflow.py     # LangGraph 工作流编排
└── frontend/app.py              # Streamlit 界面（Agent 可视化）
```

## 测试

```bash
pytest tests/ -v
```
