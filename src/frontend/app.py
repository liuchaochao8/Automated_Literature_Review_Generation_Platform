import streamlit as st
import httpx
import json

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="自动化文献综述生成平台",
    page_icon="📚",
    layout="wide",
)

# 初始化 session state
if "agent_statuses" not in st.session_state:
    st.session_state.agent_statuses = {}
if "review_result" not in st.session_state:
    st.session_state.review_result = None
if "generating" not in st.session_state:
    st.session_state.generating = False
if "last_topic" not in st.session_state:
    st.session_state.last_topic = ""

AGENTS = [
    ("expand_queries", "🔍 检索词扩展", "将主题扩展为多组检索词"),
    ("discover_papers", "📄 论文发现", "多库检索 + 相关性过滤"),
    ("categorize_papers", "📂 论文分类", "按方法论/时间线分类"),
    ("extract_information", "✂️ 信息提取", "提取问题/方法/数据集/结果"),
    ("analyze_relationships", "🔗 关系分析", "识别论文间继承/对比/矛盾"),
    ("synthesize_review", "✍️ 综述撰写", "生成综述正文"),
    ("format_references", "📝 引用格式化", "APA/IEEE 格式"),
    ("quality_check", "✅ 质量控制", "事实一致性检查"),
]


def render_agent_pipeline():
    """渲染 8 个 Agent 流水线状态"""
    cols = st.columns(8)
    for i, (key, label, desc) in enumerate(AGENTS):
        status_data = st.session_state.agent_statuses.get(key, {})
        status = status_data.get("status", "")
        summary = status_data.get("summary", "")

        with cols[i]:
            if status == "running":
                color, icon = "#FFA726", "⏳"
            elif status in ("done", "complete"):
                color, icon = "#66BB6A", "✅"
            else:
                color, icon = "#E0E0E0", "⏸️"

            st.markdown(f"""
            <div style="
                padding:10px 4px; border-radius:10px;
                background:{color}18; border:2px solid {color};
                text-align:center; min-height:95px;
            ">
                <div style="font-size:22px;">{icon}</div>
                <div style="font-size:11px; font-weight:700; margin:6px 0 2px;">{label}</div>
                <div style="font-size:9px; color:#666; line-height:1.3;">{summary or desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("关于")
    st.info("多 Agent 系统自动完成文献检索、分析、综述撰写。")
    api_url = st.text_input("API 地址", value=API_BASE)

# ---------- 主界面 ----------
st.title("自动化文献综述生成平台")
topic = st.text_input(
    "研究方向",
    placeholder="例如：Transformer 在时间序列预测中的应用",
    label_visibility="collapsed",
    disabled=st.session_state.generating,
)

col_btn, _ = st.columns([1, 5])
with col_btn:
    generate_btn = st.button(
        "🚀 生成综述",
        type="primary",
        use_container_width=True,
        disabled=not topic or st.session_state.generating,
    )

# ---------- 点击生成 ----------
if generate_btn and topic:
    st.session_state.agent_statuses = {}
    st.session_state.review_result = None
    st.session_state.generating = True
    st.session_state.last_topic = topic

# ---------- Agent 流水线（有状态时显示） ----------
if st.session_state.agent_statuses:
    render_agent_pipeline()

# ---------- 流式生成过程 ----------
if st.session_state.generating:
    status_placeholder = st.status("正在生成文献综述...", expanded=True)
    progress_bar = st.progress(0)

    try:
        with httpx.Client(timeout=600.0) as client:
            with client.stream(
                "POST",
                f"{api_url}/api/review/stream",
                json={"topic": topic or st.session_state.last_topic},
            ) as resp:
                if resp.status_code != 200:
                    status_placeholder.error(f"请求失败: {resp.text}")
                    st.session_state.generating = False
                    st.stop()

                step = 0
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    event = json.loads(line[6:])
                    agent = event.get("agent", "")

                    if agent == "start":
                        status_placeholder.write("🚀 开始生成...")
                        progress_bar.progress(5)

                    elif agent == "result":
                        st.session_state.review_result = event
                        progress_bar.progress(100)
                        status_placeholder.update(label="✅ 综述生成完成！", state="complete")
                        st.session_state.generating = False

                    else:
                        st.session_state.agent_statuses[agent] = event
                        step += 1
                        pct = min(5 + step * 11, 90)
                        progress_bar.progress(pct)
                        status_placeholder.write(f"  {event.get('summary', '')}")

    except Exception as e:
        status_placeholder.error(f"请求失败: {e}")
        st.session_state.generating = False

# ---------- 结果展示 ----------
result = st.session_state.review_result
if result:
    review = result.get("review", {})
    quality = result.get("quality_report", "")

    st.divider()
    st.header(review.get("title", "文献综述"))

    for section in review.get("sections", []):
        with st.expander(section.get("title", ""), expanded=True):
            st.markdown(section.get("content", ""))
            for sub in section.get("subsections", []):
                st.markdown(f"**{sub.get('title', '')}**")
                st.markdown(sub.get("content", ""))

    if quality:
        with st.expander("📋 质量报告"):
            st.markdown(quality)

    st.divider()
    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        review_json = json.dumps(review, ensure_ascii=False, indent=2)
        st.download_button(
            "📥 下载 JSON",
            data=review_json,
            file_name=f"{st.session_state.last_topic[:20]}_review.json",
            use_container_width=True,
        )

    with dl2:
        try:
            with httpx.Client(timeout=120.0) as c:
                r = c.post(f"{api_url}/api/review/latex", json={"topic": st.session_state.last_topic})
            if r.status_code == 200:
                st.download_button(
                    "📄 下载 LaTeX",
                    data=r.text,
                    file_name=f"{st.session_state.last_topic[:20]}_review.tex",
                    use_container_width=True,
                )
        except Exception:
            st.button("📄 LaTeX 下载失败", disabled=True, use_container_width=True)

    with dl3:
        if st.button("🔄 新查询", use_container_width=True):
            st.session_state.agent_statuses = {}
            st.session_state.review_result = None
            st.rerun()

    st.divider()

with st.expander("💡 使用提示", expanded=not st.session_state.review_result):
    st.markdown("""
    - 输入具体的研究方向效果更好，如 "Graph Neural Networks for Drug Discovery"
    - 生成过程通常需要 1–5 分钟
    - LaTeX 文件可用 Overleaf 或本地 TeX 环境编译为 PDF
    """)
