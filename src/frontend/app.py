import streamlit as st
import httpx
import json

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="文献综述生成平台",
    page_icon="○",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 初始化 session state
if "agent_statuses" not in st.session_state:
    st.session_state.agent_statuses = {}
if "agent_details" not in st.session_state:
    st.session_state.agent_details = {}
if "review_result" not in st.session_state:
    st.session_state.review_result = None
if "generating" not in st.session_state:
    st.session_state.generating = False
if "last_topic" not in st.session_state:
    st.session_state.last_topic = ""

STEPS = [
    "expand_queries", "discover_papers", "categorize_papers",
    "extract_information", "analyze_relationships", "synthesize_review",
    "format_references", "quality_check",
]

STEP_LABELS = {
    "expand_queries": "检索扩展",
    "discover_papers": "论文检索",
    "categorize_papers": "分类",
    "extract_information": "信息提取",
    "analyze_relationships": "关系分析",
    "synthesize_review": "综述撰写",
    "format_references": "格式化引用",
    "quality_check": "质量控制",
    "finalize": "完成",
}


def render_progress_indicator():
    """简洁的步骤进度条"""
    done = sum(1 for s in STEPS if st.session_state.agent_statuses.get(s, {}).get("status") in ("done", "complete"))
    total = len(STEPS)
    pct = done / total if total else 0

    st.markdown(
        f"""<div style="margin: 0 0 24px 0; height: 4px; background: #e9ecef; border-radius: 2px; overflow: hidden;">
            <div style="width: {pct * 100}%; height: 100%; background: #4a6cf7; border-radius: 2px; transition: width 0.3s;"></div>
        </div>""",
        unsafe_allow_html=True,
    )

    cols = st.columns(len(STEPS))
    for i, s in enumerate(STEPS):
        sts = st.session_state.agent_statuses.get(s, {}).get("status", "")
        label = STEP_LABELS[s]
        if sts in ("done", "complete"):
            color, bg, border = "#fff", "#4a6cf7", "#4a6cf7"
        elif sts == "running":
            color, bg, border = "#fff", "#6c757d", "#6c757d"
        else:
            color, bg, border = "#6c757d", "transparent", "#dee2e6"
        with cols[i]:
            st.markdown(
                f"""<div style="text-align:center; font-size:11px;">
                    <div style="width:22px; height:22px; line-height:22px; margin:0 auto 4px;
                        border-radius:50%; background:{bg}; border:2px solid {border};
                        color:{color}; font-size:11px; font-weight:600;">{"✓" if sts in ("done","complete") else ""}</div>
                    <div style="color:#495057; font-size:11px;">{label}</div>
                </div>""",
                unsafe_allow_html=True,
            )
    return done, total


# ---------- 主布局 ----------
st.markdown("""<style>
    .stApp { max-width: 960px; margin: 0 auto; }
    .stButton>button { font-size: 14px; }
    section[data-testid="stSidebar"] { width: 280px !important; }
</style>""", unsafe_allow_html=True)

st.title("文献综述生成")
st.markdown(f"""<p style="color:#6c757d; margin-top: -12px; font-size:14px;">
    输入研究方向，AI 自动完成文献检索、分析、综述撰写</p>""", unsafe_allow_html=True)

# ---------- 输入区 ----------
topic = st.text_input(
    "研究方向",
    placeholder="例如：Transformer in Time Series Forecasting",
    label_visibility="visible",
    disabled=st.session_state.generating,
)

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    generate_btn = st.button(
        "生成综述",
        type="primary",
        use_container_width=True,
        disabled=not topic or st.session_state.generating,
    )
with col2:
    sort_by = st.selectbox(
        "排序", index=0, label_visibility="collapsed",
        options=["citations", "newest", "relevance"],
        format_func=lambda x: {"citations": "按引用排序", "newest": "按年份排序", "relevance": "按相关度"}[x],
        disabled=st.session_state.generating,
    )

# ---------- 点击生成 ----------
if generate_btn and topic:
    st.session_state.agent_statuses = {}
    st.session_state.agent_details = {}
    st.session_state.review_result = None
    st.session_state.generating = True
    st.session_state.last_topic = topic

# ---------- 进度指示器 ----------
if st.session_state.agent_statuses:
    done, total = render_progress_indicator()

# ---------- 流式生成过程 ----------
if st.session_state.generating:
    placeholder = st.empty()

    with placeholder.container():
        status_box = st.status("正在生成文献综述...", expanded=True)
        progress_bar = st.progress(0)

    try:
        with httpx.Client(timeout=600.0) as client:
            with client.stream(
                "POST",
                f"{API_BASE}/api/review/stream",
                json={"topic": topic or st.session_state.last_topic, "sort_by": sort_by},
            ) as resp:
                if resp.status_code != 200:
                    status_box.error(f"请求失败: {resp.text}")
                    st.session_state.generating = False
                    st.stop()

                step = 0
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    event = json.loads(line[6:])
                    agent = event.get("agent", "")

                    if agent == "start":
                        status_box.write("开始生成...")
                        progress_bar.progress(5)

                    elif agent == "result":
                        st.session_state.review_result = event
                        progress_bar.progress(100)
                        status_box.update(label="综述生成完成", state="complete")
                        st.session_state.generating = False

                    else:
                        st.session_state.agent_statuses[agent] = event
                        details = event.get("details", [])
                        if details:
                            st.session_state.agent_details[agent] = details
                        step += 1
                        pct = min(5 + step * 11, 90)
                        progress_bar.progress(pct)
                        summary = event.get("summary", "")

                        status_box.write(summary)

                        if agent == "expand_queries" and details:
                            for q in details[:10]:
                                status_box.write(f"  {q[:120]}")

                        elif agent == "discover_papers" and details:
                            for p in details[:15]:
                                items = []
                                if p.get("year"):
                                    items.append(str(p["year"]))
                                if p.get("citations"):
                                    items.append(f"[{p['citations']}]")
                                if p.get("source"):
                                    items.append(p["source"])
                                tag = f" ({', '.join(items)})" if items else ""
                                status_box.write(f"  {p['title'][:80]}{tag}")

                        elif agent == "categorize_papers" and details:
                            for cat in details[:10]:
                                status_box.write(f"  {cat}")

                        elif agent == "extract_information" and details:
                            for ext in details[:10]:
                                status_box.write(f"  {ext[:120]}")

                        elif agent == "analyze_relationships" and details:
                            for r in details[:15]:
                                status_box.write(f"  [{r['type']}] {r.get('description', '')[:100]}")

                        elif agent == "synthesize_review":
                            if details:
                                for sec in details[:10]:
                                    status_box.write(f"  {sec[:100]}")
                            elif summary:
                                status_box.write(f"  {summary[:80]}")

                        elif agent == "format_references" and details:
                            for ref in details[:10]:
                                status_box.write(f"  {ref[:100]}")

                        elif agent == "quality_check":
                            report = event.get("details", summary)
                            if isinstance(report, str) and report:
                                status_box.write(f"  {report[:200]}")

    except Exception as e:
        status_box.error(f"连接中断: {e}")
        st.session_state.generating = False

# ---------- Agent 详细数据 ----------
if st.session_state.agent_details and not st.session_state.generating:
    with st.expander(f"检索词（{len(st.session_state.agent_details.get('expand_queries', []))} 组）"):
        for q in st.session_state.agent_details.get("expand_queries", []):
            st.code(q, language="text")

    papers = st.session_state.agent_details.get("discover_papers", [])
    if papers:
        with st.expander(f"论文列表（{len(papers)} 篇）"):
            for p in papers:
                parts = []
                if p.get("year"):
                    parts.append(str(p["year"]))
                if p.get("citations"):
                    parts.append(f"{p['citations']} 引用")
                if p.get("source"):
                    parts.append(p["source"])
                tag = f" ({', '.join(parts)})" if parts else ""
                st.markdown(f"- **{p['title']}**{tag}")

    rels = st.session_state.agent_details.get("analyze_relationships", [])
    if rels:
        with st.expander(f"论文关系（{len(rels)} 组）"):
            for r in rels:
                st.markdown(f"- **{r['type']}**: {r.get('description', '')}")

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
        with st.expander("质量报告"):
            st.markdown(quality)

    st.divider()
    dl1, dl2, dl3, dl4 = st.columns(4)

    with dl1:
        review_json = json.dumps(review, ensure_ascii=False, indent=2)
        st.download_button(
            "JSON",
            data=review_json,
            file_name=f"{st.session_state.last_topic[:20]}_review.json",
            use_container_width=True,
        )

    with dl2:
        try:
            with httpx.Client(timeout=30.0) as c:
                r = c.post(f"{API_BASE}/api/review/markdown/from-json", json={"review": review})
            if r.status_code == 200:
                st.download_button(
                    "Markdown",
                    data=r.text,
                    file_name=f"{st.session_state.last_topic[:20]}_review.md",
                    use_container_width=True,
                )
        except Exception:
            st.button("MD 失败", disabled=True, use_container_width=True)

    with dl3:
        try:
            with httpx.Client(timeout=30.0) as c:
                r = c.post(f"{API_BASE}/api/review/latex/from-json", json={"review": review})
            if r.status_code == 200:
                st.download_button(
                    "LaTeX",
                    data=r.text,
                    file_name=f"{st.session_state.last_topic[:20]}_review.tex",
                    use_container_width=True,
                )
        except Exception:
            st.button("LaTeX 失败", disabled=True, use_container_width=True)

    with dl4:
        if st.button("新查询", use_container_width=True):
            st.session_state.agent_statuses = {}
            st.session_state.review_result = None
            st.rerun()

st.divider()
st.markdown(f"""<p style="color:#adb5bd; font-size:12px; text-align:center;">
    API: {API_BASE} · 生成耗时取决于主题和论文数量</p>""", unsafe_allow_html=True)
