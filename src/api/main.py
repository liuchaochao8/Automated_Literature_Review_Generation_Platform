import json
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.orchestrator.workflow import LiteratureReviewWorkflow
from src.models.schemas import LiteratureReview
from src.services.logging_setup import setup_logging
from src.services.latex_export import review_to_latex, review_to_markdown
from src.config.settings import settings

logger = setup_logging("api")

app = FastAPI(title="Automated Literature Review Platform", version="0.1.0")


class ReviewRequest(BaseModel):
    topic: str
    sort_by: str = "citations"  # "citations", "newest", "relevance"


class ReviewResponse(BaseModel):
    review: dict
    quality_report: str = ""
    progress: dict = {}
    errors: list[str] = []


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "literature-review-platform"}


@app.post("/api/review", response_model=ReviewResponse)
async def generate_review(request: ReviewRequest):
    if not request.topic.strip():
        raise HTTPException(status_code=422, detail="Topic cannot be empty")

    workflow = LiteratureReviewWorkflow()
    try:
        result = await workflow.run(topic=request.topic, sort_by=request.sort_by)
        review_dict = _serialize_review(result.get("review"))
        _save_review(review_dict, request.topic)

        return ReviewResponse(
            review=review_dict,
            quality_report=result.get("quality_report", ""),
            progress=result.get("progress", {}),
            errors=result.get("errors", []),
        )
    except Exception as e:
        logger.exception("Review generation failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/review/stream")
async def generate_review_stream(request: ReviewRequest):
    """SSE 流式接口，前端可实时看到每个 agent 的执行状态。"""
    if not request.topic.strip():
        raise HTTPException(status_code=422, detail="Topic cannot be empty")

    workflow = LiteratureReviewWorkflow()

    async def event_stream():
        # 先发送开始事件
        yield f"data: {json.dumps({'agent': 'start', 'status': 'running', 'summary': '开始生成文献综述'})}\n\n"

        async for event in workflow.run_stream(topic=request.topic, sort_by=request.sort_by):
            try:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as e:
                logger.error("序列化事件失败: %s", e)
                yield f"data: {json.dumps({'agent': 'error', 'status': 'failed', 'summary': f'事件序列化失败: {e}'})}\n\n"
                return

            # 在 result 事件时保存综述
            if event.get("agent") == "result":
                try:
                    review_dict = _serialize_review(event.get("review"))
                    _save_review(review_dict, request.topic)
                except Exception as e:
                    logger.error("保存综述失败: %s", e)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


class LatexFromReviewRequest(BaseModel):
    review: dict


@app.post("/api/review/latex/from-json")
async def convert_review_to_latex(request: LatexFromReviewRequest):
    """将已有的 review JSON 转为 LaTeX，不重新跑 workflow。"""
    try:
        review_obj = LiteratureReview(**request.review)
        latex = review_to_latex(review_obj)
        title = request.review.get("title", "review")[:30]
        return StreamingResponse(
            iter([latex]),
            media_type="application/x-tex",
            headers={"Content-Disposition": f'attachment; filename="{title}.tex"'},
        )
    except Exception as e:
        logger.exception("LaTeX conversion failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/review/markdown/from-json")
async def convert_review_to_markdown(request: LatexFromReviewRequest):
    """将已有的 review JSON 转为 Markdown，不重新跑 workflow。"""
    try:
        review_obj = LiteratureReview(**request.review)
        md = review_to_markdown(review_obj)
        title = request.review.get("title", "review")[:30]
        return StreamingResponse(
            iter([md]),
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{title}.md"'},
        )
    except Exception as e:
        logger.exception("Markdown conversion failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/review/latex")
async def generate_review_latex(request: ReviewRequest):
    """返回 LaTeX 格式的综述"""
    if not request.topic.strip():
        raise HTTPException(status_code=422, detail="Topic cannot be empty")

    workflow = LiteratureReviewWorkflow()
    try:
        result = await workflow.run(topic=request.topic, sort_by=request.sort_by)
        review = result.get("review")
        review_dict = _serialize_review(review)
        _save_review(review_dict, request.topic)
        latex = review_to_latex(review)
        return StreamingResponse(
            iter([latex]),
            media_type="application/x-tex",
            headers={"Content-Disposition": f'attachment; filename="{request.topic[:30]}_review.tex"'},
        )
    except Exception as e:
        logger.exception("LaTeX generation failed")
        raise HTTPException(status_code=500, detail=str(e))


def _save_review(review_dict: dict, topic: str) -> None:
    """将生成的综述保存到 logs/reviews/ 目录（json + tex）。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = "".join(c if c.isalnum() or c in " _-" else "_" for c in topic)[:30]
    review_dir = Path(settings.log_dir) / "reviews"
    review_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = review_dir / f"{timestamp}_{safe_topic}.json"
    json_path.write_text(json.dumps(review_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    # LaTeX
    try:
        review_obj = LiteratureReview(**review_dict) if isinstance(review_dict, dict) else review_dict
        latex = review_to_latex(review_obj)
        tex_path = review_dir / f"{timestamp}_{safe_topic}.tex"
        tex_path.write_text(latex, encoding="utf-8")
    except Exception as e:
        logger.warning(f"保存 LaTeX 失败: {e}")

    logger.info("综述已保存: %s", json_path)


def _serialize_review(review) -> dict:
    if hasattr(review, "model_dump"):
        return review.model_dump()
    if isinstance(review, dict):
        return review
    return {}
