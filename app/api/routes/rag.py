import json
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from api.schemas import RagInput, RagOutput
from api.dependencies import get_rag_chain
from core.exceptions import InternalServerError
from core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/rag", tags=["RAG 검색"])


@router.post(
    "",
    response_model=RagOutput,
    summary="RAG 질의응답",
    description="벡터 인덱스와 키워드 검색을 결합한 하이브리드 RAG로 질문에 답합니다.",
)
async def rag_query(body: RagInput, chain=Depends(get_rag_chain)):
    logger.info(f"[rag] 요청: '{body.question[:80]}'")
    try:
        result = await chain.ainvoke(body.question)
        logger.info(f"[rag] 응답 완료 (길이: {len(result)})")
        return RagOutput(output=result)
    except Exception as e:
        logger.error(f"[rag] 오류: {e}")
        raise InternalServerError(detail=str(e))


_STREAM_DESCRIPTION = """
RAG 응답을 SSE(Server-Sent Events) 스트림으로 반환합니다.

**이벤트 포맷** (`data: <JSON>\\n\\n`)

| type | 설명 | 페이로드 예시 |
|------|------|--------------|
| `text` | 답변 텍스트 청크 | `{"type":"text","content":"안녕하세요"}` |
| `error` | 오류 발생 | `{"type":"error","message":"오류 내용"}` |
"""


@router.post(
    "/stream",
    summary="RAG 질의응답 (스트리밍)",
    description=_STREAM_DESCRIPTION,
    response_class=StreamingResponse,
)
async def rag_stream(body: RagInput, chain=Depends(get_rag_chain)):
    logger.info(f"[rag/stream] 요청: '{body.question[:80]}'")

    async def generator():
        try:
            total = 0
            async for chunk in chain.astream(body.question):
                total += len(chunk)
                yield f"data: {json.dumps({'type': 'text', 'content': chunk}, ensure_ascii=False)}\n\n"
            logger.info(f"[rag/stream] 스트림 완료 (누적 길이: {total})")
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            logger.error(f"[rag/stream] 오류: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")
