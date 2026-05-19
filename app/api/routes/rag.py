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


@router.post(
    "/stream",
    summary="RAG 질의응답 (스트리밍)",
    description="RAG 응답을 SSE 스트림으로 반환합니다.",
    response_class=StreamingResponse,
)
async def rag_stream(body: RagInput, chain=Depends(get_rag_chain)):
    logger.info(f"[rag/stream] 요청: '{body.question[:80]}'")

    async def generator():
        try:
            total = 0
            async for chunk in chain.astream(body.question):
                total += len(chunk)
                yield f"data: {chunk}\n\n"
            logger.info(f"[rag/stream] 스트림 완료 (누적 길이: {total})")
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"[rag/stream] 오류: {e}")
            yield f"data: [ERROR] {e}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")
