from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from api.schemas import RagInput, RagOutput
from api.dependencies import get_rag_chain
from core.exceptions import InternalServerError

router = APIRouter(prefix="/rag", tags=["RAG 검색"])


@router.post(
    "",
    response_model=RagOutput,
    summary="RAG 질의응답",
    description="벡터 인덱스와 키워드 검색을 결합한 하이브리드 RAG로 질문에 답합니다.",
)
async def rag_query(body: RagInput, chain=Depends(get_rag_chain)):
    try:
        result = await chain.ainvoke(body.question)
        return RagOutput(output=result)
    except Exception as e:
        raise InternalServerError(detail=str(e))


@router.post(
    "/stream",
    summary="RAG 질의응답 (스트리밍)",
    description="RAG 응답을 SSE 스트림으로 반환합니다.",
    response_class=StreamingResponse,
)
async def rag_stream(body: RagInput, chain=Depends(get_rag_chain)):
    async def generator():
        try:
            async for chunk in chain.astream(body.question):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {e}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")
