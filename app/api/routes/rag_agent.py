import json
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from api.schemas import RagAgentInput, RagAgentOutput, SourceItem
from api.dependencies import get_rag_agent_chain
from core.exceptions import InternalServerError

router = APIRouter(prefix="/rag/agent", tags=["RAG 에이전트"])

_ROLE_MAP = {
    "human": HumanMessage,
    "ai": AIMessage,
    "system": SystemMessage,
}


def _to_lc_messages(items):
    return [_ROLE_MAP[m.role](content=m.content) for m in items]


def _to_source_items(sources: list[tuple[str, str]]) -> list[SourceItem]:
    return [SourceItem(name=name, page=page or None) for name, page in sources]


@router.post(
    "",
    response_model=RagAgentOutput,
    summary="RAG 에이전트 질의응답",
    description="LLM이 필요 시 search_documents tool을 호출해 문서를 검색하고 답변합니다. 이전 대화 기록을 포함한 멀티턴을 지원합니다.",
)
async def rag_agent(body: RagAgentInput, chain=Depends(get_rag_agent_chain)):
    try:
        answer, sources = await chain.ainvoke({
            "question": body.question,
            "chat_history": _to_lc_messages(body.messages),
        })
        return RagAgentOutput(output=answer, sources=_to_source_items(sources))
    except Exception as e:
        raise InternalServerError(detail=str(e))


@router.post(
    "/stream",
    summary="RAG 에이전트 질의응답 (스트리밍)",
    description="답변은 SSE 텍스트 청크로, 출처는 스트림 마지막에 JSON 이벤트로 전달됩니다.",
    response_class=StreamingResponse,
)
async def rag_agent_stream(body: RagAgentInput, chain=Depends(get_rag_agent_chain)):
    async def generator():
        try:
            async for chunk in chain.astream({
                "question": body.question,
                "chat_history": _to_lc_messages(body.messages),
            }):
                if isinstance(chunk, list):
                    # 출처 목록 — 스트림 마지막에 JSON 이벤트로 전달
                    items = [{"name": n, "page": p or None} for n, p in chunk]
                    yield f"data: {json.dumps({'type': 'sources', 'items': items}, ensure_ascii=False)}\n\n"
                elif chunk:
                    yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {e}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")
