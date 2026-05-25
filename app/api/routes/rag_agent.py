import json
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from api.schemas import RagAgentInput, RagAgentOutput, SourceItem
from api.dependencies import get_rag_agent_chain
from core.exceptions import InternalServerError
from core.logger import get_logger

logger = get_logger(__name__)

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
    logger.info(f"[agent] 요청: '{body.question[:80]}' (이전 메시지: {len(body.messages)}개)")
    try:
        answer, sources = await chain.ainvoke({
            "question": body.question,
            "chat_history": _to_lc_messages(body.messages),
        })
        logger.info(f"[agent] 응답 완료 (길이: {len(answer)}, 출처: {len(sources)}개)")
        return RagAgentOutput(content=answer, sources=_to_source_items(sources))
    except Exception as e:
        logger.error(f"[agent] 오류: {e}")
        raise InternalServerError(detail=str(e))


_STREAM_DESCRIPTION = """
답변을 SSE(Server-Sent Events) 스트림으로 전달합니다.

**이벤트 포맷** (`data: <JSON>\\n\\n`)

| type | 설명 | 페이로드 예시 |
|------|------|--------------|
| `text` | 답변 텍스트 청크 | `{"type":"text","content":"안녕하세요"}` |
| `sources` | 검색된 문서 출처 (스트림 마지막에 1회) | `{"type":"sources","items":[{"name":"doc.pdf","page":"3"}]}` |
| `error` | 오류 발생 | `{"type":"error","message":"오류 내용"}` |

스트림 종료 신호: `data: [DONE]`
"""


@router.post(
    "/stream",
    summary="RAG 에이전트 질의응답 (스트리밍)",
    description=_STREAM_DESCRIPTION,
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": "SSE 스트림",
        }
    },
)
async def rag_agent_stream(body: RagAgentInput, chain=Depends(get_rag_agent_chain)):
    logger.info(f"[agent/stream] 요청: '{body.question[:80]}' (이전 메시지: {len(body.messages)}개)")

    async def generator():
        try:
            total = 0
            async for chunk in chain.astream({
                "question": body.question,
                "chat_history": _to_lc_messages(body.messages),
            }):
                if isinstance(chunk, list):
                    logger.info(f"[agent/stream] 출처 전달: {len(chunk)}개")
                    items = [{"name": n, "page": p or None} for n, p in chunk]
                    yield f"data: {json.dumps({'type': 'sources', 'items': items}, ensure_ascii=False)}\n\n"
                elif chunk:
                    total += len(chunk)
                    yield f"data: {json.dumps({'type': 'text', 'content': chunk}, ensure_ascii=False)}\n\n"
            logger.info(f"[agent/stream] 스트림 완료 (누적 길이: {total})")
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            logger.error(f"[agent/stream] 오류: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")
