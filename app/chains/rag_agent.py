import asyncio
import re
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama

from chains.base import BaseChain
from chains.utils import strip_sources_section
from core.logger import get_logger
from core.prompt_loader import load_system_message, load_format_reminder

logger = get_logger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "rag-agent.yaml"


def _strip_eos(text: str) -> str:
    return re.sub(r"</?eos>", "", text).strip()


def _parse_sources_from_result(xml_str: str) -> list[tuple[str, str]]:
    # search_documents 반환값(format_docs XML)에서 파일명·페이지를 추출한다.
    # 모델이 출처를 직접 작성하는 대신 코드가 정확한 출처를 보장하기 위해 사용한다.
    pairs = []
    for m in re.finditer(r'<page>(.*?)</page>.*?<source>(.*?)</source>', xml_str, re.DOTALL):
        page = m.group(1).strip()
        source = Path(m.group(2).strip()).name  # 전체 경로 → 파일명만
        pairs.append((source, page))
    return pairs


class _MessageBuilder:
    def __init__(self, system_content: str):
        self._system_content = system_content

    def build_initial(self, inputs: dict) -> list:
        messages = [SystemMessage(content=self._system_content)]
        messages.extend(inputs.get("chat_history", []))
        messages.append(HumanMessage(content=inputs["question"]))
        return messages


class _ToolCaller:
    def __init__(self, llm_with_tools, llm_plain, tools_map: dict, format_reminder: str | None = None):
        self._llm = llm_with_tools   # tool 호출 감지용
        self._llm_plain = llm_plain  # 최종 응답 생성용 (bind_tools 없음 → 스트리밍 정상 동작)
        self._tools_map = tools_map
        self._format_reminder = format_reminder

    async def _execute_tools(self, tool_calls: list) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        # 반환값: (ToolMessage용 결과 목록, 출처 후처리용 소스 목록)
        # ainvoke로 호출해 이벤트 루프 블로킹을 방지한다.
        seen: set[str] = set()
        results = []
        collected_sources: list[tuple[str, str]] = []
        for tc in tool_calls:
            query = tc["args"].get("query", tc["args"])
            logger.info(f"[tool-phase] '{tc['name']}' 호출 — query: '{str(query)[:80]}'")
            result = await self._tools_map[tc["name"]].ainvoke(tc["args"])
            if result in seen:
                logger.info(f"[tool-phase] '{tc['name']}' 중복 결과 무시")
                continue
            seen.add(result)
            results.append((tc["id"], result))
            logger.info(f"[tool-phase] '{tc['name']}' 결과 길이: {len(result)}")
            # search_documents 결과에서만 출처를 파싱한다.
            # 웹 tool(fetch_page 등)은 URL을 직접 답변에 쓰도록 모델에 위임한다.
            if tc["name"] == "search_documents":
                collected_sources.extend(_parse_sources_from_result(result))
        return results, collected_sources

    async def run(self, messages: list) -> tuple[str, list[tuple[str, str]]]:
        logger.info("[run] LLM 첫 번째 호출 시작")
        response = await self._llm.ainvoke(messages)
        logger.info(f"[run] tool_calls={len(response.tool_calls)}")

        # tool 호출 없이 바로 답변한 경우 — 출처 없음
        if not response.tool_calls:
            answer = _strip_eos(response.content)
            logger.info(f"[run] tool 없이 직접 응답 (길이: {len(answer)})")
            return answer, []

        messages.append(response)
        tool_results, collected_sources = await self._execute_tools(response.tool_calls)
        for tc_id, result in tool_results:
            messages.append(ToolMessage(content=result, tool_call_id=tc_id))
        if self._format_reminder:
            messages.append(HumanMessage(content=self._format_reminder))

        logger.info("[run] 최종 응답 생성 시작")
        final = await self._llm_plain.ainvoke(messages)
        logger.info(f"[run] 최종 응답 raw={repr(final.content[:120])}")
        answer = strip_sources_section(_strip_eos(final.content))
        logger.info(f"[run] 완료 (길이: {len(answer)}, 출처: {len(collected_sources)}개)")
        return answer, collected_sources

    async def stream(self, messages: list):
        # str 청크: 답변 텍스트 / list 청크: 출처 목록 (라우터에서 타입 구분)
        # astream으로 청크를 수집하면서 tool_calls 여부를 판단한다.
        logger.info("[stream] LLM 첫 번째 호출 시작")
        chunks = []
        async for chunk in self._llm.astream(messages):
            chunks.append(chunk)

        response = chunks[0]
        for c in chunks[1:]:
            response = response + c

        logger.info(f"[stream] tool_calls={len(response.tool_calls)}")

        # tool 호출 없이 바로 답변한 경우 — 수집된 청크를 순서대로 yield
        if not response.tool_calls:
            logger.info("[stream] tool 없이 직접 스트리밍 시작")
            total = 0
            for chunk in chunks:
                if chunk.content:
                    cleaned = _strip_eos(chunk.content)
                    if cleaned:
                        total += len(cleaned)
                        yield cleaned
            logger.info(f"[stream] 완료 (누적 길이: {total})")
            return

        messages.append(response)
        tool_results, collected_sources = await self._execute_tools(response.tool_calls)
        for tc_id, result in tool_results:
            messages.append(ToolMessage(content=result, tool_call_id=tc_id))
        if self._format_reminder:
            messages.append(HumanMessage(content=self._format_reminder))

        logger.info("[stream] 최종 응답 스트리밍 시작")
        total = 0
        chunk_count = 0
        async for chunk in self._llm_plain.astream(messages):
            chunk_count += 1
            # logger.debug(f"[stream] chunk #{chunk_count} content_len={len(chunk.content) if chunk.content else 0}")
            if chunk.content:
                cleaned = _strip_eos(chunk.content)
                if cleaned:
                    total += len(cleaned)
                    yield cleaned

        logger.info(f"[stream] 완료 (누적 길이: {total}, 출처: {len(collected_sources)}개)")
        # 스트리밍 종료 후 출처 목록을 마지막 이벤트로 전달한다.
        # 라우터에서 isinstance(chunk, list)로 구분해 JSON SSE 이벤트로 직렬화한다.
        if collected_sources:
            yield collected_sources


class _AgentLoop(Runnable):
    def __init__(self, tool_caller: _ToolCaller, message_builder: _MessageBuilder):
        self._tool_caller = tool_caller
        self._message_builder = message_builder

    def invoke(self, input, config=None, **_kwargs):
        return asyncio.get_event_loop().run_until_complete(self.ainvoke(input, config))

    async def ainvoke(self, input, _config=None, **_kwargs) -> tuple[str, list[tuple[str, str]]]:
        messages = self._message_builder.build_initial(input)
        return await self._tool_caller.run(messages)

    async def astream(self, input, _config=None, **_kwargs):
        messages = self._message_builder.build_initial(input)
        async for chunk in self._tool_caller.stream(messages):
            yield chunk


class RagAgentChain(BaseChain):
    """
    입력: {"question": str, "chat_history": List[BaseMessage]}
    LLM이 search_documents tool 호출 여부를 판단하고, 결과를 받아 최종 답변을 생성합니다.
    """

    def __init__(self, tools: list, **kwargs):
        super().__init__(**kwargs)
        self.tools = tools

    def setup(self) -> _AgentLoop:
        logger.info(f"RAG Agent 구성 중 (모델: {self.model}, temperature: {self.temperature})")

        llm = ChatOllama(model=self.model, temperature=self.temperature)

        return _AgentLoop(
            tool_caller=_ToolCaller(
                llm_with_tools=llm.bind_tools(self.tools),
                llm_plain=llm,
                tools_map={t.name: t for t in self.tools},
                format_reminder=load_format_reminder(_PROMPT_PATH),
            ),
            message_builder=_MessageBuilder(
                system_content=load_system_message(_PROMPT_PATH),
            ),
        )
