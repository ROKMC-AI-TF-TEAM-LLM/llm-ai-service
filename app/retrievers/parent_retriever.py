import asyncio
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from core.logger import get_logger

logger = get_logger(__name__)


class HybridParentRetriever(BaseRetriever):
    """자식 청크로 하이브리드 검색 → parent_id 조회 → 부모 Document 반환."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    child_retriever: Any
    parent_docs: Dict[str, Any]
    final_k: int = 5

    def _get_relevant_documents(self, query: str) -> List[Document]:
        candidates = self.child_retriever.invoke(query)
        seen: set = set()
        result: List[Document] = []
        for child in candidates:
            pid = child.metadata.get("parent_id")
            if pid is None or pid in seen:
                continue
            parent = self.parent_docs.get(pid)
            if parent is None:
                logger.warning(f"parent_id {pid!r} 에 해당하는 부모 없음 — 재수집 필요")
                continue
            result.append(parent)
            seen.add(pid)
            if len(result) >= self.final_k:
                break
        logger.info(f"최종 부모 문서: {len(result)}개")
        return result

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return await asyncio.to_thread(self._get_relevant_documents, query)
