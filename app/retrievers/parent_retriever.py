import asyncio
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from core.logger import get_logger

logger = get_logger(__name__)


class HybridParentRetriever(BaseRetriever):
    """자식 청크로 하이브리드 검색 → (선택적 재순위화) → parent_id 조회 → 부모 Document 반환."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    child_retriever: Any
    parent_docs: Dict[str, Any]
    final_k: int = 5
    reranker: Any = None

    def _rerank(self, query: str, candidates: List[Document]) -> List[Document]:
        """CrossEncoder 로 (query, 자식 청크) 쌍을 점수화하여 내림차순 정렬한다."""
        pairs = [(query, c.page_content) for c in candidates]
        scores = self.reranker.predict(pairs)
        # key 로 점수만 비교 (동점 시 Document 비교 방지, 안정 정렬로 원래 순서 유지)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        candidates = self.child_retriever.invoke(query)
        if not candidates:
            logger.info("검색된 자식 청크 없음")
            return []

        if self.reranker is not None:
            candidates = self._rerank(query, candidates)

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
        logger.info(f"최종 부모 문서: {len(result)}개 (재순위화: {self.reranker is not None})")
        return result

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return await asyncio.to_thread(self._get_relevant_documents, query)
