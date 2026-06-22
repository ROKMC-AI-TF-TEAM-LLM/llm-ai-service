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
    rerank_threshold: float = 0.5

    def _rerank(self, query: str, candidates: List[Document]) -> List[tuple]:
        """CrossEncoder 로 (query, 자식 청크) 쌍을 점수화해 (score, doc) 내림차순 정렬한다."""
        pairs = [(query, c.page_content) for c in candidates]
        scores = self.reranker.predict(pairs)
        # key 로 점수만 비교 (동점 시 Document 비교 방지, 안정 정렬로 원래 순서 유지)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [(float(s), c) for s, c in ranked]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        candidates = self.child_retriever.invoke(query)
        if not candidates:
            logger.info("검색된 자식 청크 없음")
            return []

        if self.reranker is not None:
            scored = self._rerank(query, candidates)
        else:
            scored = [(None, c) for c in candidates]

        seen: set = set()
        result: List[Document] = []
        selected_scores: List[float] = []
        for score, child in scored:
            pid = child.metadata.get("parent_id")
            if pid is None or pid in seen:
                continue
            parent = self.parent_docs.get(pid)
            if parent is None:
                logger.warning(f"parent_id {pid!r} 에 해당하는 부모 없음 — 재수집 필요")
                continue
            # 점수가 임계값 미만이면 이후 부모도 모두 미만(내림차순)이므로 중단한다.
            # 단, 최소 1개는 보장해 빈 결과를 방지한다 (result 가 비어있으면 통과).
            if score is not None and score < self.rerank_threshold and result:
                logger.info(
                    f"[rerank] 임계값({self.rerank_threshold}) 미만으로 컷오프 — "
                    f"다음 부모 점수={score:.3f}, 선택된 부모 {len(result)}개"
                )
                break
            result.append(parent)
            seen.add(pid)
            if score is not None:
                selected_scores.append(score)
            if len(result) >= self.final_k:
                logger.info(f"[rerank] final_k({self.final_k}) 상한 도달")
                break

        if selected_scores:
            score_str = ", ".join(f"{s:.3f}" for s in selected_scores)
            logger.info(f"[rerank] 선택된 부모 점수: [{score_str}]")
        logger.info(
            f"최종 부모 문서: {len(result)}개 "
            f"(재순위화: {self.reranker is not None}, 임계값: {self.rerank_threshold})"
        )
        return result

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return await asyncio.to_thread(self._get_relevant_documents, query)
