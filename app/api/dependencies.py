import pickle
from functools import lru_cache

from chains.rag import RagChain
from chains.rag_agent import RagAgentChain
from core.exceptions import RagError
from core.logger import get_logger
from core.settings import settings
from retrievers import (
    HybridParentRetriever,
    build_hybrid_retriever,
    build_keyword_retriever,
    build_reranker,
    build_vector_retriever,
)
from tools import build_rag_tools

logger = get_logger(__name__)


@lru_cache()
def get_rag_retriever():
    r = settings.retriever
    vectorstore_path = r.vectorstore_path

    parent_docs_path = vectorstore_path / "parent_docs.pkl"
    if not parent_docs_path.exists():
        raise RagError(
            f"parent_docs.pkl이 없습니다: {parent_docs_path}\n"
            "`python app/ingest.py`를 먼저 실행하세요."
        )
    logger.info(f"부모 문서 로드 중: {parent_docs_path}")
    with open(parent_docs_path, "rb") as f:
        parent_docs = pickle.load(f)
    logger.info(f"부모 문서 로드 완료: {len(parent_docs)}개")

    faiss_retriever = build_vector_retriever(
        vectorstore_path=vectorstore_path,
        embedding_model=r.embedding_model,
        top_k=r.candidate_k,
    )
    bm25_retriever = build_keyword_retriever(
        docs_path=vectorstore_path / "docs.pkl",
        top_k=r.candidate_k,
    )
    child_retriever = build_hybrid_retriever(
        vector_retriever=faiss_retriever,
        keyword_retriever=bm25_retriever,
        vector_weight=r.vector_weight,
        keyword_weight=r.keyword_weight,
    )

    reranker = build_reranker(r.reranker_model, r.reranker_device) if r.use_reranker else None

    return HybridParentRetriever(
        child_retriever=child_retriever,
        parent_docs=parent_docs,
        final_k=r.final_k,
        reranker=reranker,
        rerank_threshold=r.rerank_threshold,
    )


@lru_cache()
def get_rag_chain():
    return RagChain(retriever=get_rag_retriever()).create()


@lru_cache()
def get_rag_tools():
    return build_rag_tools(get_rag_retriever(), settings.web.allowed_domains)


@lru_cache()
def get_rag_agent_chain():
    return RagAgentChain(tools=get_rag_tools()).create()
