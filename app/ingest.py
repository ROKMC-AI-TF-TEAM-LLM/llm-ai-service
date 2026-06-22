import pickle
import uuid
from pathlib import Path

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.exceptions import IngestionError
from core.logger import get_logger
from core.settings import settings

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
VECTORSTORE_PATH = settings.retriever.vectorstore_path
DOCS_PATH = VECTORSTORE_PATH / "docs.pkl"
PARENT_DOCS_PATH = VECTORSTORE_PATH / "parent_docs.pkl"
EMBEDDING_MODEL = settings.retriever.embedding_model

logger = get_logger(__name__)


def ingest():
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise IngestionError(f"PDF 파일을 찾을 수 없습니다: {DATA_DIR}")

    logger.info(f"{len(pdf_files)}개의 PDF 파일 발견: {DATA_DIR}")

    r = settings.retriever
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=r.parent_chunk_size,
        chunk_overlap=r.parent_chunk_overlap,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=r.child_chunk_size,
        chunk_overlap=r.child_chunk_overlap,
    )

    all_children = []
    parent_docs = {}

    for pdf_path in pdf_files:
        logger.info(f"로딩 중: {pdf_path.name}")
        try:
            loader = PDFPlumberLoader(str(pdf_path))
            pages = loader.load()
        except Exception as e:
            raise IngestionError(f"PDF 로드 실패 [{pdf_path.name}]: {e}") from e

        parents = parent_splitter.split_documents(pages)
        pdf_child_count = 0
        for parent in parents:
            parent_id = str(uuid.uuid4())
            parent.metadata["parent_id"] = parent_id
            parent_docs[parent_id] = parent

            children = child_splitter.split_documents([parent])
            for child in children:
                child.metadata["parent_id"] = parent_id
            all_children.extend(children)
            pdf_child_count += len(children)

        logger.info(f"  → 부모 {len(parents)}개, 자식 {pdf_child_count}개")

    logger.info(f"전체 부모: {len(parent_docs)}개, 전체 자식: {len(all_children)}개")

    VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)

    logger.info("임베딩 생성 중 (시간이 걸릴 수 있습니다)...")
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = FAISS.from_documents(all_children, embedding=embeddings)
    except Exception as e:
        raise IngestionError(f"벡터스토어 생성 실패: {e}") from e

    vectorstore.save_local(str(VECTORSTORE_PATH))
    logger.info(f"FAISS 인덱스 저장 완료 → {VECTORSTORE_PATH}")

    with open(DOCS_PATH, "wb") as f:
        pickle.dump(all_children, f)
    logger.info(f"자식 청크 저장 완료 → {DOCS_PATH} ({len(all_children)}개)")

    with open(PARENT_DOCS_PATH, "wb") as f:
        pickle.dump(parent_docs, f)
    logger.info(f"부모 문서 저장 완료 → {PARENT_DOCS_PATH} ({len(parent_docs)}개)")


if __name__ == "__main__":
    ingest()
