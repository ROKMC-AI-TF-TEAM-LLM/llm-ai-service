import pickle
from pathlib import Path
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_ollama import OllamaEmbeddings

from core.logger import get_logger
from core.exceptions import IngestionError
from core.settings import settings

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
VECTORSTORE_PATH = settings.retriever.vectorstore_path
DOCS_PATH = VECTORSTORE_PATH / "docs.pkl"
EMBEDDING_MODEL = settings.retriever.embedding_model

logger = get_logger(__name__)


def ingest():
    # data/raw/ 디렉토리에서 PDF 파일 목록 수집
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise IngestionError(f"PDF 파일을 찾을 수 없습니다: {DATA_DIR}")

    logger.info(f"{len(pdf_files)}개의 PDF 파일 발견: {DATA_DIR}")

    # 문서를 500자 단위로 청킹, 50자 오버랩으로 문맥 유지
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_docs = []

    for pdf_path in pdf_files:
        logger.info(f"로딩 중: {pdf_path.name}")
        try:
            loader = PDFPlumberLoader(str(pdf_path))
            docs = loader.load_and_split(text_splitter=text_splitter)
        except Exception as e:
            raise IngestionError(f"PDF 로드 실패 [{pdf_path.name}]: {e}") from e

        all_docs.extend(docs)
        logger.info(f"  → {len(docs)}개 청크 생성")

    logger.info(f"전체 청크 수: {len(all_docs)}")

    VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)

    logger.info("임베딩 생성 중 (시간이 걸릴 수 있습니다)...")
    try:
        # bge-m3 임베딩 모델로 벡터화 후 FAISS 인덱스 생성
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = FAISS.from_documents(all_docs, embedding=embeddings)
    except Exception as e:
        raise IngestionError(f"벡터스토어 생성 실패: {e}") from e

    # FAISS 인덱스 저장 (벡터 유사도 검색용)
    vectorstore.save_local(str(VECTORSTORE_PATH))
    logger.info(f"FAISS 인덱스 저장 완료 → {VECTORSTORE_PATH}")

    # 청킹된 문서 pickle 저장 (BM25 키워드 검색용)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(all_docs, f)
    logger.info(f"문서 청크 저장 완료 → {DOCS_PATH}")


if __name__ == "__main__":
    ingest()
