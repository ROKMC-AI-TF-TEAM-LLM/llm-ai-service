import yaml
from pathlib import Path
from pydantic import BaseModel, model_validator

from core.logger import get_logger

logger = get_logger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"
_PROJECT_ROOT = Path(__file__).parent.parent.parent


class LLMSettings(BaseModel):
    model: str = "gemma4:e4b"
    temperature: float = 0.0
    ollama_host: str = "http://localhost:11434"
    prompt_file: str = "rag-chat.yaml"


class RetrieverSettings(BaseModel):
    vectorstore_path: Path = _PROJECT_ROOT / "data" / "vectorstore"
    embedding_model: str = "bge-m3"
    vector_weight: float = 0.6
    keyword_weight: float = 0.4
    top_k: int = 4
    parent_chunk_size: int = 1500
    parent_chunk_overlap: int = 200
    child_chunk_size: int = 300
    child_chunk_overlap: int = 50
    candidate_k: int = 15
    final_k: int = 5
    rerank_threshold: float = 0.5
    use_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "auto"

    @model_validator(mode="after")
    def resolve_paths(self) -> "RetrieverSettings":
        if not self.vectorstore_path.is_absolute():
            self.vectorstore_path = _PROJECT_ROOT / self.vectorstore_path
        # reranker_model 이 로컬에 존재하는 (상대)경로면 절대경로로 변환.
        # 존재하지 않으면(=허브 ID) 그대로 둔다.
        rk = Path(self.reranker_model)
        candidate = rk if rk.is_absolute() else _PROJECT_ROOT / rk
        if candidate.exists():
            self.reranker_model = str(candidate)
        return self


class WebSettings(BaseModel):
    allowed_domains: list[str] | None = None


class DatabaseSettings(BaseModel):
    url: str = "sqlite:///./app.db"
    echo: bool = False


class LoggingSettings(BaseModel):
    level: str = "INFO"


class Settings(BaseModel):
    llm: LLMSettings = LLMSettings()
    retriever: RetrieverSettings = RetrieverSettings()
    web: WebSettings = WebSettings()
    database: DatabaseSettings = DatabaseSettings()
    logging: LoggingSettings = LoggingSettings()


def _load() -> Settings:
    if not _CONFIG_PATH.exists():
        logger.warning(f"config.yaml 없음 — 기본값 사용 ({_CONFIG_PATH})")
        return Settings()
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    s = Settings(**data)
    logger.info(f"설정 로드 완료: model={s.llm.model}, embedding={s.retriever.embedding_model}, log_level={s.logging.level}")
    return s


settings = _load()
