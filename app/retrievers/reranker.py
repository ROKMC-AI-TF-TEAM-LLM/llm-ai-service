# retrievers/reranker.py
from typing import Any

from core.exceptions import RagError
from core.logger import get_logger

logger = get_logger(__name__)


def build_reranker(model_name: str, device: str = "auto") -> Any:
    """CrossEncoder 기반 재순위화 모델을 로드한다.

    sentence-transformers 는 무거운 의존성이므로 함수 내부에서 지연 import 한다.
    device: "auto"(GPU 있으면 GPU) | "cuda" | "cpu"
    """
    try:
        from sentence_transformers import CrossEncoder
        import torch
    except ImportError as e:
        raise RagError(
            "sentence-transformers 가 설치되어 있지 않습니다. "
            "`pip install sentence-transformers` 후 다시 시도하세요."
        ) from e

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        logger.info(f"CrossEncoder 재순위화 모델 로드 중: {model_name} (device={device})")
        reranker = CrossEncoder(model_name, device=device)
        logger.info(f"CrossEncoder 모델 로드 완료 (device={device})")
        return reranker
    except Exception as e:
        raise RagError(f"재순위화 모델 로드 실패 [{model_name}]: {e}") from e
