# retrievers/reranker.py
import os
from pathlib import Path
from typing import Any

from core.exceptions import RagError
from core.logger import get_logger

logger = get_logger(__name__)


def build_reranker(model_name: str, device: str = "auto") -> Any:
    """CrossEncoder 기반 재순위화 모델을 로드한다.

    sentence-transformers 는 무거운 의존성이므로 함수 내부에서 지연 import 한다.
    device: "auto"(GPU 있으면 GPU) | "cuda" | "cpu"

    model_name 이 로컬 경로면 네트워크 호출 없이 오프라인으로 로드한다(내부망).
    허브 ID면 캐시에 없을 때 HuggingFace 에서 다운로드한다(인터넷 환경).
    """
    is_local = Path(model_name).exists()
    if is_local:
        # 로컬 모델 디렉터리 → 어떤 네트워크 호출도 막는다 (오프라인/내부망)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        # 내부망 반입 시 config.json 을 직접 작성하므로, 누락을 명확히 안내한다.
        # 최소 구성: model.safetensors + tokenizer.json + config.json (3개)
        if not (Path(model_name) / "config.json").exists():
            raise RagError(
                f"{model_name}/config.json 이 없습니다.\n"
                "오프라인 반입 시 config.json 을 직접 작성했는지 확인하세요 "
                "(필수 3개: model.safetensors, tokenizer.json, config.json)."
            )
    elif ("/" in model_name or "\\" in model_name) and not model_name.startswith("BAAI/"):
        # 로컬 경로처럼 보이는데(슬래시 포함) 존재하지 않음 → 전송 누락 가능성
        raise RagError(
            f"reranker 모델 경로가 없습니다: {model_name}\n"
            "오프라인 환경이면 models/bge-reranker-v2-m3 디렉터리가 전송됐는지 확인하세요."
        )

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
        logger.info(f"CrossEncoder 재순위화 모델 로드 중: {model_name} (device={device}, local={is_local})")
        reranker = CrossEncoder(model_name, device=device)
        logger.info(f"CrossEncoder 모델 로드 완료 (device={device}, local={is_local})")
        return reranker
    except Exception as e:
        raise RagError(f"재순위화 모델 로드 실패 [{model_name}]: {e}") from e
