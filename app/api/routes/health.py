from fastapi import APIRouter
import httpx

from core.settings import settings

router = APIRouter(prefix="/health", tags=["서버 상태"])


@router.get("", summary="서버 상태 확인", description="Ollama 연결 및 벡터스토어 상태를 포함한 서버 상태를 확인합니다.")
async def health_check():
    result: dict[str, str] = {"status": "ok", "ollama": "unknown", "vectorstore": "unknown"}

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{settings.llm.ollama_host}/api/tags")
            result["ollama"] = "ok" if r.status_code == 200 else "error"
    except Exception:
        result["ollama"] = "error"
        result["status"] = "degraded"

    vectorstore_path = settings.retriever.vectorstore_path
    if vectorstore_path.exists() and any(vectorstore_path.iterdir()):
        result["vectorstore"] = "ok"
    else:
        result["vectorstore"] = "error"
        result["status"] = "degraded"

    return result
