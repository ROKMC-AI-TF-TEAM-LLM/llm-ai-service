import pickle
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Query

from api.schemas import DocumentItem, DocumentListOutput
from core.exceptions import InternalServerError
from core.logger import get_logger
from core.settings import settings

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["문서 관리"])

_DOCS_PATH = settings.retriever.vectorstore_path / "docs.pkl"


def _load_all_documents() -> list[DocumentItem]:
    applied_at = datetime.fromtimestamp(_DOCS_PATH.stat().st_mtime)

    with open(_DOCS_PATH, "rb") as f:
        docs = pickle.load(f)

    seen: set[str] = set()
    items: list[DocumentItem] = []
    for doc in docs:
        source = doc.metadata.get("source", "")
        if source and source not in seen:
            seen.add(source)
            path = Path(source)
            items.append(DocumentItem(
                name=path.name,
                type=path.suffix.lstrip(".").upper() or "UNKNOWN",
                applied_at=applied_at,
            ))

    items.sort(key=lambda d: d.name)
    return items


@router.get(
    "",
    response_model=DocumentListOutput,
    summary="RAG 적용 문서 목록",
    description=(
        "현재 RAG 벡터스토어에 인덱싱된 문서 목록을 반환합니다.\n\n"
        "무한 스크롤: 첫 요청은 `offset=0`으로 시작하고, "
        "응답의 `has_more`가 `true`이면 `offset += limit`으로 다음 페이지를 요청합니다."
    ),
)
async def list_documents(
    offset: int = Query(default=0, ge=0, description="건너뛸 문서 수"),
    limit: int = Query(default=10, ge=1, le=100, description="한 번에 반환할 문서 수"),
):
    if not _DOCS_PATH.exists():
        logger.warning("[documents] docs.pkl 없음 — 벡터스토어 미초기화 상태")
        return DocumentListOutput(documents=[], total=0, offset=offset, limit=limit, has_more=False)

    try:
        all_items = _load_all_documents()
        total = len(all_items)
        page = all_items[offset: offset + limit]

        logger.info(f"[documents] offset={offset}, limit={limit}, 반환={len(page)}/{total}")
        return DocumentListOutput(
            documents=page,
            total=total,
            offset=offset,
            limit=limit,
            has_more=(offset + limit) < total,
        )

    except Exception as e:
        logger.error(f"[documents] 오류: {e}")
        raise InternalServerError(detail=str(e))
