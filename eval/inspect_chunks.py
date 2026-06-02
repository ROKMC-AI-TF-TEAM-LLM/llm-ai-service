"""
ground_truth.json 작성을 돕는 청크 탐색 스크립트.

사용법:
  # 키워드 검색
  python eval/inspect_chunks.py --keyword "정보화업무"

  # 파일명 필터
  python eval/inspect_chunks.py --source "국방 정보화업무"

  # AND 조건
  python eval/inspect_chunks.py --source "국방 정보화업무" --keyword "보안"

  # 전체 파일명 목록 확인
  python eval/inspect_chunks.py --list-sources

출력 형식:
  [source] 파일명.pdf  page=2
    청크 내용 미리보기...
"""
import sys
import pickle
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
from core.settings import settings

DOCS_PATH = settings.retriever.vectorstore_path / "docs.pkl"
PREVIEW_LEN = 250


def load_docs():
    if not DOCS_PATH.exists():
        print(f"[오류] docs.pkl 없음: {DOCS_PATH}")
        print("먼저 `python app/ingest.py` 를 실행하세요.")
        sys.exit(1)
    with open(DOCS_PATH, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description="청크 탐색기 (ground_truth.json 작성용)")
    parser.add_argument("--keyword", "-k", default="", help="본문 검색 키워드")
    parser.add_argument("--source", "-s", default="", help="파일명 필터 (부분 일치)")
    parser.add_argument("--limit", "-n", type=int, default=20, help="최대 출력 수 (기본 20)")
    parser.add_argument("--list-sources", action="store_true", help="전체 파일명 목록 출력")
    args = parser.parse_args()

    docs = load_docs()

    if args.list_sources:
        sources = sorted({Path(d.metadata.get("source", "")).name for d in docs})
        print(f"총 {len(sources)}개 파일:\n")
        for s in sources:
            count = sum(1 for d in docs if Path(d.metadata.get("source", "")).name == s)
            print(f"  [{count:3d}청크]  {s}")
        return

    print(f"전체 청크 수: {len(docs)}\n")

    matched = []
    for doc in docs:
        src = Path(doc.metadata.get("source", "")).name
        page = doc.metadata.get("page", "?")
        content = doc.page_content

        if args.source and args.source not in src:
            continue
        if args.keyword and args.keyword not in content:
            continue
        matched.append((src, page, content))

    # 같은 파일은 page 순으로 정렬
    matched.sort(key=lambda x: (x[0], x[1] if isinstance(x[1], int) else 0))

    print(f"검색 결과: {len(matched)}개 (최대 {args.limit}개 출력)\n")
    print("=" * 70)

    for src, page, content in matched[: args.limit]:
        preview = content[:PREVIEW_LEN].replace("\n", " ")
        print(f'[source] "{src}"  page={page}')
        print(f"  {preview}")
        print()


if __name__ == "__main__":
    main()
