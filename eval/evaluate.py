"""
리트리버 검색 성능 평가 스크립트.

수식 출처:
  - Precision@k, Recall@k : EvidentlyAI / Pinecone Offline Evaluation Guide
  - MRR                   : MS MARCO 벤치마크 표준
  - NDCG@k                : BEIR / MTEB 표준 (binary relevance, log2(i+1) 할인)

사용법:
  # 기본 실행 (리트리버와 top_k 모두 config.yaml 기준)
  python eval/evaluate.py

  # ground truth 파일 직접 지정
  python eval/evaluate.py --ground-truth eval/ground_truth.json

주의:
  - ground_truth.json 의 pages 가 모두 비어있으면 해당 질문은 스킵됩니다.
  - pages 는 PDFPlumberLoader 기준 0-indexed 페이지 번호입니다.
    inspect_chunks.py 출력의 page= 값을 그대로 사용하세요.
"""
import sys
import json
import math
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from api.dependencies import get_rag_retriever
from core.settings import settings
from core.logger import configure_logging


# ── 관련성 판단 ──────────────────────────────────────────────────────────────

def _is_relevant(doc, relevant_list: list[dict]) -> bool:
    """
    retrieved 청크가 ground truth relevant 목록에 포함되는지 판단.

    relevant_list 항목 예:
      {"source": "파일명.pdf", "pages": [0, 1, 2]}

    source 는 파일명 부분 일치, page 는 pages 리스트 내 포함 여부로 확인.
    """
    src_name = Path(doc.metadata.get("source", "")).name
    page = doc.metadata.get("page", -1)
    for entry in relevant_list:
        if entry["source"] in src_name or src_name in entry["source"]:
            if page in entry["pages"]:
                return True
    return False


def _total_relevant(relevant_list: list[dict]) -> int:
    """ground truth 에 명시된 전체 관련 청크 수 (Recall 분모)."""
    return sum(len(e["pages"]) for e in relevant_list)


# ── 메트릭 수식 ──────────────────────────────────────────────────────────────

def precision_at_k(retrieved: list, relevant_list: list[dict], k: int) -> float:
    """
    Precision@k = (상위 k 중 관련 수) / k

    분모는 k 고정 (실제 반환 수가 아님).
    """
    hits = sum(1 for doc in retrieved[:k] if _is_relevant(doc, relevant_list))
    return hits / k


def recall_at_k(retrieved: list, relevant_list: list[dict], k: int) -> float:
    """
    Recall@k = (상위 k 중 관련 수) / (ground truth 전체 관련 청크 수)

    분모는 corpus 크기가 아닌 ground truth 에 표시된 관련 청크 수.
    """
    total = _total_relevant(relevant_list)
    if total == 0:
        return 0.0
    hits = sum(1 for doc in retrieved[:k] if _is_relevant(doc, relevant_list))
    return hits / total


def mrr(retrieved: list, relevant_list: list[dict]) -> float:
    """
    MRR = 1 / rank_first_relevant

    첫 번째 관련 청크의 순위만 사용 (이후 관련 청크는 무시).
    관련 청크가 없으면 0.
    """
    for i, doc in enumerate(retrieved):
        if _is_relevant(doc, relevant_list):
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: list, relevant_list: list[dict], k: int) -> float:
    """
    NDCG@k (binary relevance)

    DCG@k  = Σ(i=1~k)        rel(i) / log2(i+1)
    IDCG@k = Σ(i=1~min(R,k)) 1      / log2(i+1)   (R = 정답 수)
    NDCG@k = DCG@k / IDCG@k

    참고: BEIR / MTEB 표준, log2(i+1) 할인 인자, i 는 1-indexed.
    """
    def dcg(docs: list) -> float:
        return sum(
            (1.0 if _is_relevant(doc, relevant_list) else 0.0) / math.log2(i + 2)
            for i, doc in enumerate(docs[:k])
        )

    actual_dcg = dcg(retrieved)

    total_relevant = _total_relevant(relevant_list)
    ideal_hits = min(total_relevant, k)
    # 이상적 순서: 관련 청크를 1위부터 ideal_hits 위까지 배치
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ── 평가 실행 ────────────────────────────────────────────────────────────────

def evaluate(ground_truth_path: str):
    top_k = settings.retriever.top_k
    with open(ground_truth_path, encoding="utf-8") as f:
        dataset = json.load(f)

    # _comment 키만 있는 메타 항목 제거 + pages 가 비어있는 항목 스킵
    items = []
    skipped = []
    for item in dataset:
        if "question" not in item:
            continue
        total = _total_relevant(item.get("relevant", []))
        if total == 0:
            skipped.append(item["question"])
            continue
        items.append(item)

    if skipped:
        print(f"[경고] pages 가 비어있어 스킵된 질문 {len(skipped)}개:")
        for q in skipped:
            print(f"  - {q}")
        print()

    if not items:
        print("[오류] 평가할 수 있는 질문이 없습니다. ground_truth.json 의 pages 를 채워주세요.")
        sys.exit(1)

    print("리트리버 로딩 중...")
    retriever = get_rag_retriever()
    print(f"평가 시작: {len(items)}개 질문\n")

    rows = []
    for item in items:
        question = item["question"]
        relevant_list = item["relevant"]

        retrieved = retriever.invoke(question)

        p = precision_at_k(retrieved, relevant_list, top_k)
        r = recall_at_k(retrieved, relevant_list, top_k)
        m = mrr(retrieved, relevant_list)
        n = ndcg_at_k(retrieved, relevant_list, top_k)

        rows.append({
            "question": question,
            f"Precision@{top_k}": p,
            f"Recall@{top_k}": r,
            "MRR": m,
            f"NDCG@{top_k}": n,
        })

        print(f"Q: {question[:60]}")
        print(f"   Precision@{top_k}={p:.3f}  Recall@{top_k}={r:.3f}  MRR={m:.3f}  NDCG@{top_k}={n:.3f}")

        # 검색된 청크 상위 5개 미리보기 (디버깅용)
        for rank, doc in enumerate(retrieved[:5], start=1):
            src = Path(doc.metadata.get("source", "")).name
            page = doc.metadata.get("page", "?")
            rel_mark = "O" if _is_relevant(doc, relevant_list) else "-"
            print(f"   [{rank}] {rel_mark} page={page}  {src[:50]}")
        print()

    # 평균 계산
    metric_keys = [f"Precision@{top_k}", f"Recall@{top_k}", "MRR", f"NDCG@{top_k}"]
    avg = {k: sum(r[k] for r in rows) / len(rows) for k in metric_keys}

    # 결과 표 출력
    sep = "=" * 65
    print(sep)
    print(f"{'메트릭':<20}  {'평균':>8}  {'목표값':>10}  {'판정':>6}")
    print(sep)

    thresholds = {
        f"Precision@{top_k}": 0.5,
        f"Recall@{top_k}": 0.8,
        "MRR": 0.5,
        f"NDCG@{top_k}": 0.8,
    }

    for key in metric_keys:
        val = avg[key]
        thr = thresholds[key]
        verdict = "OK" if val >= thr else "LOW"
        print(f"  {key:<18}  {val:>8.3f}  {thr:>10.1f}  {verdict:>6}")

    print(sep)
    print(f"  평가 질문 수: {len(rows)}개   top_k: {top_k}")
    print(sep)

    return avg


# ── 진입점 ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    configure_logging("WARNING")  # 평가 중 리트리버 로그 억제

    parser = argparse.ArgumentParser(description="RAG 리트리버 검색 성능 평가")
    parser.add_argument(
        "--ground-truth", "-g",
        default=str(Path(__file__).parent / "ground_truth.json"),
        help="ground truth JSON 파일 경로",
    )
    args = parser.parse_args()

    evaluate(args.ground_truth)
