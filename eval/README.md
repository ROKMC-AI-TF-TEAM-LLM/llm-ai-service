# RAG 검색 성능 평가 (eval)

리트리버가 올바른 청크를 얼마나 잘 가져오는지 수식 기반으로 측정합니다.

---

## 디렉토리 구조

```
eval/
├── README.md              # 이 파일
├── ground_truth.json      # 질문 + 정답 청크 목록 (직접 작성)
├── inspect_chunks.py      # ground_truth.json 작성 보조 탐색기
└── evaluate.py            # Precision@k / Recall@k / MRR / NDCG@k 계산
```

---

## 평가 지표

| 지표 | 수식 | 목표값 | 낮을 때 원인 |
|------|------|--------|------------|
| **Precision@k** | (상위 k 중 관련 수) / k | 0.5+ | 노이즈 청크 과다 |
| **Recall@k** | (상위 k 중 관련 수) / (정답 총 수) | 0.8+ | 필요한 청크 누락 |
| **MRR** | 1 / rank (첫 번째 관련 청크 순위) | 0.5+ | 관련 청크가 하위에 묻힘 |
| **NDCG@k** | DCG@k / IDCG@k | 0.8+ | 순위 정렬 부정확 |

목표값 기준: Precision·Recall·NDCG → BEIR / MTEB 표준, MRR → MS MARCO 표준

### 수식 상세

```
Precision@k = (상위 k개 중 관련 청크 수) / k
              ※ 분모는 k 고정 (실제 반환 수가 아님)

Recall@k    = (상위 k개 중 관련 청크 수) / (ground_truth 전체 관련 청크 수)
              ※ 분모는 corpus 크기가 아닌 ground_truth 의 pages 총 수

MRR         = 1 / rank_first_relevant
              ※ 첫 번째 관련 청크만 봄, 이후는 무시

DCG@k       = Σ(i=1~k)        rel(i) / log₂(i+1)
IDCG@k      = Σ(i=1~min(R,k)) 1      / log₂(i+1)   R = 정답 수
NDCG@k      = DCG@k / IDCG@k
```

---

## 사용 방법

### 사전 조건

```bash
# 벡터스토어가 없으면 먼저 인덱싱 실행
python app/ingest.py
```

---

### Step 1. 청크 탐색 (ground_truth.json 작성 보조)

```bash
# 전체 파일 목록과 청크 수 확인
python eval/inspect_chunks.py --list-sources

# 키워드로 청크 검색
python eval/inspect_chunks.py --keyword "정보화업무"

# 파일명 + 키워드 AND 조건
python eval/inspect_chunks.py --source "국방 정보화업무" --keyword "적용 범위"

# 출력 수 제한 (기본 20개)
python eval/inspect_chunks.py --keyword "인공지능" --limit 10
```

출력 예시:
```
[source] "국방 정보화업무 훈령(국방부훈령)(제3080호)(20250917).pdf"  page=1
  이 훈령은 국방부 및 예하 기관의 정보화업무 전반에 적용한다...
```

---

### Step 2. ground_truth.json 작성

`eval/ground_truth.json`의 `pages` 목록을 Step 1에서 확인한 `page=` 값으로 채웁니다.

> **주의**: `page` 번호는 PDFPlumberLoader 기준 **0-indexed** (첫 페이지 = 0)

```json
[
  {
    "question": "국방 정보화업무 훈령의 적용 범위는?",
    "relevant": [
      {
        "source": "국방 정보화업무 훈령(국방부훈령)(제3080호)(20250917).pdf",
        "pages": [1, 2]
      }
    ]
  },
  {
    "question": "질문이 여러 파일에 걸친 경우",
    "relevant": [
      { "source": "파일A.pdf", "pages": [0, 3] },
      { "source": "파일B.pdf", "pages": [5] }
    ]
  }
]
```

- `source`: 파일명 부분 일치 허용 (전체 경로 불필요)
- `pages`: 비어있으면 해당 질문은 평가에서 스킵됨

---

### Step 3. 평가 실행

리트리버는 `app/api/dependencies.py`의 `get_rag_retriever()`를 사용합니다.
리트리버 방식을 변경해도 `evaluate.py` 수정 없이 자동 반영됩니다.
top_k는 `config.yaml` 값을 그대로 사용합니다.

```bash
# 기본 실행
python eval/evaluate.py

# ground_truth 파일 경로 직접 지정
python eval/evaluate.py --ground-truth eval/ground_truth.json
```

출력 예시:
```
Q: 국방 정보화업무 훈령의 적용 범위는?
   Precision@4=0.500  Recall@4=0.750  MRR=1.000  NDCG@4=0.812
   [1] O page=1  국방 정보화업무 훈령(국방부훈령)...
   [2] - page=7  국방데이터 관리 훈령(국방부훈령)...
   [3] O page=2  국방 정보화업무 훈령(국방부훈령)...
   [4] - page=0  인공지능 발전과 신뢰...

=================================================================
메트릭                평균       목표값    판정
=================================================================
  Precision@4        0.500         0.5      OK
  Recall@4           0.750         0.8     LOW
  MRR                1.000         0.5      OK
  NDCG@4             0.812         0.8      OK
=================================================================
```

---

## 검색 성능 개선 방향

| 증상 | 원인 | 조정 위치 |
|------|------|----------|
| Precision@k 낮음 | 노이즈 청크 과다 | `top_k` 축소, `vector_weight` 증가 |
| Recall@k 낮음 | 필요한 청크 누락 | `top_k` 확대, 청크 크기 조정 |
| MRR 낮음 | 관련 청크가 하위 순위 | 하이브리드 가중치 재조정 |
| NDCG@k 낮음 | 순위 정렬 부정확 | `vector_weight` / `keyword_weight` 튜닝 |

`config.yaml`에서 조정 가능한 파라미터:

```yaml
retriever:
  top_k: 4            # 검색 결과 수 (리트리버 반환 수 + 메트릭 평가 기준)
  vector_weight: 0.6  # FAISS 가중치
  keyword_weight: 0.4 # BM25 가중치
```
