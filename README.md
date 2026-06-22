# LangChain Ollama RAG API Server

로컬 Ollama LLM을 활용한 LangChain 기반 REST API 서버입니다.
**Parent-Child 계층형 청킹**, **하이브리드 검색(벡터 + 키워드)**, **Cross-Encoder 재순위화**, 다중 턴 에이전트, 웹 페이지 탐색 tool을 지원합니다.

---

## 주요 기능

| 기능 | 설명 |
|---|---|
| Parent-Child 계층형 청킹 | 작은 자식 청크(300자)로 정밀 검색, 큰 부모 청크(1500자)로 풍부한 컨텍스트를 LLM에 전달 |
| 하이브리드 검색 | FAISS 벡터 검색(0.6) + BM25 키워드 검색(0.4) 앙상블 |
| Cross-Encoder 재순위화 | `bge-reranker-v2-m3`로 검색 후보를 재정렬, 점수 임계값 기반 동적 출처 반환 |
| RAG 에이전트 | LLM이 tool(문서 검색·웹 탐색)을 직접 선택하는 다중 턴 에이전트 |
| 스트리밍 | 모든 응답 엔드포인트에 SSE 스트리밍 지원 |
| 검색 성능 평가 | Precision@k / Recall@k / MRR / NDCG@k 측정 (`eval/`) |

---

## 사용 모델

| 역할 | 모델 | 실행 | 설명 |
|---|---|---|---|
| LLM (추론) | `gemma4:e4b` | Ollama | Google Gemma 4 E4B, native tool calling 지원 |
| 임베딩 | `bge-m3` | Ollama | BAAI BGE-M3, 다국어 지원 |
| 재순위화 | `BAAI/bge-reranker-v2-m3` | sentence-transformers | Cross-Encoder, 최초 1회 HuggingFace에서 자동 다운로드(~560MB), 이후 로컬 캐시 |

> 재순위화 모델은 GPU가 있으면 자동으로 사용합니다(`reranker_device: auto`). GPU 사용 시 CUDA 빌드 torch가 필요합니다.

---

## 모델 설치

### Ollama 설치

[https://ollama.com](https://ollama.com) 에서 OS에 맞는 설치 파일을 다운로드하세요.

### Ollama 모델 pull

```bash
ollama pull gemma4:e4b   # LLM
ollama pull bge-m3       # 임베딩
```

GGUF 파일로 직접 등록하려면 `ollama-modelfile/` 디렉토리의 Modelfile을 참고하세요.

```bash
ollama create gemma4-custom -f ollama-modelfile/<...>/Modelfile
ollama list              # 등록된 모델 목록
```

### 재순위화 모델

별도 설치 불필요. 서버 최초 기동 시 `BAAI/bge-reranker-v2-m3`가 HuggingFace에서 자동 다운로드되어 `~/.cache/huggingface/`에 캐시됩니다.

---

## 디렉토리 구조

```
llm-ai-service/
├── config.yaml                  # 전체 설정 (LLM, retriever, reranker 등)
├── data/
│   ├── raw/                     # 원본 PDF 문서
│   └── vectorstore/             # FAISS 인덱스 + 자식/부모 청크 캐시
├── eval/                        # 검색 성능 평가
│   ├── evaluate.py              # Precision/Recall/MRR/NDCG 계산
│   ├── inspect_chunks.py        # ground_truth 작성 보조 탐색기
│   ├── ground_truth.json        # 질문 + 정답 페이지
│   └── README.md
├── ollama-modelfile/            # 모델별 Ollama Modelfile
└── app/
    ├── main.py                  # FastAPI 서버 진입점 (lifespan에서 체인 사전 로드)
    ├── ingest.py                # PDF → Parent-Child 청킹 → 벡터스토어 인덱싱
    ├── api/
    │   ├── router.py            # 라우터 등록 (health, rag, rag_agent, documents)
    │   ├── routes/              # 엔드포인트별 핸들러
    │   ├── schemas.py           # 요청/응답 스키마
    │   └── dependencies.py      # 체인/retriever 의존성 주입 (lru_cache)
    ├── chains/
    │   ├── base.py              # BaseChain 추상 클래스
    │   ├── rag.py               # 단순 RAG 체인
    │   ├── rag_agent.py         # Tool calling 기반 RAG 에이전트
    │   └── utils.py             # 공통 유틸 (format_docs 등)
    ├── retrievers/
    │   ├── vector_retriever.py  # FAISS 벡터 검색 (자식 청크)
    │   ├── keyword_retriever.py # BM25 키워드 검색 (자식 청크)
    │   ├── hybrid_retriever.py  # 앙상블 (벡터 + 키워드)
    │   ├── parent_retriever.py  # 자식 검색 → 재순위화 → 부모 반환
    │   └── reranker.py          # Cross-Encoder 로더
    ├── tools/
    │   ├── retriever_tools.py   # search_documents tool
    │   ├── web_search_tools.py  # fetch_page / fetch_page_links tool
    │   └── __init__.py          # build_rag_tools() — tool 조합 진입점
    ├── prompts/
    │   ├── rag-chat.yaml        # 단순 RAG 시스템 프롬프트
    │   └── rag-agent.yaml       # RAG 에이전트 시스템 프롬프트 + format_reminder
    └── core/
        ├── settings.py          # config.yaml 로드 + pydantic 검증
        ├── exceptions.py        # 커스텀 HTTP/앱 예외 클래스
        ├── prompt_loader.py     # YAML 프롬프트 로더
        └── logger.py            # 로거 설정
```

---

## 설정

프로젝트 루트의 `config.yaml`에서 모든 설정을 관리합니다.

```yaml
llm:
  model: gemma4:e4b          # 사용할 Ollama 모델명
  temperature: 0.0
  prompt_file: rag-chat.yaml # 단순 RAG 프롬프트

retriever:
  embedding_model: bge-m3
  vector_weight: 0.6         # FAISS 벡터 검색 가중치
  keyword_weight: 0.4        # BM25 키워드 검색 가중치
  parent_chunk_size: 1500    # 부모 청크 크기 (LLM 컨텍스트)
  parent_chunk_overlap: 200
  child_chunk_size: 300      # 자식 청크 크기 (임베딩/검색)
  child_chunk_overlap: 50
  candidate_k: 15            # 자식 청크 검색 후보 수 (재순위화 입력 풀)
  final_k: 5                 # LLM에 전달할 최종 부모 문서 수 (상한)
  rerank_threshold: 0.5      # 재순위화 점수(0~1) 컷오프, 미만 제외 (최소 1개 보장)
  use_reranker: true         # Cross-Encoder 재순위화 사용 여부
  reranker_model: BAAI/bge-reranker-v2-m3
  reranker_device: auto      # auto | cuda | cpu

web:
  allowed_domains: null
  # 내부망 환경에서는 허용 도메인 지정
  # allowed_domains:
  #   - intranet.company.com

database:
  url: sqlite:///./app.db
  echo: false

logging:
  level: INFO
```

---

## 실행 방법

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

> GPU로 재순위화를 가속하려면 CUDA 빌드 torch가 필요합니다. 예: `pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121`. GPU가 없으면 `reranker_device: auto`가 자동으로 CPU로 폴백합니다.

### 2. 문서 인덱싱

`data/raw/` 디렉토리에 PDF 파일을 넣은 후 실행합니다.

```bash
cd app
python ingest.py
```

`data/vectorstore/`에 다음이 생성됩니다.

- `index.faiss` / `index.pkl` — 자식 청크 임베딩 인덱스
- `docs.pkl` — 자식 청크 (BM25 키워드 검색용)
- `parent_docs.pkl` — 부모 청크 (`{parent_id: Document}`)

> 청킹 설정을 바꾸면 **반드시 재인덱싱**해야 합니다. `parent_docs.pkl`이 없으면 서버 기동 시 에러가 발생합니다.

### 3. 서버 실행

```bash
cd app
python main.py
```

서버 실행 후 [http://localhost:8001/docs](http://localhost:8001/docs) 에서 Swagger UI를 확인할 수 있습니다.

---

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/api/health` | 서버·Ollama·벡터스토어 상태 확인 |
| POST | `/api/rag` | 하이브리드 RAG 질의응답 |
| POST | `/api/rag/stream` | 하이브리드 RAG (SSE 스트리밍) |
| POST | `/api/rag/agent` | RAG 에이전트 질의응답 (tool calling, 다중 턴) |
| POST | `/api/rag/agent/stream` | RAG 에이전트 (SSE 스트리밍) |
| GET | `/api/documents` | 인덱싱된 문서 목록 (페이지네이션) |

스트리밍 응답은 `data: <JSON>\n\n` 형식의 SSE 이벤트(`text` / `sources` / `error` / `done`)를 전달합니다.

---

## RAG 파이프라인

### 검색 흐름

```
질문
  ↓
하이브리드 검색 (FAISS + BM25, 자식 청크 candidate_k=15)
  ↓
Cross-Encoder 재순위화 (질문 × 자식청크 점수화)
  ↓
점수 임계값 필터 (rerank_threshold 이상, 최소 1개 보장)
  ↓
parent_id로 부모 문서 조회 (중복 제거, 최대 final_k=5)
  ↓
부모 청크를 컨텍스트로 LLM 답변 생성
```

### RAG 에이전트 (`/api/rag/agent`)

LLM이 tool을 직접 선택하는 에이전트 방식으로 동작합니다.

```
사용자 질문
    ↓
1. search_documents  →  내부 PDF 문서 검색 (위 RAG 파이프라인)
    ↓ 정보 부족 시
2. fetch_page_links  →  웹 페이지 링크 목록 추출
3. fetch_page        →  특정 URL 본문 가져오기
    ↓
최종 답변 생성 (검색 결과를 컨텍스트로 주입)
```

웹 tool은 `config.yaml`의 `allowed_domains`로 접근 도메인이 제한됩니다.
`null`이면 제한 없음(개발 환경), 도메인 목록을 지정하면 해당 도메인만 허용(내부망 환경)됩니다.

---

## 검색 성능 평가

`eval/` 디렉토리에서 리트리버 검색 성능을 측정합니다. 자세한 내용은 [eval/README.md](eval/README.md) 참고.

```bash
python eval/evaluate.py
```

| 지표 | 목표 | 설명 |
|---|---|---|
| Precision@k | 0.5+ | 상위 k 중 관련 비율 (노이즈) |
| Recall@k | 0.8+ | 정답 페이지 커버리지 |
| MRR | 0.5+ | 첫 관련 문서의 순위 |
| NDCG@k | 0.8+ | 순위 품질 |

> Recall·NDCG는 parent-child 청킹의 페이지 중복을 고려해 고유 `(source, page)` 단위로 집계합니다.

---

## Git 컨벤션

### 커밋 타입

| 타입 | 설명 |
|---|---|
| `FEAT` | 새로운 기능 추가 |
| `FIX` | 버그 수정 |
| `HOTFIX` | 긴급 버그 수정 |
| `REFACTOR` | 코드 리팩토링 |
| `ENHANCEMENT` | 기존 기능 개선 |
| `DOC` | 문서화 |
| `TEST` | 테스트 |
| `CHORE` | 빌드, 패키지 설정 |
| `STYLE` | 코드 포맷, 오타 수정 |

### 브랜치

```
feat/#이슈번호-간단한-설명
fix/#32-쿼리-최적화
```

### 커밋 메시지

```
FEAT/#1 : User 도메인 구현
FIX/#32 : 쿼리 최적화
```
