# LangChain Ollama API Server

로컬 Ollama LLM을 활용한 LangChain 기반 REST API 서버입니다.  
하이브리드 RAG(벡터 + 키워드 검색), 다중 턴 대화, 웹 페이지 탐색 tool을 지원합니다.

---

## 사용 모델

| 역할 | 모델 | 설명 |
|---|---|---|
| LLM (추론) | `gemma4:e4b` | Google Gemma 4 E4B, native tool calling 지원 |
| 임베딩 | `bge-m3` | BAAI BGE-M3, 다국어 지원 |

---

## 모델 설치

### Ollama 설치

[https://ollama.com](https://ollama.com) 에서 OS에 맞는 설치 파일을 다운로드하세요.

### 방법 1 — Ollama Hub에서 직접 pull (권장)

```bash
# LLM 모델
ollama pull gemma4:e4b

# 임베딩 모델
ollama pull bge-m3
```

### 방법 2 — HuggingFace GGUF 파일로 직접 등록

GGUF 파일을 직접 사용하려면 `huggingface-cli`로 다운로드 후 Modelfile로 등록합니다.

```bash
pip install huggingface-hub

# GGUF 다운로드 예시 (Gemma 4 E4B Q4_K_M)
huggingface-cli download \
  unsloth/gemma-4-E4B-it-GGUF \
  gemma-4-E4B-it-Q4_K_M.gguf \
  --local-dir C:\path\to\models \
  --local-dir-use-symlinks False
```

Modelfile은 `ollama-modelfile/` 디렉토리를 참고하세요.

```bash
# Modelfile로 모델 등록
ollama create gemma4-custom -f ollama-modelfile/gemma-4-E4B-it-Q4_K_M.gguf/Modelfile
```

### Ollama 모델 관리

```bash
ollama list          # 등록된 모델 목록
ollama run gemma4:e4b  # 모델 직접 실행 (테스트)
```

---

## 디렉토리 구조

```
langserve_ollama/
├── config.yaml                  # 전체 설정 (LLM, retriever, DB 등)
├── data/
│   ├── raw/                     # 원본 PDF 문서
│   └── vectorstore/             # FAISS 인덱스 + BM25 문서 캐시
├── ollama-modelfile/            # 모델별 Ollama Modelfile
└── app/
    ├── main.py                  # FastAPI 서버 진입점
    ├── ingest.py                # PDF → 벡터스토어 인덱싱
    ├── api/
    │   ├── router.py            # 전체 라우터 등록
    │   ├── routes/              # 엔드포인트별 핸들러
    │   ├── schemas.py           # 요청/응답 스키마
    │   └── dependencies.py      # 체인/retriever 의존성 주입 (lru_cache)
    ├── chains/
    │   ├── base.py              # BaseChain 추상 클래스
    │   ├── chains.py            # 기본 체인 (LLM, Chat, Topic, Translator)
    │   ├── rag.py               # 단순 RAG 체인
    │   ├── rag_chat.py          # Tool calling 기반 RAG 대화 에이전트
    │   └── utils.py             # 공통 유틸 (format_docs 등)
    ├── retrievers/
    │   ├── vector_retriever.py  # FAISS 벡터 검색
    │   ├── keyword_retriever.py # BM25 키워드 검색
    │   └── hybrid_retriever.py  # 앙상블 (벡터 + 키워드)
    ├── tools/
    │   ├── retriever_tools.py   # search_documents tool
    │   ├── web_search_tools.py  # fetch_page / fetch_page_links tool
    │   └── __init__.py          # build_rag_tools() — tool 조합 진입점
    ├── prompts/
    │   ├── rag-agent.yaml       # RAG 대화 에이전트 시스템 프롬프트
    │   ├── rag-exaone.yaml      # 단순 RAG 프롬프트
    │   └── ...
    └── core/
        ├── settings.py          # config.yaml 로드 + pydantic 검증
        ├── exceptions.py        # 커스텀 HTTP 예외 클래스
        ├── prompt_loader.py     # YAML 프롬프트 로더
        └── logger.py            # 로거 설정
```

---

## 설정

프로젝트 루트의 `config.yaml`에서 모든 설정을 관리합니다.

```yaml
llm:
  model: gemma4:e4b       # 사용할 Ollama 모델명
  temperature: 0.0

retriever:
  embedding_model: bge-m3
  vector_weight: 0.6      # 벡터 검색 가중치
  keyword_weight: 0.4     # 키워드 검색 가중치
  top_k: 4                # 검색 결과 수

web:
  allowed_domains: null
  # 내부망 환경에서는 허용 도메인 지정
  # allowed_domains:
  #   - intranet.company.com

database:
  url: sqlite:///./app.db
  echo: false
```

---

## 실행 방법

### 1. 의존성 설치

```bash
pip install -r requirements.txt
# 또는
poetry install
```

### 2. 문서 인덱싱

`data/raw/` 디렉토리에 PDF 파일을 넣은 후 실행합니다.

```bash
cd app
python ingest.py
```

FAISS 인덱스와 BM25 문서 캐시가 `data/vectorstore/`에 저장됩니다.

### 3. 서버 실행

```bash
cd app
python main.py
```

서버가 실행되면 [http://localhost:8001/docs](http://localhost:8001/docs) 에서 Swagger UI를 확인할 수 있습니다.

---

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/api/health` | 서버 상태 확인 |
| POST | `/api/llm` | LLM 직접 호출 |
| POST | `/api/translate` | 텍스트 한국어 번역 |
| POST | `/api/topic` | 주제 설명 생성 |
| POST | `/api/rag` | 하이브리드 RAG 질의응답 |
| POST | `/api/chat` | 다중 턴 대화 |
| POST | `/api/rag/chat` | RAG + 다중 턴 대화 (tool calling) |

각 엔드포인트에 `/stream`을 붙이면 SSE 스트리밍 응답을 받을 수 있습니다.  
예: `POST /api/rag/chat/stream`

---

## RAG 에이전트 동작 방식

`/api/rag/chat`은 LLM이 tool을 직접 선택하는 에이전트 방식으로 동작합니다.

```
사용자 질문
    ↓
1. search_documents  →  내부 PDF 문서 검색 (하이브리드 RAG)
    ↓ 정보 부족 시
2. fetch_page_links  →  웹 페이지에서 링크 목록 추출
3. fetch_page        →  특정 URL 본문 가져오기
    ↓
최종 답변 생성
```

웹 tool은 `config.yaml`의 `allowed_domains` 설정에 따라 접근 가능한 도메인이 제한됩니다.  
`null`이면 제한 없음(개발 환경), 도메인 목록을 지정하면 해당 도메인만 허용(내부망 환경)됩니다.

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
