# 내부망(오프라인) Reranker 반입 절차

내부망은 인터넷에 접속할 수 없어, reranker 모델(`BAAI/bge-reranker-v2-m3`)을
HuggingFace에서 자동 다운로드할 수 없다. 모델 파일을 **직접 반입**해야 한다.

보안검사 대상 파일 수를 최소화하기 위해, **바이너리/대용량 2개만 반입**하고
작은 텍스트 설정 파일(`config.json`)은 **내부망에서 직접 작성**한다.

> 검증: `config.json` + `model.safetensors` + `tokenizer.json` **3개만으로** 정상 로드되며
> 점수가 전체 6개 구성과 동일함(0.9892)을 확인함. 나머지 3개
> (`sentencepiece.bpe.model`, `tokenizer_config.json`, `special_tokens_map.json`)는
> `tokenizer.json`에 중복 포함되어 불필요.

---

## 1. 반입할 파일 (보안검사 대상) — 2개

인터넷 환경의 `models/bge-reranker-v2-m3/`에서 아래 2개만 반입한다.

| 파일 | 크기 | 비고 |
|------|------|------|
| `model.safetensors` | ~2.2 GB | 모델 가중치 (바이너리, 타이핑 불가) |
| `tokenizer.json` | ~17 MB | 토크나이저 (어휘·특수토큰 내장, 타이핑 불가) |

## 2. 내부망에서 직접 작성할 파일 — 1개

내부망 `models/bge-reranker-v2-m3/` 폴더에 `config.json`을 아래 내용 **그대로** 작성한다.
(한 글자라도 다르면 로드 실패 또는 동작 변경)

```json
{
  "_name_or_path": "BAAI/bge-m3",
  "architectures": ["XLMRobertaForSequenceClassification"],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 0,
  "classifier_dropout": null,
  "eos_token_id": 2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 1024,
  "id2label": {"0": "LABEL_0"},
  "initializer_range": 0.02,
  "intermediate_size": 4096,
  "label2id": {"LABEL_0": 0},
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 8194,
  "model_type": "xlm-roberta",
  "num_attention_heads": 16,
  "num_hidden_layers": 24,
  "output_past": true,
  "pad_token_id": 1,
  "position_embedding_type": "absolute",
  "torch_dtype": "float32",
  "transformers_version": "4.38.1",
  "type_vocab_size": 1,
  "use_cache": true,
  "vocab_size": 250002
}
```

## 3. 최종 폴더 구성 (내부망)

```
<프로젝트 루트>/models/bge-reranker-v2-m3/
├── config.json          # 직접 작성
├── model.safetensors    # 반입
└── tokenizer.json       # 반입
```

`config.yaml`의 `reranker_model: models/bge-reranker-v2-m3` 확인.
코드는 로컬 경로를 감지하면 `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`를 자동 설정해
네트워크 호출 없이 로드한다. (`app/retrievers/reranker.py`)

## 4. 검증

```bash
python -c "import sys; sys.path.insert(0,'app'); \
from core.settings import settings; from retrievers.reranker import build_reranker; \
m = build_reranker(settings.retriever.reranker_model, 'auto'); \
print(round(float(m.predict([('국방 데이터','국방 데이터 관리 규정')])[0]), 4))"
```

- `0.9892` 부근 점수가 출력되면 성공.
- `config.json 이 없습니다` 에러 → 2번에서 파일을 작성/저장했는지 확인.
- `Unrecognized model ... model_type` 에러 → `config.json` 내용 오타(특히 `"model_type": "xlm-roberta"`) 확인.
