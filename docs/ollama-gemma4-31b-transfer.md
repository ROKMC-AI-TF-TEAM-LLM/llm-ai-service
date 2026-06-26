# 내부망(오프라인) Ollama 모델 반입 — gemma4:31b

LLM `gemma4:31b`를 인터넷 차단 내부망으로 반입하는 절차.
내부망에는 **Ollama가 이미 설치**돼 있고, **모델 파일만** 옮기면 된다.

> `bge-m3` 절차서(`docs/ollama-bge-m3-transfer.md`)와 동일한 방식.
> 차이: gemma4는 `params` 레이어가 있어 **전송 blob이 3개**(bge-m3는 2개).

---

## 핵심: blob은 내용 주소화(content-addressed)

blob 파일명 = **그 파일 내용의 sha256 해시**.
→ blob은 **손으로 작성 불가** (한 바이트만 달라도 해시 불일치 → Ollama 거부).
→ 반면 **매니페스트(`31b`)는 태그 이름으로 저장**돼 JSON으로 읽히므로 **손으로 작성 가능**.

Ollama는 **매니페스트가 참조하는 blob이 하나라도 없으면 모델을 거부**한다
(`list`엔 보여도 실행 실패). 특정 layer를 빼려면 blob 삭제 + 매니페스트에서 그 layer도 삭제해야 한다.

---

## gemma4:31b 최소 구성 — 총 4개 (검증 완료)

`license` layer(MIT 텍스트)는 추론에 불필요. 매니페스트에서 license layer를 제거하면
**blob 3개 + 매니페스트**로 정상 동작함을 격리 환경에서 확인함
(`ollama show` + 실제 토큰 생성 성공).

| # | 파일 | 크기 | 처리 |
|---|------|------|------|
| 1 | `manifests/registry.ollama.ai/library/gemma4/31b` (license layer 제거판) | 0.5KB | **손으로 작성** |
| 2 | `blobs/sha256-280af6832eca23cb322c4dcc65edfea98a21b8f8ab07dc7553bd6f7e6e7a3313` | **19.87GB** | **전송**(가중치) |
| 3 | `blobs/sha256-0940386273ff9ddd5ede7c5ddaa0e925b50154e198ea977fb64aa1ca94a3a137` | 474B | **전송**(config) |
| 4 | `blobs/sha256-56380ca2ab89f1f68c283f4d50863c0bcab52ae3f1b9a88e4ab5617b176f71a3` | 42B | **전송**(params: temp/top_k/top_p) |
| — | ~~`blobs/sha256-7339fa418c9ad3e8e12e74ad0fd26a9cc4be8703f9c110728a992b193be85cb2`~~ | 11KB | **제거**(license) |

> **보안검사 대상 = blob 3개**(가중치 + config + params). 셋 다 내용주소화라 타이핑 불가 → 반드시 전송.
> `params`(42B)는 작아도 생성 파라미터를 담으므로 **빼면 안 됨**. template 레이어는 없음(GGUF에 내장).

### 손으로 작성할 매니페스트 (`.../gemma4/31b`)

아래 내용 **그대로** 저장(확장자 없음). digest/size가 한 글자라도 다르면 안 됨.

```json
{"schemaVersion":2,"mediaType":"application/vnd.docker.distribution.manifest.v2+json","config":{"mediaType":"application/vnd.docker.container.image.v1+json","digest":"sha256:0940386273ff9ddd5ede7c5ddaa0e925b50154e198ea977fb64aa1ca94a3a137","size":474},"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:280af6832eca23cb322c4dcc65edfea98a21b8f8ab07dc7553bd6f7e6e7a3313","size":19868969920},{"mediaType":"application/vnd.ollama.image.params","digest":"sha256:56380ca2ab89f1f68c283f4d50863c0bcab52ae3f1b9a88e4ab5617b176f71a3","size":42}]}
```

> 단순함을 원하면 license까지 5개를 그대로 전송해도 된다(매니페스트 원본 사용).
> 파일 수를 최소화하려면 위 4개 구성(전송 3 + 작성 1)을 사용.

---

## 1단계: 인터넷 PC에서 blob 3개 수집

`gemma4:31b`를 먼저 받아야 파일이 생긴다(태그 존재 확인).
```bash
ollama pull gemma4:31b
```

```bash
OLL="$HOME/.ollama/models"; OUT="./ollama_transfer/blobs"; mkdir -p "$OUT"
cp "$OLL/blobs/sha256-280af6832eca23cb322c4dcc65edfea98a21b8f8ab07dc7553bd6f7e6e7a3313" "$OUT/"  # 가중치 19.87GB
cp "$OLL/blobs/sha256-0940386273ff9ddd5ede7c5ddaa0e925b50154e198ea977fb64aa1ca94a3a137" "$OUT/"  # config
cp "$OLL/blobs/sha256-56380ca2ab89f1f68c283f4d50863c0bcab52ae3f1b9a88e4ab5617b176f71a3" "$OUT/"  # params
cd "$OUT" && sha256sum sha256-* > ../blobs.sha256          # (선택) 무결성 체크섬
```

> 매니페스트는 전송하지 않고 내부망에서 직접 작성한다(위 JSON). license blob은 수집하지 않는다.

---

## 2단계: 내부망 PC에 배치

blob 3개를 복사하고, 매니페스트는 위 JSON 그대로 작성한다:

```
~/.ollama/models/
├── manifests/registry.ollama.ai/library/gemma4/31b      ← 직접 작성 (license layer 없는 JSON)
└── blobs/
    ├── sha256-280af6832eca23cb322c4dcc65edfea98a21b8f8ab07dc7553bd6f7e6e7a3313   ← 전송(가중치)
    ├── sha256-0940386273ff9ddd5ede7c5ddaa0e925b50154e198ea977fb64aa1ca94a3a137   ← 전송(config)
    └── sha256-56380ca2ab89f1f68c283f4d50863c0bcab52ae3f1b9a88e4ab5617b176f71a3   ← 전송(params)
```

> Windows 경로: `C:\Users\<사용자>\.ollama\models\...` (`OLLAMA_MODELS` 환경변수가 있으면 그 경로).

(체크섬을 떴다면) blob 무결성 검증:
```bash
cd ~/.ollama/models/blobs && sha256sum -c /path/to/blobs.sha256
```

---

## 3단계: 검증

```bash
ollama list                       # 목록에 gemma4:31b 가 보이면 등록 성공
ollama show gemma4:31b            # 31.3B / Q4_K_M / params 표시되면 매니페스트·config·params 정상

# 실제 생성 확인
ollama run gemma4:31b "안녕하세요"
```

- `gemma4:31b`가 안 보임 → 매니페스트 경로/파일명 확인 (`.../library/gemma4/31b`, 확장자 없음)
- `list`엔 보이나 실행 시 "not found, try pulling" → 참조 blob 3개가 모두 있는지,
  파일명이 `sha256-...`(하이픈)로 정확한지 확인 (이름 변경/오타 금지)

---

## ⚠️ 하드웨어 주의

- 모델 19.87GB (Q4_K_M, 31.3B 파라미터, context 262144). 콜드 로드에 RAM/VRAM 많이 필요.
- **내부망 머신 RAM 최소 24~32GB 권장.**
- RTX 4050(6GB) 같은 소형 GPU로는 전량 로드 불가 → CPU/RAM 오프로드로 동작(느림). 내부망 GPU/RAM 사양 확인.

---

## 주의사항

- **파일명·경로 절대 변경 금지**: blob은 내용 해시로 식별 → 한 글자만 달라도 인식 안 됨.
- **blob 파일명은 `sha256-`(하이픈)**, 매니페스트 안 표기는 `sha256:`(콜론).
- 매니페스트가 참조하는 blob은 빠짐없이 존재해야 함(하나라도 없으면 실행 거부).
- 참고: `params`·`license` blob은 `gemma4:e4b`와 **공유**(내용주소화 dedup). 두 태그를 함께 둘 경우 중복 blob은 한 번만 두면 된다.
