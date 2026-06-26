# 내부망(오프라인) Ollama 모델 반입 — bge-m3

임베딩 모델 `bge-m3`를 인터넷 차단 내부망으로 반입하는 절차.
내부망에는 **Ollama가 이미 설치**돼 있고, **모델 파일만** 옮기면 된다.

> Ollama는 `ollama pull`로 레지스트리에서 모델을 받는데, 내부망은 인터넷이 없으므로
> `~/.ollama/models`의 **매니페스트 + blob 파일을 직접 복사**한다. (검증 완료)

---

## Ollama 모델 저장 구조

```
~/.ollama/models/
├── manifests/registry.ollama.ai/library/<모델>/<태그>   ← 매니페스트(JSON, 태그가 파일명)
└── blobs/sha256-<해시>                                   ← 실제 레이어(가중치·config·license)
```

- 매니페스트가 `sha256:<해시>`로 blob을 참조 → 디스크 blob 파일명은 `sha256-<해시>` (콜론→하이픈)
- blobs는 모든 모델이 공유 → **그 모델 매니페스트가 참조하는 blob만** 복사
- Windows 경로: `C:\Users\<사용자>\.ollama\models\...` (`OLLAMA_MODELS` 환경변수가 있으면 그 경로)

---

## 핵심: blob은 내용 주소화(content-addressed)

blob 파일명 = **그 파일 내용의 sha256 해시**. (검증: config blob 파일명 `0c4c9c2a…` == 내용 해시 `0c4c9c2a…`)
→ blob은 **손으로 작성 불가** (한 바이트만 달라도 해시 불일치 → Ollama 거부).
→ 반면 **매니페스트(`latest`)는 태그 이름으로 저장**돼 JSON으로 읽히므로 **손으로 작성 가능**.

또한 Ollama는 **매니페스트가 참조하는 blob이 하나라도 없으면 모델을 거부**한다
(`list`엔 보여도 실행 실패). 따라서 특정 layer를 빼려면 blob 삭제 + 매니페스트에서 그 layer도 삭제해야 한다.

## bge-m3 최소 구성 — 총 3개 (검증 완료)

`license` layer는 MIT 라이선스 텍스트(템플릿)일 뿐 추론에 불필요.
매니페스트에서 license layer를 제거하면 **blob 2개 + 매니페스트**로 정상 동작함을 격리 환경에서 확인함.

| # | 파일 | 크기 | 처리 |
|---|------|------|------|
| 1 | `manifests/registry.ollama.ai/library/bge-m3/latest` (license layer 제거판) | 0.5KB | **손으로 작성** |
| 2 | `blobs/sha256-daec91ffb5dd0c27411bd71f29932917c49cf529a641d0168496c3a501e3062c` | 1.15GB | **전송**(가중치) |
| 3 | `blobs/sha256-0c4c9c2a325fb1cdafec606e6809cb745f1cb26a6d919994400d27372303e276` | 337B | **전송**(config) |

> **보안검사 대상 = blob 2개**(가중치 + config). 둘 다 내용주소화라 타이핑 불가 → 반드시 전송.
> 매니페스트는 손으로 작성(아래 내용 그대로). `license` blob은 가져오지 않음.

### 손으로 작성할 매니페스트 (`.../bge-m3/latest`)

아래 내용 **그대로** 저장(확장자 없음). digest/size가 한 글자라도 다르면 안 됨.

```json
{"schemaVersion":2,"mediaType":"application/vnd.docker.distribution.manifest.v2+json","config":{"mediaType":"application/vnd.docker.container.image.v1+json","digest":"sha256:0c4c9c2a325fb1cdafec606e6809cb745f1cb26a6d919994400d27372303e276","size":337},"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:daec91ffb5dd0c27411bd71f29932917c49cf529a641d0168496c3a501e3062c","size":1157671200}]}
```

> 참고: 단순함을 원하면 license까지 4개를 그대로 전송해도 된다(매니페스트 원본 사용).
> 파일 수를 최소화하려면 위 3개 구성(전송 2 + 작성 1)을 사용.

---

## 1단계: 인터넷 PC에서 blob 2개 수집

```bash
OLL="$HOME/.ollama/models"; OUT="./ollama_transfer/blobs"; mkdir -p "$OUT"
cp "$OLL/blobs/sha256-daec91ffb5dd0c27411bd71f29932917c49cf529a641d0168496c3a501e3062c" "$OUT/"  # 가중치
cp "$OLL/blobs/sha256-0c4c9c2a325fb1cdafec606e6809cb745f1cb26a6d919994400d27372303e276" "$OUT/"  # config
cd "$OUT" && sha256sum sha256-* > ../blobs.sha256          # (선택) 무결성 체크섬
```

> 매니페스트는 전송하지 않고 내부망에서 직접 작성한다(위 JSON). license blob은 수집하지 않는다.

---

## 2단계: 내부망 PC에 배치

blob 2개를 복사하고, 매니페스트는 위 JSON 그대로 작성한다:

```
~/.ollama/models/
├── manifests/registry.ollama.ai/library/bge-m3/latest    ← 직접 작성 (license layer 없는 JSON)
└── blobs/
    ├── sha256-daec91ffb5dd0c27411bd71f29932917c49cf529a641d0168496c3a501e3062c   ← 전송(가중치)
    └── sha256-0c4c9c2a325fb1cdafec606e6809cb745f1cb26a6d919994400d27372303e276   ← 전송(config)
```

(체크섬을 떴다면) blob 무결성 검증:
```bash
cd ~/.ollama/models/blobs && sha256sum -c /path/to/blobs.sha256
```

---

## 3단계: 검증

```bash
ollama list                       # 목록에 bge-m3 가 보이면 등록 성공

# 임베딩 동작 확인 (벡터가 반환되면 정상)
curl http://localhost:11434/api/embeddings -d '{"model":"bge-m3","prompt":"테스트"}'
```

- `bge-m3`가 `ollama list`에 안 보임 → 매니페스트 경로/파일명 확인
  (`.../library/bge-m3/latest`, 확장자 없음)
- 임베딩 호출 시 모델 로드 실패 → 참조 blob 4개가 모두 있는지, 파일명이
  `sha256-...`(하이픈)로 정확한지 확인 (이름 변경/오타 금지)

---

## 주의사항

- **파일명·경로 절대 변경 금지**: blob은 내용 해시로 식별 → 한 글자만 달라도 인식 안 됨.
- **blob 파일명은 `sha256-`(하이픈)**, 매니페스트 안 표기는 `sha256:`(콜론).
- **매니페스트가 참조하는 blob은 빠짐없이** 존재해야 함(하나라도 없으면 `list`엔 보여도 실행 거부). 최소 구성에서는 매니페스트에 config+가중치만 두므로 blob 2개면 충분.
- 다른 모델(gemma4 등)을 추가로 반입할 때도 동일 방식: 매니페스트 + 그 매니페스트가 참조하는 blob. blobs 폴더는 모델 간 공유되므로 중복 blob은 한 번만 두면 된다. (license layer 제거로 파일 수를 줄이는 최적화도 동일하게 적용 가능 — 단 모델별로 매니페스트를 그에 맞게 작성)
