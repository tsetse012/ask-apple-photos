[English](README.md)

# ask-apple-photos

맥 사진 앱 보관함에서 사진을 말로 찾는 명령줄 도구입니다. 같은 컴퓨터 안에서 웹 화면으로도 쓸 수 있습니다.

예:

```bash
.venv/bin/python ask.py "저번달쯤 강원도 카페에서 찍은 사진"
.venv/bin/python ask.py "양식"
```

사진 앱이 이미 만들어 둔 정보(찍은 날짜·시각, 장소, 장면 라벨, 장소 유형, 얼굴 수, 사진 속 글자)를
`index/cards.sqlite`라는 표로 옮겨 두고, 질문을 Claude에게 주면 Claude가 그 표를 도구로 검색합니다.
사진 자체는 질문 단계에서 밖으로 나가지 않습니다. 사진 내용까지 찾고 싶으면 선택 단계에서
미리보기를 Google Gemini로 보내 한 줄 설명(캡션)을 만들 수 있습니다. 그 단계는 직접 켜야 합니다.

## 되는 것 / 안 되는 것

- **macOS 전용입니다.** 사진 앱 보관함(`.photoslibrary`)이 있어야 합니다.
- **윈도우·리눅스에서는 안 됩니다.** 사진 앱 내부 파일과 Apple Vision을 씁니다.
- **그냥 사진 폴더는 안 됩니다.** 사진 앱에 가져온 보관함만 읽습니다.
- 질문과 라벨은 **한국어 기준**으로 맞춰져 있습니다. 영어로 물어도 되지만 결과가 한국어 낱말에 맞춰집니다.

## 필요한 것

| 항목 | 설명 |
|---|---|
| macOS | 사진 앱 보관함을 쓰는 컴퓨터 |
| Python 3.12 | `python3.12` 명령이 있어야 합니다 |
| Xcode 명령줄 도구 | 로컬 보강 단계가 `swiftc`로 작은 Vision 헬퍼를 만듭니다. `xcode-select --install` |
| Anthropic API 키 | 질문 단계에만 씁니다. 환경변수 `ANTHROPIC_API_KEY` |
| Google Gemini API 키 | 캡션을 켤 때만 씁니다. 환경변수 `GEMINI_API_KEY` |

### 권한

터미널이 사진 보관함 파일을 읽지 못하면 색인을 만들다가 멈출 수 있습니다.
그때는 **시스템 설정 → 개인정보 보호 및 보안 → 전체 디스크 접근 권한**에 터미널 앱(예: Terminal, iTerm)을 추가하세요.

`launchd` 같은 백그라운드 실행으로 자동화하면 이 권한이 없어 **오류 없이 멈추는** 경우가 있습니다.
자동 실행을 붙일 때는 터미널에서 먼저 한 번 끝까지 돌려 보고, 백그라운드에서는 짧은 시간 제한을 두세요.

## 빠른 시작

복사해서 차례로 실행하세요.

```bash
git clone <이 저장소 주소> ask-apple-photos
cd ask-apple-photos

python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 1) 색인 만들기 — 사진 앱 보관함을 읽어 index/cards.sqlite 생성
.venv/bin/python index/build_index.py

# 2) 로컬 보강(무료, 네트워크 없음) — 공개 Vision 분류기·OCR
.venv/bin/python index/enrich_vision.py

# 3) 묻기
export ANTHROPIC_API_KEY=...
.venv/bin/python ask.py "저번달쯤 강원도 카페에서 찍은 사진"

# 4) (선택) 캡션 — 사진 미리보기를 Google로 보내 장면 설명 만들기
export GEMINI_API_KEY=...
.venv/bin/python index/enrich_caption.py                 # 무엇이 나가는지 안내만 하고 종료 코드 2
.venv/bin/python index/enrich_caption.py --send-previews # 실제 전송

# 5) (선택) 내 컴퓨터 안의 웹 화면
export PHOTO_ASK_WEB_PASSWORD="$(openssl rand -hex 16)"
echo "$PHOTO_ASK_WEB_PASSWORD"
.venv/bin/uvicorn --factory web.app:create_app --host 127.0.0.1 --port 4100 --no-proxy-headers
# 브라우저: http://127.0.0.1:4100  (아이디 photo, 위에서 출력한 비밀번호)
```

`--no-proxy-headers`는 uvicorn이 `X-Forwarded-For` 헤더로 클라이언트 주소를 바꾸지 못하게 합니다. 그래야 IP별 속도 제한을 위조한 헤더로 우회할 수 없습니다.

보관함 경로가 다르면 `--library "/경로/Photos Library.photoslibrary"` 또는
`PHOTO_ASK_LIBRARY` 환경변수로 지정합니다.

6,093건(사진·영상, macOS 27.0.1, 맥 한 대) 실측으로 색인 21~24초, Vision 분류 62~73초, OCR 37~51초였습니다(같은 조건에서 세 번 잰 범위). 두 번째부터는 바뀐 사진만 갱신합니다.

사진 앱에서 지운 사진의 카드는 지워지지만, 한 번에 기존 카드의 5%를 넘게 지워야 하면
복사가 불완전했을 수 있어 지우지 않고 경고만 합니다(`--allow-mass-delete`로 허용).

## 명령

### 색인

```bash
.venv/bin/python index/build_index.py [--library 경로] [--index 파일] [--rebuild]
                                      [--allow-mass-delete] [--no-preview-cache]
```

- 보관함은 **수정하지 않습니다.** 내부 DB 파일을 임시 폴더로 복사한 뒤 그 복사본을 엽니다.
- 숨김 사진은 기본으로 검색에서 빠집니다.
- 사진 앱 장면 확신도(`label_scores`)와 웹 화면용 미리보기 캐시(`index/previews/`)도 함께 만듭니다.
- 사진 앱 내부 파일 중 일부(`database/search/psi.sqlite`, `leo.sqlite`, 장면 분류표)는 macOS 버전에 따라
  없거나 이름이 다를 수 있습니다. 없으면 경고만 하고 색인은 계속 만듭니다.

### 로컬 보강 (무료, 네트워크 없음)

```bash
.venv/bin/python index/enrich_vision.py [--step classify|ocr|all] [--force] [--limit N] [--report]
.venv/bin/python index/sync_previews.py [--index 파일] [--dir 폴더] [--include-hidden] [--dry-run] [--copy]
```

- `enrich_vision.py`는 Apple의 공개 Vision 분류기와 한국어 OCR을 작은 Swift 헬퍼로 돌립니다.
  헬퍼는 첫 실행 때 `swiftc`로 자동 컴파일합니다(`swiftc`가 없으면 안내 후 종료).
- 사진은 이 단계에서 밖으로 나가지 않습니다.
- 샌드박스(파일 접근이 제한된 실행 환경) 안에서는 OCR이 첫 장 뒤부터 실패할 수 있습니다. 샌드박스 밖에서 실행하세요.
- `sync_previews.py`는 웹 화면이 읽을 미리보기 사본을 만듭니다. 보관함을 읽을 수 있는 터미널에서 실행하세요.

### 묻기

```bash
.venv/bin/python ask.py "질문" [--interactive] [--limit N] [--open] [--sheet]
                             [--show-paths] [--model 이름] [--effort low|medium|high] [--index 파일]
```

- 후보가 50건을 넘으면 모델이 한 번 되묻습니다. `--interactive`를 주면 그 자리에서 답을 입력합니다.
- `--open`은 미리보기 **사본**을 `open`으로 띄웁니다(보관함 원본을 열지 않음).
- `--sheet`는 `out/<질의 id>/index.html` 컨택트 시트를 만듭니다.
- 결과 표의 기본 식별자는 uuid 앞 8자입니다. 원본 경로까지 보려면 `--show-paths`.
- 호출 기록은 `logs/llm.jsonl`, 대화 원문은 `logs/transcripts/<질의 id>.json`에 남습니다.

### 캡션 (선택, 사진이 밖으로 나감)

```bash
.venv/bin/python index/enrich_caption.py                 # 안내만, 네트워크 없음, 종료 코드 2
.venv/bin/python index/enrich_caption.py --report        # 통계만, 네트워크 없음
.venv/bin/python index/enrich_caption.py --reparse       # 저장된 원문으로 다시 정리, 네트워크 없음
.venv/bin/python index/enrich_caption.py --send-previews # 실제 전송
#   추가 옵션: --model --budget --concurrency --limit --retry-failed
#             --skip-hidden/--no-skip-hidden  --skip-screenshots/--no-skip-screenshots
```

- `--send-previews` 없이는 **사진을 보내지 않습니다.** 무엇이 어디로 나가는지 출력하고 종료 코드 2로 끝납니다.
- 숨김 사진은 기본으로 보내지 않습니다.
- 중간에 멈춰도 다시 실행하면 이미 캡션이 있는 사진은 건너뜁니다(재개).
- 멈추려면 `logs/caption.stop` 파일을 만듭니다.
- 캡션이 0장이어도 질문 단계는 그대로 동작합니다. 요약과 도구가 "캡션이 없다"고 알려 주고,
  장면 라벨·Vision 라벨·사진 속 글자로 찾습니다.

### 웹 화면 (내 컴퓨터 안에서만)

```bash
export PHOTO_ASK_WEB_PASSWORD="$(openssl rand -hex 16)"
echo "$PHOTO_ASK_WEB_PASSWORD"
.venv/bin/uvicorn --factory web.app:create_app --host 127.0.0.1 --port 4100 --no-proxy-headers
```

- `--no-proxy-headers`는 uvicorn이 `X-Forwarded-For` 헤더로 클라이언트 주소를 바꾸지 못하게 합니다. 그래야 IP별 속도 제한을 위조한 헤더로 우회할 수 없습니다.
- 아래 명령으로 띄우면 `127.0.0.1`에만 열립니다. 다른 주소로 열지 마세요.
- 아이디는 `photo`, 비밀번호는 `PHOTO_ASK_WEB_PASSWORD`입니다. 비밀번호가 없거나 16자보다 짧으면 서버가 뜨지 않습니다.
- 속도 제한은 소켓에 보이는 클라이언트 주소를 씁니다. `X-Real-Ip`·`X-Forwarded-For` 헤더는 믿지 않습니다.
- 사진 보관함을 직접 열지 않고 `index/previews/` 캐시만 보여 줍니다.
- 세션은 메모리에만 있고 마지막 사용 30분 뒤 사라집니다.

## 무엇이 밖으로 나가나

| 단계 | 어디로 | 무엇이 | 사진 |
|---|---|---|---|
| 색인·로컬 보강 | 나가지 않음 | — | 0장 |
| 질문 | Anthropic | 질문 문장, 보관함 요약(건수·낱말·기간), 도구 결과(건수·날짜·장소·라벨·캡션 장면 한 줄) | **0장** |
| 캡션 (`--send-previews`일 때만) | Google Gemini | 미리보기 JPEG 1장 + 고정 지시문 | 요청마다 1장 |

- 질문 단계는 사진을 보내지 않습니다(`photo_ask/agent.py:3`). 도구 결과에도 사진 속 글자 전체와 파일 경로는 실리지 않습니다.
- 캡션 단계는 **직접 켜야** 합니다. Gemini 무료 등급 키는 입력이 제품 개선에 쓰일 수 있으니,
  사진을 보내도 되는지 확인한 뒤 유료 등급 키를 권장합니다.
- API 키는 환경변수로만 읽습니다. 파일에 적어 두는 기능은 없습니다.

## 비용

가격표와 상한은 코드에 있습니다(`photo_ask/config.py:28`, `photo_ask/config.py:45`).
비용은 그 가격표로 계산한 추정입니다. 상한 검사는 새 호출 전에 `logs/llm.jsonl`에 이미 기록된 비용으로 하며,
진행 중인 호출이나 실패한 요청의 재시도가 있으면 상한을 조금 넘을 수 있습니다.

| 환경변수 | 기본값 | 뜻 |
|---|---|---|
| `PHOTO_ASK_BUDGET_USD` | 3.0 | 질문 단계 누적 상한(USD) |
| `PHOTO_ASK_QUERY_BUDGET_USD` | 0.25 | 질문 하나의 상한(USD) |
| `PHOTO_ASK_CAPTION_BUDGET_USD` | 6.0 | 캡션 누적 상한(USD) |

2026-11-06 실측(macOS 27.0.1, 맥 한 대, 사진·영상 6,093건, 같은 질문 4개를 세 번(12회), 캡션 3장을 세 번):

- 색인 21~24초, Vision 분류 62~73초, OCR 37~51초.
- 질문 1개: 5~13초, $0.011~$0.051.
- Gemini 캡션: 3장 $0.00236, 장당 약 $0.0008.

같은 조건에서 세 번 잰 작은 표본의 범위이며, 다른 실행에서는 이 범위를 벗어날 수 있습니다. 비용은 가격표로 계산한 추정이며 청구서와 대조하지 않았습니다. 모델과 가격에 따라 달라집니다.

## 환경변수

| 이름 | 기본값 | 쓰는 곳 |
|---|---|---|
| `ANTHROPIC_API_KEY` | 없음(필수) | 질문 |
| `GEMINI_API_KEY` | 없음(캡션 켤 때 필수) | 캡션 |
| `PHOTO_ASK_LIBRARY` | `~/Pictures/Photos Library.photoslibrary` | 색인 |
| `PHOTO_ASK_TZ` | `Asia/Seoul` | 질문 날짜·시간대 |
| `PHOTO_ASK_INDEX` | `index/cards.sqlite` | 전체 |
| `PHOTO_ASK_LOG_DIR` | `logs/` | 전체 |
| `PHOTO_ASK_PREVIEW_CACHE` | `index/previews/` | 웹 화면 |
| `PHOTO_ASK_MODEL` | `claude-sonnet-5-5` | 질문 |
| `PHOTO_ASK_EFFORT` | `low` | 질문 |
| `PHOTO_ASK_BUDGET_USD` | `3.0` | 질문 |
| `PHOTO_ASK_QUERY_BUDGET_USD` | `0.25` | 질문 |
| `PHOTO_ASK_MAX_TURNS` | `10` | 질문 |
| `PHOTO_ASK_CAPTION_MODEL` | `gemini-3.1-flash-lite` | 캡션 |
| `PHOTO_ASK_CAPTION_BUDGET_USD` | `6.0` | 캡션 |
| `PHOTO_ASK_WEB_PASSWORD` | 없음(필수, 16자 이상) | 웹 화면 |

## 한계

- **찾은 사진의 정확도는 아직 재지 않았습니다.** 맞는 사진이 몇 %인지 수치로 확인하지 않았습니다.
- 질문 에이전트가 LLM이라 같은 질문을 다시 해도 후보 수가 달라질 수 있습니다(같은 질문을 두 번 실행한 시험에서 실제로 달라진 경우가 있었습니다).
- 질문·라벨·캡션 낱말은 한국어 기준입니다.
- 사진 앱 내부 파일(`Photos.sqlite`, `database/search/leo.sqlite`, 장면 분류표)을 읽으므로
  macOS 업데이트로 형식이 바뀌면 깨질 수 있습니다.
- 얼굴에 이름을 붙이는 색인은 하지 않습니다. 얼굴 **수**만 셉니다.
- OCR은 사진 앱 미리보기 해상도로 돌립니다. 작은 글자는 놓칠 수 있습니다.
- 웹 화면은 미리보기 캐시만 보여 줍니다. 원본 화질이 아닙니다.
- 사진을 편집해도 이미 만든 캡션은 자동으로 갱신되지 않습니다.
- 다른 도구와의 정확도 비교는 하지 않았습니다. 시험 환경은 macOS 27.0.1 한 대뿐이고, 다른 macOS 버전에서는 시험하지 않았습니다.

## 구조와 설계

`docs/DESIGN.md`(한국어)에 세 단계 구조, 도구 설계 이유, 겪은 문제와 해결을 적었습니다.

## 테스트

실제 보관함을 열지 않고 개인 사진과 API 키도 필요 없습니다. 대부분 가짜 카드·가짜 클라이언트로 돌지만,
`tests/test_build_enrich.py:200`은 macOS 장면 분류표가 있을 때 그 파일을 읽습니다(없으면 건너뜁니다).

```bash
env -u ANTHROPIC_API_KEY -u GEMINI_API_KEY .venv/bin/python -m unittest discover -s tests
```

## 라이선스

MIT. `LICENSE`를 보세요.
