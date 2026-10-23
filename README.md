# hermes-standalone

[NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)의 원본 에이전트,
CLI, 게이트웨이, 세션 저장소를 유지하면서 **한 개의 Python 파일**로 배포하는 포크입니다.
런타임에는 CPython 3.14 이상의 표준 라이브러리를 사용합니다. 순수 파이썬 의존성과
리소스·라이선스를 파일 안에 포함하고, 네이티브 의존 경계는 파이썬 또는 OS API로 교체합니다.

실제로 복사한 단일 파일에서 CLI → 분리된 게이트웨이 → 별도 관리 워커 → HTTP 모델 응답
→ 원본 도구 실행 → 세션·워커 영수증 저장까지 검증했습니다. 모든 선택 기능의 이식과
모든 OS의 동등성 검증이 끝난 상태는 아닙니다. 구체적인 상태는
[capabilities.json](standalone/capabilities.json)에 기록합니다.

원본 기준 커밋은 `bd3cbe91a90097f9804f16ea5c1768f5cbaaf0b7`입니다.
기존 설명은 [README.upstream.md](README.upstream.md)에 보존했습니다.

**실행**

```sh
python3.14 -I -S hermes.py setup
python3.14 -I -S hermes.py
python3.14 -I -S hermes.py chat -q "프로젝트의 할 일을 정리해 줘"
python3.14 -I -S hermes.py dashboard
python3.14 -I -S hermes.py acp --check
```

공급자 계정과 API 키는 원본 Hermes처럼 설정합니다. 모델·메시징·클라우드 서비스는
외부 서비스이며, 브라우저·셸·컨테이너는 각 도구가 사용하는 실행 환경입니다.
이 파일은 실행 중 pip·uv·npm·PM으로 파이썬 의존성을 설치하지 않습니다.
의존 집합 변경과 자체 업데이트에는 새 배포 파일을 빌드하거나 교체해야 합니다.

인자 없는 실행은 원본 classic Python REPL을 사용합니다. React/Ink UI와 Electron
실행 파일은 포함하지 않습니다. 대시보드는 미리 빌드한 원본 정적 자산을 사용하므로
실행할 때 Node가 필요하지 않습니다. 현재 공개 소스의 기본 빌드에는 대시보드 자산이
없으며, 완전한 라이선스 고지와 출처 검증을 통과한 자산을 별도로 준비해야 합니다.

일반적인 `python hermes.py`도 같은 파일을 `-I -S -B -X utf8`로 다시 실행합니다.
내장 코드와 리소스는 사용자 전용 캐시에 해시별로 추출합니다. 별도 가상환경은 만들지
않습니다. `--cache-dir`은 명령 앞에 지정합니다.

```sh
python3.14 -I -S hermes.py --cache-dir ./private-cache doctor --json --require-agent
python3.14 -I -S hermes.py selftest --json
python3.14 -I -S hermes.py probe-agent --stage construct --json
python3.14 -I -S hermes.py manifest
python3.14 -I -S hermes.py licenses httpx
```

`doctor`의 에이전트 준비 상태는 Python 버전과 핵심 런타임 이식 상태를 나타냅니다.
모든 선택 기능의 동등성을 뜻하지는 않습니다. `selftest`와 `probe-agent`는 임시 프로필을
사용하며 사용자의 기존 키·설정·DB를 열지 않습니다. 진단의 고정 응답 fixture와 실제
HTTP/게이트웨이 검증은 구분합니다. 이 저장소의 검증에는 실제 서비스 자격 증명을 사용하지 않습니다.

**빌드**

빌드 도구 자체도 표준 라이브러리만 사용하며 패키지 설치나 빌드 훅을 실행하지 않습니다.
의존 아카이브 다운로드는 명시적인 준비 단계입니다. 이후 빌드는 오프라인으로 동작합니다.

```sh
python3.14 -I -S scripts/standalone_assets.py restore
python3.14 -I -S standalone/vendor.py fetch
python3.14 -I -S standalone/build.py
python3.14 -I -S standalone/check.py
python3.14 -I -S standalone/check.py --artifact dist/hermes.py --artifact-only
```

Python 3.11 이상으로 패키징 도구를 실행할 수 있지만 원본 Hermes 런타임과 전체 통합 검증에는
Python 3.14 이상이 필요합니다. 입력은 [uv.lock](uv.lock)과 별도로 출처를 구분한
[추가 잠금 파일](standalone/extra-dependencies.lock.json)에 고정합니다. 네이티브 바이너리,
실행 파일, 바이트코드, 경로 탈출과 아카이브 충돌은 거부합니다. 원본 메타데이터와 라이선스,
검토한 소스 패치의 전후 해시·원문·원본 RECORD를 보존합니다.
포함된 제삼자 코드에는 각 원본 라이선스가 적용됩니다. Piper 1.8.0 관련 이식에는
GPL-3.0-or-later 원문과 출처를 포함하며, 별도로 읽는 음성 모델은 해당 모델 카드의 조건을 따릅니다.

큰 중국어 모델과 일본어 사전은 [분할된 원본 자산](build-resources/README.md)에서 복원합니다.
복원에는 네트워크나 추가 패키지가 필요하지 않으며, 각 조각·압축 스트림·원본 파일의
SHA-256과 크기를 확인합니다. 원본과 다른 기존 파일은 덮어쓰지 않습니다.

대시보드를 포함하려면 원본의 잠긴 웹 워크스페이스를 빌드한 뒤 준비된 자산을 가져옵니다.
프런트엔드 컴파일은 별도의 빌드 준비 단계이며 파이썬 번들러는 Node나 패키지 관리자를 실행하지 않습니다.

```sh
python3.14 -I -S standalone/web_assets.py \
  --source /path/to/built/web-workspace \
  --product /path/to/web-dist \
  --output hermes_cli/web_dist
```

이 도구는 자산별 해시, 컴파일 출처와 제삼자 라이선스를 함께 보존하며 기존 자산 디렉터리를
덮어쓰지 않습니다. 웹 자산 없이 빌드하면 CLI·게이트웨이는 사용할 수 있지만 대시보드 자산은 없습니다.
기존 웹 빌드에서는 일부 패키지의 고지가 누락되어 공개 대상에서 제외했습니다.
대시보드 준비 작업은 CI의 수동 `include_dashboard` 옵션이며, 필요한 고지가 없거나
설치된 의존 그래프가 불완전하면 실패합니다. 새 웹 워크스페이스의 전체 빌드 검증은 남아 있습니다.

**용량과 결정론적 minify**

기본 배포물은 선택된 소스·리소스 바이트를 보존하고 전체 아카이브를 LZMA/XZ로 압축합니다.
같은 입력과 같은 도구 체인에서는 정렬 순서·ZIP 시간·메타데이터·인코딩이 고정되어 동일한 파일을 만듭니다.
전체 후보 아카이브를 비교한 결과 기본 preset을 6에서 8로 올려 소스를 바꾸지 않고
배포 파일을 약 1.15 MB 줄였습니다. 두 설정 모두 64 MiB 디코더 제한에서 바이트 단위로 복원됐습니다.
같은 아카이브에서 사전 크기 48 MiB와 `pb=0`을 적용하면 압축 데이터가 추가로
294,416바이트 줄었습니다. 이 설정도 동일한 디코더 메모리 제한 안에서 원본 해시와 일치하며 기본값으로 사용합니다.

압축 데이터는 [Python 소스 인코딩 선언](https://peps.python.org/pep-0263/)을 사용하는
Latin-1 문자열에 저장합니다. 기본 형식은 압축 데이터를 256KiB 문자열들의 튜플로 나눠
CPython이 거대한 문자열 하나를 컴파일할 때의 메모리 사용을 줄입니다. 각 문자열에서
NUL·CR·역슬래시·따옴표만 이스케이프하고 원래 바이트로 복원합니다. 압축 데이터의 전체
크기와 SHA256을 먼저 검증한 뒤, ZIP을 임시 파일로 스트리밍 복원하고 원본 파일·모델·사전의
크기와 해시를 확인합니다. 기존 형식 1–3도 계속 읽으며, 기존 단일 문자열은
`--payload-encoding latin1`로 생성할 수 있습니다.
동일한 전체 후보의 Base85 버전 63.13 MB와 비교해 51.67 MB로 18.14% 줄었으며, 복원 해시와
실제 관리 워커 실행을 확인했습니다. 이 수치는 같은 후보 데이터의 인코딩 비교이며 최종 기능 추가에 따라 용량은 달라집니다.
이스케이프를 필요한 곳에만 넣으면 같은 후보·같은 압축 데이터에서 추가로 581,638바이트가 줄어
51.09 MB가 됩니다. 모든 바이트 쌍·따옴표 연속·NUL 뒤 숫자·CRLF와 반복 생성의 정확성을 검사합니다.
ASCII 파일이 필요한 도구와 연동하려면 `--payload-encoding base85`로 빌드합니다.
생성 파일은 바이너리 복사로 전달하고, 편집기에서 다른 문자 인코딩으로 다시 저장하지 않습니다.

거대한 소스 문자열의 파싱 비용을 피하려면 표준 라이브러리 zipapp 형식을 선택할 수 있습니다.
이는 ZIP 컨테이너인 한 파일이며, 기본값인 텍스트 `.py`와 구분해 `.pyz`로 출력할 수 있습니다.
같은 압축 데이터·소스·리소스를 저장하고, 실행할 때 압축 데이터와 ZIP·매니페스트·캐시의
전체 크기·해시 검증을 그대로 수행합니다. 자식 프로세스도 같은 파일을 분리된 인터프리터로 실행합니다.
`--payload-encoding`은 텍스트 소스 형식의 옵션이며, zipapp은 압축 바이트를 그대로 저장합니다.
AST minify는 두 형식 모두 아카이브 안의 앱 소스에 동일하게 적용됩니다.

```sh
python3.14 -I -S standalone/build.py --container zipapp --output dist/hermes.pyz
python3.14 -I -S standalone/build.py --container zipapp --minify ast --output dist/hermes-ast.pyz
python3.14 -I -S dist/hermes.pyz doctor --json
```

새 컨테이너의 소규모 빌드·분리 실행·실제 관리 워커·변조 거부 검증은 두 Python 버전에서
통과했습니다. 측정값과 전체 리소스를 담은 복사 실행 검증 상태는
[빌드 보고서](standalone/BUILD_REPORT.md)에 기록합니다.

Piper가 원래 포함하는 아랍어·히브리어 보조 모델은 원본 파일 그대로 제공합니다.
아카이브 안에서는 4바이트씩 바이트 위치를 모아 압축하고, 추출할 때 정확히 역변환한 뒤
원본 SHA-256을 검사합니다. 양자화나 가중치 변경은 없습니다. 두 모델만 따로 비교한
XZ 압축 크기는 합계 24,040,316바이트에서 22,028,412바이트로 줄었습니다.
일본어의 Sudachi 사전도 연결 비용의 차분과 고정 필드의 바이트 위치를 모아 압축하고,
추출 전에 원본 202,236,003바이트와 SHA-256을 복원합니다. 같은 설정의 사전 단독 XZ는
48,980,520바이트에서 38,606,472바이트로 줄었습니다. 항목·가중치·정밀도는 줄이지 않습니다.
전체 배포 파일의 크기는 같은 입력으로 다시 빌드한 결과를 기준으로 기록합니다.

```sh
python3.14 -I -S standalone/build.py --minify ast --output dist/hermes-ast.py
python3.14 -I -S standalone/size_report.py --artifact dist/hermes.py \
  --output standalone/.cache/size-report.json
```

AST 모드는 앱의 파이썬 소스를 결정론적으로 재출력한 뒤 위치를 제외한 전체 AST,
재귀적인 컴파일 코드 지문, 재처리의 멱등성이 모두 같은지 검사합니다. 이름·리터럴·문서 문자열·
assert·annotation·라이선스를 임의로 제거하지 않습니다. 검사가 실패하거나 결과가 커지면
그 파일의 원문을 유지합니다. 원본 해시가 고정된 모델 설정·데이터용 Python 파일도 원문을 유지합니다.
공급자 라이브러리는 이 모드로 재작성하지 않습니다.
파일별 검증 결과는 `hermes-ast.py.minify.json`에, 그 해시는 배포 매니페스트에 남깁니다.

소스 읽기·`inspect.getsource()`·traceback 위치·tracing/coverage까지 동일하다는 의미는
아닙니다. 이런 관찰 결과를 포함해 원문을 보존하려면 기본 모드를 사용합니다.
AST 결과의 빌드와 실행에는 기록된 것과 같은 Python 구현·패치 버전이 필요합니다.
다른 도구 체인에서는 캐시를 만들기 전에 실행을 거부합니다. 기본 원문 모드에는 이 추가 제한이 없습니다.

**현재 이식 범위**

- 원본 Chat Completions·Responses·Anthropic 도구 루프와 JSON/SSE, Bedrock,
  Google/Azure 인증, 클라우드 음성·이미지·비디오 HTTP 경계.
- 원본 CLI·게이트웨이·관리 워커·프로필·DB·플러그인 호스트·코드 커널·ACP,
  MCP stdio/HTTP/SSE·OAuth·취소·도구 탐색.
- 대시보드 라우터·인증·정적 자산·PTY, 웹 검색 공급자, Telegram·Slack·Discord 텍스트,
  Olm/Megolm 암호화 Matrix, DingTalk·Feishu·Teams·Google Chat·SMS.
- Matrix 키 교환·회전·교차 서명·복구·암호화 첨부파일·기존 pickle DB 마이그레이션,
  Meet의 원본 입장·자막·발화 큐와 직접 CDP 브라우저 연결.
- Discord DAVE/MLS의 가입·커밋·키 갱신·재초기화와 중첩 RTP 암호화.
  고정 Rust 원본과 실제 원본 Discord TLS/UDP 소비 경로를 대조하며,
  인증 실패나 준비되지 않은 암호 세션을 평문으로 우회하지 않습니다.
- PNG/APNG·GIF·BMP·JPEG·WebP·TIFF·ICO와 이미지 연산, TTF/OTF 텍스트,
  PDF·OOXML·ODF·EPUB·RTF·구형 Office 텍스트 추출. OCR은 명시적인 원격 공급자가 필요합니다.
- 운영체제 프로세스·서비스·전원 경계 및 표준 라이브러리 기반 화면 캡처.
- PulseAudio·ALSA·WinMM·AudioQueue를 통한 녹음·재생·barge-in,
  순수 Python SILK v3 음성 디코딩과 원본 전사·QQ Bot 변환 경로.
  SILK는 독립 C SDK의 전체 패킷·주파수 전환·손실 복원 PCM과 바이트 단위로 대조합니다.
- Opus SILK/CELT/hybrid 수신·FEC·손실 복원·모드 전환과 CELT CBR 송신.
  수신 wrapper의 독립 1,384프레임은 PCM16 최대 1 LSB 차이이며,
  SILK 구성 요소는 1,014프레임의 PCM과 상태가 원본과 정확히 일치합니다.
  원본 Discord 믹서→코덱→DAVE/RTP 송수신·WAV 녹음·재생도 검증했습니다.
  송신 SILK/hybrid/FEC는 미구현이며 요청된 FEC와 실효 지원을 구분합니다.
- 로컬 Whisper의 CT2 가중치·logmel·원본 tokenizer·Transformer·beam search·단어 정렬·VAD.
  실제 발화 FP32 전사는 원본 토큰과 일치하며, INT8는 원본 GENERIC CPU 결과와 같고
  AVX2와 쉼표 하나 차이가 있습니다. 11초 음성 처리에 약 15~18분이 걸립니다.
  기존 로컬 전사 선택·모델 캐시·유휴 해제를 연결했으며 WAV/ADPCM·FLAC·Ogg/Opus·Ogg/Vorbis·WebM/Matroska의 Opus·Vorbis·MP3·AAC-LC/HE-AAC/ADTS/M4A와
  원본 SWR 리샘플링을 지원합니다. 실제 MP3·AAC·M4A 파일도 원본 모델의 전사·시간 정보를 대조합니다.
  HE-AAC의 SBR·Parametric Stereo는 원본 GENERIC 연산 프로필로 361개 파일과
  1,444개 파일·출력 조합을 검증했으며, CPU 최적화에 따른 수치 차이는 별도로 기록합니다.
  Vorbis는 260개 파일의 원본 PCM과 드문 floor·블록 크기·연속 스트림 상태를 대조했고,
  기존 전사 경로의 1,040개 출력 조합도 검사합니다. 실제 파일 11개로
  원본 전사 모델의 실행 기록과 문장·시간 정보가 유지되는지 확인합니다.
  WebM·Matroska는 원본 2,138개 패킷, lacing·지연·패딩·zlib/LZO/헤더 압축과
  952개 PCM 출력 조합을 검사합니다. Opus는 기존 libopus 기준을 사용하며,
  일부 입력에서 원본 FFmpeg 디코더와 큰 PCM 차이가 나는 제한을 별도로 기록합니다.
  손상된 Vorbis는 원본과 같이 오류 전의 PCM을 유지하고 첫 잘못된 패킷에서 멈춥니다.
- Windows SSH 보안 파일·artifact 재진입, 원본 Hugging Face HTTP/LFS trace 업로드,
  Relay 미들웨어·ATOF 파일/HTTP/WebSocket/NDJSON·ATIF와 OTLP HTTP/gRPC trace·log·metric 경로.
  가격표·캐시·구간별 비용 계산과 지표의 집계·주기 전송·실패 기록도 원본과 대조합니다.
  ATIF의 메타데이터 파일명·HTTP/S3 저장·서명·재시도·원격 실패 시 로컬 보존을 연결했습니다.
- 기존 Piper TTS 공급자를 순수 Python PiperVoice·VITS 실행기에 연결했습니다.
  eSpeak·TEXT·아랍어·히브리어·태국어·중국어·일본어 발음 변환과 원본 보조 모델을 포함합니다.
  실제 lessac 음성 모델을 사용한 도구 호출의 출력은 원본 PCM과 최대 1 LSB 차이이며,
  여러 화자·합성 설정·프로필별 캐시·예열·해제도 검증합니다.
  원본 eSpeak 데이터 373개와 음성·사전·음소 실행기·연결 단계를 포함하고,
  연속 문장 사이의 버퍼·음소표 상태도 원본과 대조합니다.
  음성 모델은 원래와 같이 사용자 데이터이며, 첫 사용 시 데이터 다운로드를 지원합니다.
  태국어는 원본 알고리즘·숫자 확장·통계 데이터를 유지하고 실행에 사용되지 않는
  초기화 리소스 3,263,655바이트를 제외했습니다. 이 수치는 배포 파일 압축 후의 절감량과는 다릅니다.
  중국어는 전체 G2PW 모델을 사용한 29개 발음·ID·실제 합성 경로가 두 Python 버전에서 원본과 일치합니다.
  내부 확률값은 엄격한 오차 기준을 통과하지 못했으며, 이 실패를 발음 결과와 구분해 기록합니다.
  일본어는 전체 MeCab·Sudachi 사전과 기본 읽기 교정을 포함하고, 원본 9,001개 입력의
  분석·126,017개 문맥 레이블 및 58개 PiperVoice 경로를 대조했습니다.
  두 언어의 작은 실제 ONNX 음성 모델로 PCM·WAV도 대조하며, 학습된 해당 언어 음성의 품질 검증을 뜻하지는 않습니다.
  배포 파일별 검증 결과는
  [BUILD_REPORT.md](standalone/BUILD_REPORT.md)에 기록합니다.
- 원본 openWakeWord TFLite 모델 3개의 Python 추론과 스트리밍 상태·확인 횟수·pause/resume.
  20초 기준 스트림의 6개 임계값별 트리거 순서가 원본과 일치합니다. 현재 측정 호스트에서는
  80ms 입력당 중앙값 178ms로 실시간보다 느리며, 연속 입력에서는 원래 큐 규칙에 따라 프레임이 버려질 수 있습니다.

아직 남은 범위에는 추가 전사 입력 형식·로컬 TTS와 Sherpa/Porcupine 엔진, Opus의 SILK/hybrid/FEC 송신,
복잡한 글꼴 shaping·일부 이미지 코덱·Relay 확장과 선택 OS 기능이 있습니다. 파이썬 암호 연산은 상수 시간
실행을 보장하지 않습니다. 실제 macOS/Windows·장시간 운용·서비스 계정 검증은 별도입니다.
원본 기능을 조용히 성공 처리하거나 전역 가짜 SDK 모듈을 주입하지 않습니다.

**검증**

```sh
python3.14 -I -S standalone/check.py
python3.14 -I -S standalone/check.py --artifact dist/hermes.py --artifact-only
python3.14 -I -S standalone/upstream_check.py --help
```

검증은 임시 홈, 독립된 로컬 HTTP/WebSocket/gRPC/stdio 상대, 공개 테스트 키와 독립 코덱
fixture를 사용합니다. 실제 배포물 검사는 별도 경로로 복사한 파일에서 관리 워커와 DB 영수증을
확인합니다. 원본 회귀 검사는 변경 영역에 맞춰 별도로 실행합니다.
CI의 OS 행렬 설정은 해당 OS에서 실제로 통과했다는 증거와 구분합니다.
