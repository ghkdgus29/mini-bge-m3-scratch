# 실행 방법

## 바로 실행하기

별도 설치 없이 바로 실행하면 됩니다. 모델과 데이터는 **첫 실행 시 HuggingFace 에서 자동으로 받습니다**.

```bash
uv sync                  # 처음 한 번: 의존성 설치 (.venv 생성)
uv run main.py           # 실제 데이터로 전체 파이프라인
```

## 실제 데이터 실행 시 자동으로 받는 파일

| 종류 | HuggingFace 출처 | 크기 |
|---|---|---|
| 모델 (xlm-roberta-base) | `FacebookAI/xlm-roberta-base` | ~1.1 GB |
| Wikipedia (Stage 0, 1) | `wikimedia/wikipedia` 20231101.en | ~420 MB |
| SQuAD train (Stage 2) | `rajpurkar/squad` | ~15 MB |
| SQuAD validation (평가) | `rajpurkar/squad` | ~2 MB |

파일은 HuggingFace 캐시(`~/.cache/huggingface/`)에 저장되므로, 두 번째 실행부터는 다시 받지 않습니다.

## 로컬 파일 사용 (선택)

미리 받아 둔 파일이 있으면 자동으로 그 파일을 씁니다.

```
data/raw/
  wikipedia/train-00000-of-00041.parquet
  squad/train-00000-of-00001.parquet
  squad/validation-00000-of-00001.parquet
models/
  xlm-roberta-base/   (config.json, model.safetensors, tokenizer.json, tokenizer_config.json)
```

다른 모델을 쓰려면 `--model 경로또는모델ID` 옵션을 사용합니다.

## 옵션

```
uv run main.py --device cpu|cuda|mps   # 장치 지정 (기본: 자동 선택)
uv run main.py --steps 20 50 50 50     # 단계별 step 수 (S0 S1 S2a S2b)
uv run main.py --model xlm-roberta-large  # 다른 모델
uv run main.py --no-skd                # self-knowledge distillation 끄기 (논문 표 5 ablation)
```

## 소요 시간 참고

실제 데이터 전체 실행(`uv run main.py`)은 CPU(i5-13400) 기준 **약 1시간 반** 예상.
Stage 0 ~2분, Stage 1 ~22분, Stage 2a ~25분이며 단계마다 평가가 몇 분씩 더 걸립니다.

## Windows 와 Mac

같은 명령으로 돌아갑니다. 장치는 자동 선택됩니다: NVIDIA GPU(cuda) > Apple Silicon(mps) > CPU.
Mac 에서 mps 오류가 나면 `--device cpu` 로 실행하세요.
