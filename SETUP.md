# 설치: 모델과 데이터 직접 받기

`uv run main.py`(실제 데이터)를 돌리려면 아래 **파일을 직접 받아서** 지정한 폴더에 넣어야 합니다.
코드는 인터넷에서 아무것도 내려받지 않고, 이 폴더의 파일만 읽습니다. 파일이 없으면 어느 위치에 넣어야 하는지 알려 주는 오류가 납니다.

> 모델/데이터 파일 없이 코드 흐름만 빠르게 확인하려면 이 문서는 필요 없고 `uv run main.py --smoke` 만 실행하면 됩니다 (작은 가짜 데이터와 작은 모델, 1~2분).

## 1. 모델: xlm-roberta-base (약 1.1GB)

저장소: <https://huggingface.co/FacebookAI/xlm-roberta-base/tree/main>

| 파일 | 크기 | 직접 링크 |
|---|---|---|
| `config.json` | 작음 | [받기](https://huggingface.co/FacebookAI/xlm-roberta-base/resolve/main/config.json) |
| `model.safetensors` | 1.1GB | [받기](https://huggingface.co/FacebookAI/xlm-roberta-base/resolve/main/model.safetensors) |
| `tokenizer.json` | 9MB | [받기](https://huggingface.co/FacebookAI/xlm-roberta-base/resolve/main/tokenizer.json) |
| `tokenizer_config.json` | 작음 | [받기](https://huggingface.co/FacebookAI/xlm-roberta-base/resolve/main/tokenizer_config.json) |

저장 위치: `models/xlm-roberta-base/`

같은 저장소의 나머지 파일(`pytorch_model.bin`, `tf_model.h5`, `flax_model.msgpack`, `model.onnx` 등)은 받지 않아도 됩니다.
(`sentencepiece.bpe.model`, 5MB 은 `tokenizer.json` 이 있으면 필요 없지만 같이 넣어 두면 안전합니다.)

## 2. SQuAD: Stage 2 학습과 평가용 (약 16MB)

저장소: <https://huggingface.co/datasets/rajpurkar/squad/tree/main/plain_text>

| 파일 | 크기 | 직접 링크 |
|---|---|---|
| `train-00000-of-00001.parquet` | 14.5MB | [받기](https://huggingface.co/datasets/rajpurkar/squad/resolve/main/plain_text/train-00000-of-00001.parquet) |
| `validation-00000-of-00001.parquet` | 1.8MB | [받기](https://huggingface.co/datasets/rajpurkar/squad/resolve/main/plain_text/validation-00000-of-00001.parquet) |

저장 위치: `data/raw/squad/`

## 3. Wikipedia: Stage 0, 1 학습용 (약 420MB)

저장소: <https://huggingface.co/datasets/wikimedia/wikipedia/tree/main/20231101.en>

| 파일 | 크기 | 직접 링크 |
|---|---|---|
| `train-00000-of-00041.parquet` | 420MB | [받기](https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.en/train-00000-of-00041.parquet) |

저장 위치: `data/raw/wikipedia/` (저장소 안의 `20231101.en/` 폴더 이름은 붙이지 않고, 파일만 넣습니다)

41개 조각 중 첫 번째만 쓰고, 그 안에서 문서 1,500개만 뽑아 씁니다.

## 최종 폴더 구조

```
bge-m3-scratch/
├── main.py  m3.py  data.py  toy.py
├── models/
│   └── xlm-roberta-base/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── sentencepiece.bpe.model   (선택)
└── data/
    └── raw/
        ├── squad/
        │   ├── train-00000-of-00001.parquet
        │   └── validation-00000-of-00001.parquet
        └── wikipedia/
            └── train-00000-of-00041.parquet
```

`models/` 와 `data/` 는 `.gitignore` 에 들어 있어서 git 에 올라가지 않습니다.

## 실행

```
uv sync                  # 처음 한 번: 의존성 설치 (.venv 생성)
uv run main.py --smoke   # 1) 먼저 빠른 동작 확인 (모델/데이터 파일 불필요, 1~2분)
uv run main.py           # 2) 실제 데이터로 전체 파이프라인
```

- `models/xlm-roberta-base/` 가 있으면 자동으로 그 모델을 씁니다. 다른 모델/폴더는 `--model 경로` 로 지정합니다.
- 첫 실행에서 원본 parquet 에서 필요한 부분만 뽑아 `data/real_*.json` 으로 저장합니다. 두 번째 실행부터는 이 파일을 읽습니다 (원본을 지우고 싶다면 이 파일을 먼저 만든 뒤에 지우세요).
- 학습 step 수 조절: `--steps 20 50 50 50` (Stage 0, 1, 2a, 2b 순서)

## Windows 와 Mac

- 같은 명령으로 돌아갑니다. 의존성은 `uv sync` 가 OS 에 맞게 설치합니다 (`.venv` 는 OS 마다 따로 만들어야 하므로 폴더를 복사하지 말고 각 PC 에서 `uv sync`).
- 장치는 자동 선택됩니다: NVIDIA GPU(cuda) > Apple Silicon(mps) > CPU.
  Mac 에서 mps 오류가 나면 `--device cpu` 로 실행하세요.
- 모델/데이터 폴더를 다른 PC 로 옮길 때는 `models/` 와 `data/raw/` 만 복사하면 됩니다.

## 참고

- 실제 데이터 전체 실행(`uv run main.py`)은 CPU(i5-13400)에서 **약 1시간 반**이 걸릴 것으로 예상합니다.
  측정한 값은 Stage 0 약 2분, Stage 1 약 22분, Stage 2a 약 25분이고, 평가가 단계마다 몇 분씩 더 듭니다.
- 이 시간은 공식 저장소 코드에 맞춰 고치기 전 버전으로 측정한 것입니다. 고친 뒤에는 `--smoke` 와 실제 모델의 Stage 0 loss 계산만 확인했고,
  실제 데이터 전체 실행은 하지 않았습니다. 논문/공식 코드와의 차이와 결과 해석은 [DIFFERENCES.md](DIFFERENCES.md) 를 보세요.
