# bge-m3-scratch

BGE-M3(논문 *M3-Embedding*, [arXiv 2402.03216](https://arxiv.org/abs/2402.03216))의 **학습 과정을 작은 규모로 직접 따라 해 보며 공부하는 리포**입니다.
논문의 학습 순서를 그대로 구현하되, 데이터와 모델 크기만 CPU에서 돌아갈 만큼 줄였습니다.

## BGE-M3가 뭔가

문서 검색용 임베딩 모델입니다. 하나의 인코더가 세 가지 점수를 동시에 냅니다.

- **dense**: [CLS] 벡터 하나로 문장을 표현
- **sparse**: 토큰별 중요도 (BM25 같은 단어 일치 방식을 학습으로)
- **multi-vector**: 토큰마다 벡터 하나씩 (ColBERT 방식)

세 점수를 서로 가르치는 self-knowledge distillation(SKD)이 핵심 아이디어입니다.

## 학습 순서

| 단계 | 하는 일 | 데이터 |
|---|---|---|
| Stage 0 | RetroMAE: 라벨 없는 텍스트로 인코더 사전학습 | 위키 문단 |
| Stage 1 | 비지도 대조학습 (dense만, in-batch negative) | 위키 (제목, 본문) 쌍 |
| Stage 2a | 세 점수 학습 (증류 전) + hard negative 7개 | SQuAD (질문, 정답 문단) |
| Stage 2b | SKD로 통합 학습 | SQuAD |

모델은 사전학습된 `xlm-roberta-base`에서 시작합니다. 학습 후 SQuAD 개발셋 300문서에서 dense / sparse / multi-vector / 조합 점수를 비교합니다.

## 파일

| 파일 | 내용 |
|---|---|
| [main.py](main.py) | 전체 파이프라인 (단계별 학습, 평가) |
| [m3.py](m3.py) | 모델(세 헤드), 점수 함수, 손실, RetroMAE, split-batch |
| [data.py](data.py) | 데이터 준비 (위키, SQuAD, smoke용 가짜 데이터, 합성 데이터 함수) |
| [SETUP.md](SETUP.md) | 모델과 데이터를 직접 받아서 넣는 방법 |
| [DIFFERENCES.md](DIFFERENCES.md) | 논문, 공식 구현(FlagEmbedding)과 이 프로젝트의 차이 |

## 실행

```bash
uv sync                  # 의존성 설치

uv run main.py --smoke   # 빠른 동작 확인: 작은 가짜 영어 문서 + 작은 모델로 전체 단계를 1~2분에 돌려 봄
uv run main.py           # 실제 데이터 전체 (CPU에서 약 1시간 반 예상)
```

`uv run main.py`는 모델과 데이터를 인터넷에서 받지 않고 `models/`, `data/raw/`의 파일을 읽습니다. 먼저 [SETUP.md](SETUP.md)대로 파일을 넣어 주세요.

옵션: `--no-skd` (증류 끄기) · `--device cpu|cuda|mps` · `--steps S0 S1 S2A S2B` (단계별 step 수)

Windows와 Mac 모두 같은 명령으로 동작합니다. VS Code에서는 F5로 디버깅할 수 있습니다 (`.vscode/launch.json`).

## 주의

- 데이터가 논문의 수십만~수천만 분의 일이라, 나오는 수치는 논문과 비교할 수 없습니다. 각 단계가 어떻게 이어지고 손실이 어떻게 작동하는지 보는 용도입니다.
- 실제 데이터 전체 실행은 공식 구현에 맞춰 고친 뒤로는 아직 하지 않았습니다 (`--smoke`와 일부 구성요소만 확인). 자세한 상태는 [DIFFERENCES.md](DIFFERENCES.md)를 보세요.
