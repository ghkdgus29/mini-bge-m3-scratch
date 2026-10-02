"""데이터 준비: 사람이 읽을 수 있는 실제 텍스트 (영어 Wikipedia + SQuAD), 그리고 빠른 동작 확인용 smoke 데이터

반환 형식은 모두 같다.
  wiki : [(제목, 본문 문단), ...]   라벨 없는 대량 텍스트에서 자동 추출한 쌍 -> Stage 0, 1
  lab  : [(질문, 정답 문단), ...]   사람이 라벨링한 쌍                        -> Stage 2
  test : [(질문, 정답 문단), ...]   평가용 (학습에 쓰지 않은 문서)
"""

import json
import random
from pathlib import Path

CACHE = Path(__file__).parent / "data"  # 원본에서 추려 낸 결과를 저장해 두는 곳 (두 번째 실행부터는 이 파일만 읽는다)
RAW = CACHE / "raw"  # 직접 받아 둔 원본 파일 위치 (SETUP.md 참고)
LOCAL_WIKI = RAW / "wikipedia" / "train-00000-of-00041.parquet"


# ============ 실제 데이터 ============
def wiki_pairs(batches, n, exclude_titles):
    """Wikipedia 문서 -> (제목, 첫 번째 충분히 긴 문단). 논문 §3.1 의 비지도 데이터가 이런 구조 추출이다
    ((제목, 본문), (제목, 초록), (지시문, 출력) 등). 사람이 라벨을 달지 않았다.
    문서당 문단 하나만 쓴다: 같은 제목이 한 배치에 두 번 들어가면 서로를 negative 로 오해하기 때문."""
    pairs = []
    for titles, texts in batches:
        for title, text in zip(titles, texts):
            if title in exclude_titles:
                continue
            paras = [p.strip() for p in text.split("\n\n") if 40 <= len(p.split()) <= 250]
            if paras:
                pairs.append((title, paras[0]))
        if len(pairs) >= n:
            break
    return pairs[:n]


def one_per_context(rows, n, rng):
    """SQuAD 는 한 문단에 질문이 여러 개라서, 문단당 질문 하나만 남긴다 (남기면 같은 문단이 서로 다른 쿼리의
    정답이 되어, in-batch negative 가 사실은 정답인 상황이 생긴다)."""
    seen = {}
    for r in rows:
        seen.setdefault(r["context"], r)
    out = [(r["question"], r["context"]) for r in seen.values()]
    rng.shuffle(out)
    return out[:n]


def _need(path):
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다. SETUP.md 를 보고 파일을 받아 이 위치에 넣어 주세요.")
    return path


def _wiki_batches():
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(_need(LOCAL_WIKI))  # 파일이 커서 한 번에 다 읽지 않고 앞에서부터 조금씩 읽는다
    for b in pf.iter_batches(batch_size=200, columns=["title", "text"]):
        yield b["title"].to_pylist(), b["text"].to_pylist()


def _squad_rows(split):
    import pyarrow.parquet as pq

    path = _need(RAW / "squad" / f"{split}-00000-of-00001.parquet")
    return pq.read_table(path, columns=["title", "context", "question"]).to_pylist()


def real_data(n_wiki, n_lab, n_test, seed=0):
    cache = CACHE / f"real_{n_wiki}_{n_lab}_{n_test}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    rng = random.Random(seed)
    train, dev = _squad_rows("train"), _squad_rows("validation")
    dev_titles = {r["title"].replace("_", " ") for r in dev}  # 평가에 쓸 문서가 Stage 0/1 에 섞이지 않게 제외
    data = {
        "wiki": wiki_pairs(_wiki_batches(), n_wiki, dev_titles),
        "lab": one_per_context(train, n_lab, rng),
        "test": one_per_context(dev, n_test, rng),
    }
    CACHE.mkdir(exist_ok=True)
    cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


# ============ smoke 데이터 (빠른 동작 확인용: 작고 가짜지만 전체 단계를 1~2분에 돌릴 수 있다) ============
def smoke_data(seed=0):
    """가상의 마을 설명 문서를 템플릿으로 만든다. 사람이 읽을 수는 있지만 내용은 가짜다."""
    rng = random.Random(seed)
    syl = ["al", "bro", "cor", "den", "el", "fen", "gar", "hol", "ir", "jun", "kel", "lor", "mar", "nor", "os", "pel"]
    names = list({(rng.choice(syl) + rng.choice(syl) + rng.choice(syl)).capitalize() for _ in range(400)})[:180]
    rivers = ["Aren", "Belo", "Cinder", "Dova", "Esk", "Fyrn", "Gale", "Hush"]
    goods = ["wool", "salt", "apple", "copper", "timber", "honey", "pottery", "silk"]
    towns = []
    for name in names:
        river, good, year = rng.choice(rivers), rng.choice(goods), rng.randint(1100, 1900)
        para = (
            f"{name} is a small town in the northern part of the country. It was founded in {year} on the banks of the "
            f"{river} river. For centuries the town has been famous for its {good} market, which attracts merchants from "
            f"nearby villages every spring. The old stone bridge of {name} is a popular place for visitors."
        )
        towns.append((name, para, f"Which river runs through {name}?", f"When was {name} founded?", f"What is {name} famous for?"))
    pair = lambda t, k: (t[k], t[1])
    return {
        "wiki": [(t[0], t[1]) for t in towns[:120]],
        "lab": [pair(t, rng.choice([2, 3, 4])) for t in towns[120:150]],
        "test": [pair(t, rng.choice([2, 3, 4])) for t in towns[150:180]],
    }


# ============ 합성 데이터 (논문 §3.1 / 부록 A.2) -- 이 프로젝트에서는 실행하지 않음 ============
SYNTH_PROMPT = (
    "You are a curious AI assistant, please generate one specific and valuable question based on the following text. "
    "The generated question should revolve around the core content of this text, and avoid using pronouns (e.g., "
    '"this"). Note that you should generate only one question, without including additional content:'
)  # 논문 부록 A.2 의 프롬프트 그대로


def synthesize_questions(articles, llm, rng=None):
    """[생략: 호출하지 않는다] LLM 으로 (질문, 긴 문서) 쌍을 만드는 합성 데이터 단계.

    ## 이 단계에서 실제로 하는 일 (논문 §3.1 "synthetic data", MultiLongDoc 데이터셋)
      1) Wikipedia, Wudao, mC4 에서 **긴 문서**를 고른다 (평균 4,000~9,000 토큰).
      2) 문서에서 문단 하나를 무작위로 뽑는다.
      3) LLM(논문은 GPT-3.5)에게 "이 문단에 대한 질문을 하나 만들어라"고 시킨다.
      4) (생성된 질문, **문단이 아니라 문서 전체**)를 positive 쌍으로 저장한다.

    ## 왜 하는가
      - 사람이 만든 검색 라벨 데이터(MS MARCO, NQ, SQuAD 등)는 대부분 **짧은 문단** 단위라서,
        8,192 토큰짜리 긴 문서 안에서 정답 위치를 찾는 능력을 가르칠 데이터가 거의 없다.
      - 질문은 문서의 한 문단에서만 나오므로, 모델은 긴 문서 전체를 읽고도 그 한 문단의 내용을
        문서 벡터에 반영해야 맞힐 수 있다 -> 긴 문서 검색 학습 신호가 된다.
      - 여러 언어 문서로 만들면 다국어 파인튜닝 데이터가 늘어나는 효과도 있다.

    ## 예시
      문서(위키 '에펠탑' 전체, 약 5,000 토큰)에서 뽑은 문단:
        "The tower was built from 1887 to 1889 as the entrance arch to the 1889 World's Fair ..."
      LLM 이 만든 질문:  "For which event was the Eiffel Tower constructed as an entrance arch?"
      학습 쌍:           (위 질문,  에펠탑 문서 전체)   <- 대명사("this")를 쓰지 않게 하는 이유는, 질문만 봐도
                                                          무슨 얘기인지 알 수 있어야 검색 쿼리로 쓸 수 있어서다.

    ## 학습 목적: Stage 2 (파인튜닝)의 라벨 데이터에 섞어서, 긴 입력 검색 능력(multi-granularity)을 기른다.

    ## 이 프로젝트에서 생략한 이유
      - LLM API 호출이 필요하다 (논문은 GPT-3.5). 비용과 API 키가 든다.
      - 이 프로젝트는 문서를 128 토큰으로 자르고 영어만 쓰므로, 긴 문서 학습 신호도 다국어 효과도 얻을 수 없다.
      - SQuAD 라는 사람 라벨 쌍이 이미 있어 Stage 2 를 돌리는 데 부족하지 않다.

    사용하려면:  lab += synthesize_questions(긴_문서_리스트, llm=lambda prompt: my_llm(prompt))
      articles: [{"title": str, "text": str}, ...]    llm: 프롬프트 문자열 -> 응답 문자열 함수
    (긴 문서를 쓰려면 main.py 의 PLEN 을 늘리고 모델의 최대 길이도 확장해야 한다. 논문은 RetroMAE 단계에서
     위치 임베딩을 8,192 로 확장했다.)"""
    rng = rng or random.Random(0)
    pairs = []
    for a in articles:
        paras = [p for p in a["text"].split("\n\n") if len(p.split()) >= 40]
        if paras:
            question = llm(SYNTH_PROMPT + "\n" + rng.choice(paras)).strip()
            pairs.append((question, a["text"]))  # positive 는 뽑은 문단이 아니라 문서 전체
    return pairs
