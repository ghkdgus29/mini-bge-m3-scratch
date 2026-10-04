"""BGE-M3 학습 파이프라인 미니 재현 (논문: arXiv 2402.03216) -- 실제 영어 텍스트, 사전학습 XLM-R 에서 시작

논문의 학습 순서를 그대로, 규모만 줄여서 따라간다.
  Stage 0  RetroMAE        : 라벨 없는 텍스트(Wikipedia)로 인코더 사전학습               (§B.1)
  Stage 1  비지도 대조학습    : (제목, 본문) 쌍, dense 점수만, in-batch negative만          (§3.3)
  Stage 2  파인튜닝         : SQuAD (질문, 정답 문단) + hard negative(ANCE식) 7개          (§B.1)
           2a 워밍업        : 세 점수를 함께 학습하되 증류는 아직 안 함 (식 4)
           2b 통합 학습     : self-knowledge distillation (s_inter 가 선생님)
  + 효율적 배치            : 길이별 그룹 배치, split-batch + gradient checkpointing        (§3.4)
  (합성 데이터 단계는 data.synthesize_questions 에 설명과 함께 구현만 해 두고 실행하지 않는다)

실행
  uv run main.py           실제 데이터. 모델과 데이터셋은 첫 실행 때 HuggingFace 에서 자동으로 받는다
  옵션: --no-skd (논문 표 5 ablation)  --device cpu|cuda|mps  --steps S0 S1 S2a S2b  --model 모델ID또는폴더
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Mac(MPS)에서 지원 안 되는 연산은 CPU 로 대체

import torch  # noqa: E402

from data import real_data  # noqa: E402
from m3 import M3, W, embed, loss_plain, loss_skd, nce, retromae_loss, scores  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 인코딩 때문에 한글이 깨지는 것 방지

N_NEG = 7  # 쿼리당 hard negative 수 (논문 §B.1)
HN_RANGE = (10, 210)  # hard negative 를 뽑는 순위 범위. 공식 hn_mine.py 기본값 (상위 10위는 정답일 위험이 있어 건너뜀)
HEAD_MULT = 20  # 새로 붙인 헤드/디코더는 무작위 초기화라 인코더보다 학습률을 크게 준다 (논문은 워밍업 6000 step 으로 해결)
PLEN, QLEN, TLEN = 128, 64, 32  # 문서 / 질문 / 제목 최대 토큰 수


# ============ 장치 ============
def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon (Mac)
        return torch.device("mps")
    return torch.device("cpu")


# ============ 모델 / 토크나이저 ============
def build(a):
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    from transformers import logging as hf_logging

    hf_logging.set_verbosity_error()

    def split(mlm):  # MLM 모델 -> (인코더, MLM 출력층). 인코더는 이후 M3 의 공유 인코더가 되고, 출력층은 Stage 0 에서만 쓴다
        return mlm.base_model, (mlm.lm_head if hasattr(mlm, "lm_head") else mlm.cls)

    # 로컬 폴더가 있으면 그대로 쓰고, 없으면 HuggingFace Hub 에서 자동으로 내려받는다
    return AutoTokenizer.from_pretrained(a.model), *split(AutoModelForMaskedLM.from_pretrained(a.model))


# ============ 효율적 배치 (§3.4) ============
def pad(seqs, pad_id, dev):
    ids = torch.full((len(seqs), max(map(len, seqs))), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = s
    return ids.to(dev), (ids != pad_id).to(dev)


def batch_stream(lens, B, grouped=True, width=16):
    """grouped=True: 길이가 비슷한 샘플끼리 한 배치로 묶어 패딩을 줄인다 (논문: 길이 구간별로 그룹을 만들어 그 안에서 샘플링)."""
    while True:
        order = torch.randperm(len(lens)).tolist()
        chunks = []
        if grouped:
            groups = {}
            for i in order:
                groups.setdefault(lens[i] // width, []).append(i)
            chunks = [g[i : i + B] for g in groups.values() for i in range(0, len(g) - B + 1, B)]
            random.shuffle(chunks)
        if not chunks:  # 그룹이 너무 잘게 쪼개져 한 배치도 못 만들면 그냥 섞는다
            chunks = [order[i : i + B] for i in range(0, len(order) - B + 1, B)]
        yield from chunks


def pad_ratio(lens, grouped, B, n=30):
    s, tot, used = batch_stream(lens, B, grouped), 0, 0
    for _ in range(n):
        c = [lens[i] for i in next(s)]
        tot, used = tot + max(c) * len(c), used + sum(c)
    return 1 - used / tot


# ============ 학습 루프 ============
def fit(model, name, steps, lr, step_fn):
    model.train()
    new = [p for n, p in model.named_parameters() if n.startswith(("sparse_head", "colbert_head", "decoder"))]
    old = [p for n, p in model.named_parameters() if not n.startswith(("sparse_head", "colbert_head", "decoder"))]
    opt = torch.optim.AdamW([{"params": old, "lr": lr}, {"params": new, "lr": lr * HEAD_MULT}], weight_decay=0.01)
    warm = max(1, int(0.1 * steps))  # warmup ratio 0.1, weight decay 0.01 (논문 §B.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, (steps - s) / max(1, steps - warm)))
    t0 = time.time()
    for s in range(steps):
        loss = step_fn()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if s % max(1, steps // 10) == 0 or s == steps - 1:
            print(f"  [{name}] step {s + 1:3d}/{steps}  loss {loss.item():.3f}  ({time.time() - t0:.0f}s)", flush=True)


# ============ 평가 (논문 표 1 형식: dense / sparse / multi / dense+sparse / all) ============
@torch.no_grad()
def evaluate(model, title, qs, ps, pad_id, dev, chunk=25):
    model.eval()
    Q, P = embed(model, *pad(qs, pad_id, dev)), embed(model, *pad(ps, pad_id, dev))
    out = [scores(Q.rows(i, i + chunk), P) for i in range(0, len(qs), chunk)]
    sd, ss, sm = (torch.cat(x) for x in zip(*out))
    cols = {"dense": sd, "sparse": ss, "multi": sm, "dense+sparse": W[0] * sd + W[1] * ss, "all": W[0] * sd + W[1] * ss + W[2] * sm}
    ranks = {k: (s > s.diag()[:, None]).sum(1) for k, s in cols.items()}  # 정답 문서보다 점수가 높은 문서 수
    print(f"\n  == 평가: {title} (문서 {len(ps)}개 중 질문의 정답 문서 찾기) ==")
    print("  " + " " * 8 + "".join(f"{k:>14}" for k in cols))
    print("  " + f"{'R@1':<8}" + "".join(f"{(r == 0).float().mean():>14.3f}" for r in ranks.values()))
    print("  " + f"{'MRR@10':<8}" + "".join(f"{torch.where(r < 10, 1 / (r + 1.0), 0.0).mean():>14.3f}" for r in ranks.values()))
    print()


# ============ 전체 파이프라인 ============
def main(a):
    random.seed(0)
    torch.manual_seed(0)
    dev = pick_device(a.device)
    T0 = time.time()

    # ---- 데이터 ----
    data = real_data(n_wiki=1500, n_lab=400, n_test=300)
    b0, b1, b2, steps = 16, 32, 4, a.steps or (20, 50, 50, 50)
    tok, encoder, lm_head = build(a)
    model = M3(encoder, tok.all_special_ids, lm_head).to(dev)
    pad_id = tok.pad_token_id
    print(f"장치 {dev} | 모델 {a.model} 파라미터 {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

    def ids(texts, ml):
        return [torch.tensor(x) for x in tok(texts, truncation=True, max_length=ml)["input_ids"]]

    wiki_t, wiki_p = ids([t for t, _ in data["wiki"]], TLEN), ids([p for _, p in data["wiki"]], PLEN)
    lab_q, lab_p = ids([q for q, _ in data["lab"]], QLEN), ids([p for _, p in data["lab"]], PLEN)
    test_q, test_p = ids([q for q, _ in data["test"]], QLEN), ids([p for _, p in data["test"]], PLEN)

    # ---- 데이터가 어떻게 생겼는지 한 건씩 보여준다 ----
    title, para = data["wiki"][0]
    q, ctx = data["lab"][0]
    print(f"\n[Stage 0/1 데이터] 문서 {len(wiki_p)}개. 예) 제목 = {title!r}\n  본문 = {para[:150]!r}...")
    print(f"  토큰화: {tok.convert_ids_to_tokens(wiki_p[0][:10].tolist())} ... (총 {len(wiki_p[0])}개)")
    print(f"[Stage 2 데이터] 질문-정답문단 {len(lab_p)}쌍. 예) 질문 = {q!r}\n  정답 문단 = {ctx[:150]!r}...")
    print(f"[평가 데이터] 학습에 안 쓴 문서 {len(test_p)}개와 질문 {len(test_q)}개")
    lens = [len(p) for p in wiki_p]
    print(f"패딩 비율: 랜덤 배치 {pad_ratio(lens, False, b1):.1%}  ->  길이별 그룹 배치 {pad_ratio(lens, True, b1):.1%}")

    # ---- Stage 0: RetroMAE ----
    print("\nStage 0  RetroMAE (라벨 없는 텍스트로 인코더 사전학습)")
    s0 = batch_stream(lens, b0)

    def step0():
        return retromae_loss(model, *pad([wiki_p[i] for i in next(s0)], pad_id, dev), tok.mask_token_id)

    fit(model, "retromae", steps[0], 5e-5, step0)

    # ---- Stage 1: 비지도 대조학습 (dense 만, in-batch negative 만, 큰 배치) ----
    print("\nStage 1  비지도 대조학습 ((제목, 본문) 쌍, dense only, in-batch negative)")
    s1 = batch_stream(lens, b1)

    def step1():
        idx = next(s1)
        Q = embed(model, *pad([wiki_t[i] for i in idx], pad_id, dev))
        P = embed(model, *pad([wiki_p[i] for i in idx], pad_id, dev))
        return nce(Q.dense @ P.dense.T, torch.arange(len(idx), device=dev))  # 같은 배치의 다른 문서가 전부 negative

    fit(model, "unsup", steps[1], 3e-5, step1)
    evaluate(model, "Stage 1 직후 (sparse/multi 헤드는 아직 학습 안 함)", test_q, test_p, pad_id, dev)

    # ---- hard negative 채굴 (ANCE 식: 현재 모델의 dense 검색으로 헷갈리는 문서를 찾는다) ----
    @torch.no_grad()
    def dense_of(seqs):
        model.eval()
        return torch.cat([embed(model, *pad(seqs[i : i + 32], pad_id, dev)).dense for i in range(0, len(seqs), 32)])

    s = dense_of(lab_q) @ dense_of(lab_p).T
    top = s.topk(min(HN_RANGE[1], len(lab_p)), dim=1).indices.tolist()  # 쿼리마다 가까운 순서대로 문서 순위
    hard = []  # hard[i] = 쿼리 i 의 hard negative 후보 목록 (학습 때 여기서 7개를 뽑는다)
    for i, t in enumerate(top):
        cand = [j for j in t[HN_RANGE[0] :] if j != i]  # 상위 10위는 건너뛰고, 정답은 제외
        if len(cand) < N_NEG:  # 후보가 모자라면 아무 문서로 채운다 (공식 hn_mine.py 도 그렇게 한다)
            cand += random.sample([j for j in range(len(lab_p)) if j != i and j not in cand], N_NEG - len(cand))
        hard.append(cand)
    print(f"\nhard negative 채굴 예) 질문 = {data['lab'][0][0]!r}\n  모델이 정답 다음으로 헷갈린 문서 = {data['lab'][hard[0][0]][1][:150]!r}...")

    # ---- Stage 2: 파인튜닝 ----
    G = 1 + N_NEG  # 쿼리당 후보 수: [positive, hard negative x7]
    s2 = batch_stream([len(p) for p in lab_p], b2)

    def step2(loss_fn):
        idx = next(s2)
        cand = [c for i in idx for c in [lab_p[i]] + [lab_p[j] for j in random.sample(hard[i], N_NEG)]]
        Q = embed(model, *pad([lab_q[i] for i in idx], pad_id, dev))
        P = embed(model, *pad(cand, pad_id, dev))  # 후보 b2*8개: 다른 쿼리의 후보도 전부 negative
        return loss_fn(*scores(Q, P), torch.arange(len(idx), device=dev) * G)  # 정답 위치 = i*G

    print("\nStage 2a 워밍업 (세 점수를 함께 학습하되 증류는 아직 안 함: 식 4)")
    fit(model, "warmup", steps[2], 2e-5, lambda: step2(loss_plain))
    evaluate(model, "Stage 2a 직후", test_q, test_p, pad_id, dev)

    print(f"Stage 2b 통합 학습 ({'self-knowledge distillation: 식 4 + 식 6' if not a.no_skd else 'SKD 없음 (ablation): 식 4 계속'})")
    fit(model, "unified", steps[3], 1e-5, lambda: step2(loss_skd if not a.no_skd else loss_plain))
    evaluate(model, "Stage 2b 직후 (최종)", test_q, test_p, pad_id, dev)
    print(f"전체 소요 시간 {(time.time() - T0) / 60:.1f}분")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-skd", action="store_true", help="self-knowledge distillation 끄기")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    ap.add_argument("--model", default="FacebookAI/xlm-roberta-base", help="시작 모델: HuggingFace 모델 ID 또는 로컬 폴더 (논문은 xlm-roberta-large)")
    ap.add_argument("--steps", type=int, nargs=4, metavar=("S0", "S1", "S2A", "S2B"), help="단계별 학습 step 수")
    main(ap.parse_args())
