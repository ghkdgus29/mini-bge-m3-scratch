"""BGE-M3 학습 과정 미니 재현 (논문: arXiv 2402.03216, 토이 데이터, CPU에서 수 분)

논문의 학습 순서를 그대로 축소해서 따라간다.
  Stage 0  RetroMAE        : 라벨 없는 텍스트로 인코더 사전학습 (§B.1)
  Stage 1  비지도 대조학습    : 약한 쌍, dense 점수만, in-batch negative만 (§3.3)
  Stage 2  파인튜닝         : 정제된 쌍 + hard negative(ANCE식) 7개 (§B.1)
           2a 워밍업        : dense / sparse / multi-vector 를 각자 따로 학습
           2b 통합 학습     : self-knowledge distillation (s_inter 가 선생님)
  + 효율적 배치            : 길이별 그룹 배치, split-batch + gradient checkpointing (§3.4)

실행: uv run main.py            (SKD 켠 상태)
      uv run main.py --no-skd   (SKD 없이 2b를 각자 학습으로 대체: 논문 표 5 ablation)
"""

import argparse
import random
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

PAD, CLS, MASK, NSPECIAL = 0, 1, 2, 3
V, D, MAXLEN = 500, 64, 65  # 어휘 크기, 차원, 최대 길이([CLS] 포함)
LENS = (16, 32, 64)  # 문서 본문 길이 (길이별 그룹 배치를 보이기 위해 섞어둠)
N_TOPIC, TOPIC_WORDS = 40, 12

W = (1.0, 0.3, 1.0)  # 통합 점수 가중치 w1,w2,w3 (논문 §3.3)
LAM = (1.0, 0.1, 1.0)  # 손실 가중치 λ1,λ2,λ3 (논문 §3.3)
TAU = 0.05  # 온도 (논문 본문에는 값이 없어 임의로 지정)
N_NEG = 7  # 쿼리당 hard negative 수 (논문 §B.1)
SUB = 32  # split-batch 의 sub-batch 크기


# ============ 토이 데이터 ============
# 토이 '언어': 주제마다 전용 단어가 있고, 문서 = 주제 단어 60% + 아무 단어 40%.
# 같은 주제의 다른 문서가 hard negative 가 된다.
def make_corpus(n, topics):
    out = []
    for _ in range(n):
        L, t = random.choice(LENS), topics[random.randrange(N_TOPIC)]
        pick = t[torch.randint(0, TOPIC_WORDS, (L,))]
        body = torch.where(torch.rand(L) < 0.6, pick, torch.randint(NSPECIAL, V, (L,)))
        out.append(torch.cat([torch.tensor([CLS]), body]))
    return out


def make_query(p, noise):  # 문서에서 토큰 6~12개를 뽑고, noise 비율만큼 엉뚱한 토큰으로 교체
    k = random.randint(6, 12)
    q = p[torch.randperm(len(p) - 1)[:k].sort().values + 1].clone()
    flip = torch.rand(k) < noise
    q[flip] = torch.randint(NSPECIAL, V, (int(flip.sum()),))
    return torch.cat([torch.tensor([CLS]), q])


def pad(seqs):
    ids = torch.zeros(len(seqs), max(map(len, seqs)), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = s
    return ids, ids != PAD


# ============ 효율적 배치 (§3.4) ============
def batch_stream(lens, B, grouped=True):
    """grouped=True: 같은 길이의 샘플끼리 한 배치로 묶어 패딩을 줄인다."""
    while True:
        order = torch.randperm(len(lens)).tolist()
        if grouped:
            groups = {}
            for i in order:
                groups.setdefault(lens[i], []).append(i)
            chunks = [g[i : i + B] for g in groups.values() for i in range(0, len(g) - B + 1, B)]
            random.shuffle(chunks)
        else:
            chunks = [order[i : i + B] for i in range(0, len(order) - B + 1, B)]
        yield from chunks


def pad_ratio(lens, grouped, B=64, n=30):
    s, tot, used = batch_stream(lens, B, grouped), 0, 0
    for _ in range(n):
        c = [lens[i] for i in next(s)]
        tot, used = tot + max(c) * len(c), used + sum(c)
    return 1 - used / tot


# ============ 모델 ============
class MiniM3(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, D, padding_idx=PAD)
        self.pos = nn.Embedding(MAXLEN, D)
        mk = lambda n: nn.TransformerEncoder(
            nn.TransformerEncoderLayer(D, 4, D * 4, batch_first=True), n, enable_nested_tensor=False
        )
        self.encoder = mk(2)  # 공유 인코더
        self.decoder = mk(1)  # RetroMAE 용 얕은 디코더 (Stage 0 에서만 사용)
        self.sparse_head = nn.Linear(D, 1)  # W_lex
        self.colbert_head = nn.Linear(D, D)  # W_mul

    def encode(self, ids, mask):
        x = self.tok(ids) + self.pos(torch.arange(ids.size(1)))
        return self.encoder(x, src_key_padding_mask=~mask)  # (B, L, D)

    def forward(self, ids, mask):
        h = self.encode(ids, mask)
        # dense: [CLS] 위치 벡터 하나
        dense = F.normalize(h[:, 0], dim=-1)
        # sparse: 토큰별 가중치(ReLU) -> 어휘 크기 벡터, 같은 토큰은 max. 특수 토큰은 제외
        w = F.relu(self.sparse_head(h)).squeeze(-1) * mask * (ids >= NSPECIAL)
        sparse = torch.zeros(len(ids), V).scatter_reduce(1, ids, w, "amax")
        # multi-vector: 토큰마다 벡터 하나씩
        colbert = F.normalize(self.colbert_head(h), dim=-1)
        return dense, sparse, colbert


def embed(model, ids, mask, sub=SUB):
    """split-batch (논문 Algorithm 1): 큰 배치를 sub-batch 로 쪼개 gradient checkpointing 으로
    인코딩하고 임베딩만 이어붙인다. 중간 활성값을 버리므로 같은 메모리로 더 큰 배치(=더 많은
    in-batch negative)를 쓸 수 있다. (논문은 여기에 GPU 간 임베딩 broadcast 도 더하지만 CPU 라 생략)"""
    outs = [checkpoint(model, i, m, use_reentrant=False) for i, m in zip(ids.split(sub), mask.split(sub))]
    return tuple(torch.cat(x) for x in zip(*outs))


# ============ Stage 0: RetroMAE ============
def mask_tokens(ids, mask, rate):
    m = (torch.rand(ids.shape) < rate) & mask & (ids >= NSPECIAL)
    return ids.masked_fill(m, MASK), m


def retromae_loss(model, ids, mask):
    """인코더는 30%를 가린 입력을 읽고, 얕은 디코더는 70%를 가린 입력 + 인코더의 [CLS] 벡터만 받아
    원문을 복원한다. 디코더가 약하므로 복원하려면 [CLS] 벡터에 문장 정보를 꾹꾹 담아야 한다.
    (원 RetroMAE 의 enhanced decoding 은 생략한 단순화 버전)"""
    enc_in, m_enc = mask_tokens(ids, mask, 0.3)
    h = model.encode(enc_in, mask)
    loss_enc = F.cross_entropy(h[m_enc] @ model.tok.weight.T, ids[m_enc])

    dec_in, m_dec = mask_tokens(ids, mask, 0.7)
    x = model.tok(dec_in) + model.pos(torch.arange(ids.size(1)))
    x = torch.cat([h[:, :1], x[:, 1:]], 1)  # 위치 0 을 인코더의 [CLS] 은닉으로 교체
    d = model.decoder(x, src_key_padding_mask=~mask)
    loss_dec = F.cross_entropy(d[m_dec] @ model.tok.weight.T, ids[m_dec])
    return loss_enc + loss_dec


# ============ 점수 함수와 손실 ============
def scores(q, p, qmask, pmask):
    """쿼리 Nq개 x 문서 Np개 -> (Nq, Np) 점수표 3장 (dense, sparse, multi-vector)"""
    s_d = q[0] @ p[0].T
    s_s = q[1] @ p[1].T  # 공통 토큰의 가중치 곱의 합
    sim = torch.einsum("qid,pjd->qpij", q[2], p[2]).masked_fill(~pmask[None, :, None, :], -1e4)
    s_m = (sim.max(-1).values * qmask[:, None, :]).sum(-1) / qmask.sum(-1)[:, None]  # 쿼리 토큰별 최고 매칭 평균
    return s_d, s_s, s_m


def nce(s, target):  # InfoNCE (식 2): target 열이 정답, 나머지는 전부 negative
    return F.cross_entropy(s / TAU, target)


def loss_independent(sd, ss, sm, target):  # 워밍업 / SKD 없는 ablation: 각자 따로
    return (LAM[0] * nce(sd, target) + LAM[1] * nce(ss, target) + LAM[2] * nce(sm, target)) / 3


def loss_skd(sd, ss, sm, target):  # 식 (3)~(6)
    si = W[0] * sd + W[1] * ss + W[2] * sm  # 통합 점수 s_inter
    L = (LAM[0] * nce(sd, target) + LAM[1] * nce(ss, target) + LAM[2] * nce(sm, target) + nce(si, target)) / 4
    t = F.softmax(si.detach() / TAU, -1)  # 선생님 분포. 정답 신호로만 쓰므로 gradient 차단
    kd = lambda s: -(t * F.log_softmax(s / TAU, -1)).sum(-1).mean()  # 식 (5)
    L2 = (LAM[0] * kd(sd) + LAM[1] * kd(ss) + LAM[2] * kd(sm)) / 3
    return (L + L2) / 2


# ============ 학습 루프 ============
def fit(model, name, steps, lr, step_fn):
    model.train()
    t0 = time.time()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warm = max(1, int(0.1 * steps))  # warmup ratio 0.1, weight decay 0.01 (논문 §B.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, (steps - s) / (steps - warm)))
    for s in range(steps):
        loss = step_fn()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if s % (steps // 4) == 0 or s == steps - 1:
            print(f"  [{name}] step {s:3d}  loss {loss.item():.3f}  ({time.time() - t0:.0f}s)")


# ============ 평가 (논문 표 1 형식: dense / sparse / multi / dense+sparse / all) ============
@torch.no_grad()
def evaluate(model, title, qs, ps, chunk=50):
    model.eval()
    (qi, qm), (pi, pm) = pad(qs), pad(ps)
    Q, P = embed(model, qi, qm), embed(model, pi, pm)
    out = [scores(tuple(x[i : i + chunk] for x in Q), P, qm[i : i + chunk].float(), pm) for i in range(0, len(qs), chunk)]
    sd, ss, sm = (torch.cat(x) for x in zip(*out))
    cols = {"dense": sd, "sparse": ss, "multi": sm, "dense+sparse": W[0] * sd + W[1] * ss, "all": W[0] * sd + W[1] * ss + W[2] * sm}
    print(f"\n  == 평가: {title} (문서 {len(ps)}개 중 쿼리의 원문 찾기) ==")
    print("  " + " " * 8 + "".join(f"{k:>14}" for k in cols))
    ranks = {k: (s > s.diag()[:, None]).sum(1) for k, s in cols.items()}
    print("  " + f"{'R@1':<8}" + "".join(f"{(r == 0).float().mean():>14.3f}" for r in ranks.values()))
    print("  " + f"{'MRR@10':<8}" + "".join(f"{torch.where(r < 10, 1 / (r + 1.0), 0.0).mean():>14.3f}" for r in ranks.values()))
    print()


# ============ 전체 파이프라인 ============
def main(skd=True):
    random.seed(0)
    torch.manual_seed(0)
    topics = torch.randint(NSPECIAL, V, (N_TOPIC, TOPIC_WORDS))
    unsup = make_corpus(3000, topics)  # 라벨 없는 대량 텍스트 (Stage 0, 1)
    lab = make_corpus(800, topics)  # 정제된 라벨 데이터 (Stage 2)
    test = make_corpus(400, topics)  # 평가용: 학습에 안 쓴 문서
    lab_q = [make_query(p, 0.1) for p in lab]
    test_q = [make_query(p, 0.1) for p in test]
    model = MiniM3()

    lens = [len(p) for p in unsup]
    print(f"패딩 비율: 랜덤 배치 {pad_ratio(lens, False):.1%}  ->  길이별 그룹 배치 {pad_ratio(lens, True):.1%}\n")

    # ---- Stage 0: RetroMAE ----
    print("Stage 0  RetroMAE (라벨 없는 텍스트로 인코더 사전학습)")
    s0 = batch_stream(lens, 64)

    def step0():
        ids, mask = pad([unsup[i] for i in next(s0)])
        return retromae_loss(model, ids, mask)

    fit(model, "retromae", 300, 1e-3, step0)

    # ---- Stage 1: 비지도 대조학습 (dense 만, in-batch negative 만, 큰 배치) ----
    print("\nStage 1  비지도 대조학습 (약한 쌍, dense only, in-batch negative)")
    s1 = batch_stream(lens, 128)

    def step1():
        idx = next(s1)
        p = [unsup[i] for i in idx]
        (qi, qm), (pi, pm) = pad([make_query(x, 0.3) for x in p]), pad(p)  # 쿼리마다 노이즈 30%
        Q, P = embed(model, qi, qm)[0], embed(model, pi, pm)[0]
        return nce(Q @ P.T, torch.arange(len(idx)))

    fit(model, "unsup", 300, 1e-3, step1)
    evaluate(model, "Stage 1 직후 (sparse/multi 헤드는 아직 학습 안 함)", test_q, test)

    # ---- hard negative 채굴 (ANCE 식: 현재 모델의 dense 검색으로 헷갈리는 문서를 찾는다) ----
    with torch.no_grad():
        model.eval()
        (qi, qm), (pi, pm) = pad(lab_q), pad(lab)
        s = embed(model, qi, qm)[0] @ embed(model, pi, pm)[0].T
        s.fill_diagonal_(-1e4)  # 정답 제외
        hard = s.topk(30, dim=1).indices.tolist()  # 쿼리마다 상위 30개 중에서 7개를 뽑아 쓴다

    # ---- Stage 2: 파인튜닝 ----
    G = 1 + N_NEG  # 쿼리당 후보 수: [positive, negative x7]
    s2 = batch_stream([len(p) for p in lab], 32)

    def step2(loss_fn):
        idx = next(s2)
        cand = [c for i in idx for c in [lab[i]] + [lab[j] for j in random.sample(hard[i], N_NEG)]]
        (qi, qm), (pi, pm) = pad([lab_q[i] for i in idx]), pad(cand)
        Q, P = embed(model, qi, qm), embed(model, pi, pm)  # 후보 256개: 다른 쿼리의 후보도 전부 negative
        sd, ss, sm = scores(Q, P, qm.float(), pm)
        return loss_fn(sd, ss, sm, torch.arange(len(idx)) * G)  # 정답 위치 = i*G

    print("\nStage 2a 워밍업 (dense / sparse / multi-vector 각자 따로)")
    fit(model, "warmup", 200, 5e-4, lambda: step2(loss_independent))
    evaluate(model, "Stage 2a 직후", test_q, test)

    print(f"Stage 2b 통합 학습 ({'self-knowledge distillation' if skd else 'SKD 없음 (ablation)'})")
    fit(model, "unified", 200, 3e-4, lambda: step2(loss_skd if skd else loss_independent))
    evaluate(model, "Stage 2b 직후 (최종)", test_q, test)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-skd", action="store_true", help="self-knowledge distillation 끄기")
    main(skd=not ap.parse_args().no_skd)
