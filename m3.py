"""BGE-M3 핵심 부품: 세 가지 헤드를 가진 모델, split-batch, 점수 함수, 손실, RetroMAE (논문 arXiv 2402.03216)"""

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

W = (1.0, 0.3, 1.0)  # 통합 점수 가중치 w1,w2,w3 (논문 §3.3)
LAM = (1.0, 0.1, 1.0)  # 손실 가중치 λ1,λ2,λ3 (논문 §3.3)
TAU = 0.02  # InfoNCE 온도. 논문 본문에는 값이 없고, 공식 저장소(FlagEmbedding)의 기본값이 0.02


class Enc(NamedTuple):
    """인코딩 결과 한 묶음. 문장 N개에 대해 세 가지 표현이 한꺼번에 들어 있다."""

    dense: torch.Tensor  # (N, H)    [CLS] 위치 벡터 하나 (정규화됨)
    w: torch.Tensor  # (N, L)    토큰별 sparse 가중치 (ReLU)
    vec: torch.Tensor  # (N, L, H) 토큰별 multi-vector (정규화됨)
    ids: torch.Tensor  # (N, L)    입력 토큰 ID (sparse 점수에서 같은 토큰을 찾는 데 필요)
    mask: torch.Tensor  # (N, L)    진짜 토큰이면 True, 패딩이면 False

    def rows(self, i, j):
        return Enc(*(x[i:j] for x in self))


class M3(nn.Module):
    def __init__(self, encoder, special_ids, lm_head):
        super().__init__()
        H = encoder.config.hidden_size
        self.encoder = encoder  # 1) 공유 인코더 (사전학습된 XLM-R 계열)
        self.lm_head = lm_head  # Stage 0(RetroMAE) 에서만 쓰는 사전학습된 MLM 출력층 (공식 RetroMAE 코드도 이걸 쓴다)
        self.sparse_head = nn.Linear(H, 1)  # 3-b) W_lex: 은닉 벡터 -> 숫자 하나
        self.colbert_head = nn.Linear(H, H)  # 3-c) W_mul: d x d 투영
        self.decoder = nn.TransformerEncoderLayer(H, encoder.config.num_attention_heads, H * 4, batch_first=True)
        self.register_buffer("special", torch.tensor(sorted(set(special_ids))))  # [CLS] 등 (sparse 에서 제외)

    def encode(self, ids, mask):
        return self.encoder(input_ids=ids, attention_mask=mask.long()).last_hidden_state  # (B, L, H)

    def forward(self, ids, mask):
        h = self.encode(ids, mask)
        dense = F.normalize(h[:, 0], dim=-1)  # 3-a) dense: [CLS] 위치 벡터 하나
        keep = mask & ~torch.isin(ids, self.special)  # 특수 토큰·패딩은 sparse 어휘에서 제외
        w = F.relu(self.sparse_head(h)).squeeze(-1) * keep  # 3-b) sparse: 토큰별 중요도
        vec = F.normalize(self.colbert_head(h), dim=-1)  # 3-c) multi-vector: 토큰마다 벡터 하나씩
        return dense, w, vec


def sub_batch_size(length):
    """공식 구현(compute_sub_batch_size)의 표: 입력이 길수록 sub-batch 를 작게 쪼갠다."""
    for l, b in [(6000, 1), (5000, 2), (4000, 3), (3000, 3), (2000, 5), (1000, 9), (512, 16), (0, 32)]:
        if length >= l:
            return b


def embed(model, ids, mask, sub=None):
    """split-batch (논문 Algorithm 1, §3.4): 큰 배치를 sub-batch 로 쪼개 하나씩 인코딩하고 임베딩만
    이어붙인다. 학습 중에는 gradient checkpointing 으로 중간 활성값을 버렸다가 backward 때 다시
    계산하므로, 같은 메모리로 훨씬 큰 배치(=더 많은 in-batch negative)를 쓸 수 있다.
    (공식 구현은 HF 의 층 단위 checkpointing 을 쓰고, 여기서는 sub-batch 전체를 checkpoint 로 감싼다. 원리는 같다.
     논문은 여기에 GPU 간 임베딩 broadcast 도 더하지만 단일 장치라 생략)"""
    sub = sub or sub_batch_size(ids.size(1))
    run =(lambda i, m: checkpoint(model, i, m, use_reentrant=False)) if torch.is_grad_enabled() else model
    outs = [run(i, m) for i, m in zip(ids.split(sub), mask.split(sub))]
    dense, w, vec = (torch.cat(x) for x in zip(*outs))
    return Enc(dense, w, vec, ids, mask)


# ============ 점수 함수 ============
def sparse_scores(q, p):
    """s_lex = 쿼리·문서에 같이 나온 토큰의 (쿼리 가중치 x 문서 가중치) 합. 같은 토큰이 여러 번 나오면 max.
    어휘 전체(25만) 크기 벡터를 만들면 낭비라, 이번에 등장한 토큰끼리만 번호를 다시 매겨서 계산한다."""
    uniq, inv = torch.unique(torch.cat([q.ids.flatten(), p.ids.flatten()]), return_inverse=True)
    inv_q, inv_p = inv[: q.ids.numel()].view_as(q.ids), inv[q.ids.numel() :].view_as(p.ids)

    def vocab_vec(x, inv_x):
        z = torch.zeros(x.ids.size(0), len(uniq), device=x.w.device)
        return z.scatter_reduce(1, inv_x, x.w, "amax")  # 같은 토큰은 max (w >= 0 이라 0 초기값 안전)

    return vocab_vec(q, inv_q) @ vocab_vec(p, inv_p).T


def scores(q, p):
    """쿼리 Nq개 x 문서 Np개 -> (Nq, Np) 점수표 3장 (dense, sparse, multi-vector)"""
    s_d = q.dense @ p.dense.T
    s_s = sparse_scores(q, p)
    # multi-vector 는 [CLS] 를 뺀 나머지 토큰만 쓴다 (공식 구현: last_hidden_state[:, 1:])
    qm, pm = q.mask[:, 1:], p.mask[:, 1:]
    sim = torch.einsum("qid,pjd->qpij", q.vec[:, 1:], p.vec[:, 1:]).masked_fill(~pm[None, :, None, :], -1e4)
    s_m = (sim.max(-1).values * qm[:, None, :]).sum(-1) / qm.sum(-1)[:, None]  # 쿼리 토큰별 최고 매칭의 평균
    return s_d, s_s, s_m


# ============ 손실 ============
def nce(s, target):  # InfoNCE (식 2): target 열이 정답, 같은 행의 나머지 후보는 전부 negative
    return F.cross_entropy(s / TAU, target)


def _plain(sd, ss, sm, target):  # 식 (3), (4): 통합 점수 s_inter 와, 증류 없는 손실 L
    si = W[0] * sd + W[1] * ss + W[2] * sm
    L = (LAM[0] * nce(sd, target) + LAM[1] * nce(ss, target) + LAM[2] * nce(sm, target) + nce(si, target)) / 4
    return si, L


def loss_plain(sd, ss, sm, target):  # 증류를 끈 손실 = 식 (4). 공식 구현에서 use_self_distill=False 일 때의 손실
    return _plain(sd, ss, sm, target)[1]


def loss_skd(sd, ss, sm, target):  # self-knowledge distillation, 식 (3)~(6)
    si, L = _plain(sd, ss, sm, target)
    t = F.softmax(si.detach() / TAU, -1)  # 선생님 분포. 정답 신호로만 쓰므로 gradient 는 차단
    kd = lambda s: -(t * F.log_softmax(s / TAU, -1)).sum(-1).mean()  # 식 (5)
    L2 = (LAM[0] * kd(sd) + LAM[1] * kd(ss) + LAM[2] * kd(sm)) / 3
    return (L + L2) / 2


# ============ Stage 0: RetroMAE ============
def mask_tokens(ids, mask, rate, special, mask_id):
    m = (torch.rand(ids.shape, device=ids.device) < rate) & mask & ~torch.isin(ids, special)
    return ids.masked_fill(m, mask_id), m


def pick(m, k):  # 가려진 위치 중 최대 k개만 골라 손실 계산 (어휘가 25만이라 전부 계산하면 CPU 에서 너무 무겁다)
    pos = m.nonzero()
    pos = pos[torch.randperm(len(pos), device=pos.device)[:k]]
    return pos[:, 0], pos[:, 1]


def retromae_loss(model, ids, mask, mask_id, k=128):
    """인코더는 30%를 가린 입력을 읽고, 얕은 디코더는 50%를 가린 입력 + 인코더의 [CLS] 벡터만 받아
    원문을 복원한다. 디코더가 약하고 입력도 많이 가려져 있어서, 복원하려면 [CLS] 벡터에 문장 정보를
    꾹꾹 담아야 한다 -> [CLS] 가 문장 전체를 요약하도록 인코더가 학습된다.
    마스킹 비율(0.3 / 0.5)과 사전학습된 MLM 출력층 사용은 공식 저장소의 RetroMAE 코드와 같다.
    (공식은 whole-word 마스킹 + enhanced decoding(위치별 어텐션 마스크)을 쓰는데, 여기서는 토큰 단위 마스킹 +
     일반 Transformer 층 1개인 단순 버전)"""
    enc_in, m_enc = mask_tokens(ids, mask, 0.3, model.special, mask_id)
    h = model.encode(enc_in, mask)
    r, c = pick(m_enc, k)
    loss_enc = F.cross_entropy(model.lm_head(h[r, c]), ids[r, c])

    dec_in, m_dec = mask_tokens(ids, mask, 0.5, model.special, mask_id)
    x = model.encoder.embeddings(input_ids=dec_in)  # 단어 + 위치 임베딩
    x = torch.cat([h[:, :1], x[:, 1:]], 1)  # 위치 0 을 인코더의 [CLS] 은닉으로 교체
    d = model.decoder(x, src_key_padding_mask=~mask)
    r, c = pick(m_dec, k)
    loss_dec = F.cross_entropy(model.lm_head(d[r, c]), ids[r, c])
    return loss_enc + loss_dec
