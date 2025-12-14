from pytorch_metric_learning.losses import NTXentLoss
import torch
from torch import nn

nxt_loss = NTXentLoss(temperature=0.5)

N_repeat = 256
N = 4
feature_dim = 128
temperature = 1
loss = 0
embeddings = torch.randn(N, N_repeat, feature_dim)
embeddings_mean = torch.mean(embeddings, dim=1, keepdim=True)

instance_labels = torch.arange(0, N).repeat_interleave(N_repeat)

positive_cosine_sim = nn.functional.cosine_similarity(embeddings,
                                                      embeddings_mean,
                                                      dim=2)[..., None]
exp_cosine_sim_nom = torch.exp(positive_cosine_sim / temperature).sum(dim=-1)

negative_cosine_sim = nn.functional.cosine_similarity(
    embeddings.view(1, -1, feature_dim),
    embeddings.view(-1, 1, feature_dim),
    dim=-1).view(N, N, N_repeat, N_repeat)

mask = torch.ones(N_repeat, N_repeat).fill_diagonal_(0, wrap=True).view(1, 1, N_repeat, N_repeat).repeat(N, N, 1, 1)
exp_cosine_sim_denom = torch.exp((negative_cosine_sim * mask).sum(dim=-1).sum(dim=1) / temperature)

loss = torch.log(exp_cosine_sim_nom) - torch.log(exp_cosine_sim_denom + exp_cosine_sim_nom)
