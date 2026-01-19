import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools


class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        temperature_init = 0.07
        self.temp = nn.Parameter(
            torch.log(torch.tensor(temperature_init, dtype=torch.float32))
        )

    def forward(self, embeddings, **kwargs):
        B, n, d = embeddings.shape
        loss = 0.0
        for i, j in itertools.combinations(range(n), 2):
            logits = torch.matmul(embeddings[:, i], embeddings[:, j].T) * torch.exp(
                self.temp
            )
            labels = torch.arange(
                logits.shape[0], device=logits.device, dtype=torch.long
            )
            loss += F.cross_entropy(logits, labels)
        return loss
    

class SigLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()

        temperature_init = 10.0
        bias_init = -10.0
        self.temp = nn.Parameter(
            torch.log(torch.tensor(temperature_init, dtype=torch.float32))
        )
        self.bias = nn.Parameter(torch.tensor(bias_init, dtype=torch.float32))

    def forward(self, embeddings, **kwargs):
        B, n, d = embeddings.shape
        loss = 0.0
        for i, j in itertools.combinations(range(n), 2):
            logits = (
                torch.matmul(embeddings[:, i], embeddings[:, j].T)
                * torch.exp(self.temp)
                + self.bias
            )
            eye = torch.eye(B).to(logits.device)
            labels = 2 * eye - torch.ones_like(logits)
            loglik = F.logsigmoid(labels * logits)
            nll = -torch.sum(loglik, axis=-1)
            loss += torch.mean(nll)
        return loss
    

class MultiPlaceMSELoss(nn.Module):
    def __init__(self, lmbd=1.0):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.lmbd = lmbd

    def forward(self, features, **kwargs):
        B, n, c, h, w = features.shape
        loss = torch.zeros(1, device=features.device)
        for i, j in itertools.combinations(range(n), 2):
            loss += self.mse_loss(features[:, i], features[:, j])
        return loss * self.lmbd
    
class MultiPlaceCosSimLoss(nn.Module):
    def __init__(self, lmbd=1.0):
        super().__init__()
        self.lmbd = lmbd
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, features, **kwargs):
        B, n, c, h, w = features.shape
        loss = torch.zeros(1, device=features.device)
        for i, j in itertools.combinations(range(n), 2):
            cos_sim = self.cosine_similarity(
                features[:, i].reshape(B, -1), features[:, j].reshape(B, -1)
            )
            loss += torch.mean(1.0 - cos_sim)
        return loss * self.lmbd
