import torch
import torch.nn as nn
from torchemb.utils import acosh


class PseudoEuclideanEmbedding1(nn.Module):
    def __init__(self, num_embeddings, positive_embedding_dim, negative_embedding_dim,
                 padding_idx=None):
        super(PseudoEuclideanEmbedding, self).__init__()
        self.positive_embedding_dim = positive_embedding_dim
        self.negative_embedding_dim = negative_embedding_dim
        self.padding_idx = padding_idx
        self.positive_emb = nn.Embedding(num_embeddings, positive_embedding_dim,
                                         padding_idx=padding_idx)
        self.negative_emb = nn.Embedding(num_embeddings, negative_embedding_dim,
                                         padding_idx=padding_idx)

    def forward(self, input):
        x = self.positive_emb(input)
        y = self.negative_emb(input)
        return x, y


class PseudoEuclideanEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, positive_embedding_dim, negative_embedding_dim,
                 padding_idx=None):
        self.positive_embedding_dim = positive_embedding_dim
        self.negative_embedding_dim = negative_embedding_dim
        embedding_dim = positive_embedding_dim + negative_embedding_dim
        super(PseudoEuclideanEmbedding, self).__init__(num_embeddings=num_embeddings,
                                                       embedding_dim=embedding_dim,
                                                       padding_idx=padding_idx)


class PseudoEuclideanSpace(object):
    def __init__(self, positive_dim, negative_dim):
        self.positive_dim = positive_dim
        self.negative_dim = negative_dim
        
    def dot(self, x, y):
        assert(self.positive_dim + self.negative_dim == x.shape[-1])
        assert(self.positive_dim + self.negative_dim == y.shape[-1])        
        x1, x2 = x[..., :self.positive_dim], x[..., self.positive_dim:]
        y1, y2 = y[..., :self.positive_dim], y[..., self.positive_dim:]
        return (x1 * y1).sum(-1) - (x2 * y2).sum(-1)

    def hyperbolic_angle(self, x, y):
        nx = self.dot(x, x)
        ny = self.dot(y, y)
        denom = torch.clamp(torch.sqrt(torch.abs(nx * ny)), 1.0e-6)
        assert(not torch.any(torch.isnan(nx)))
        assert(not torch.any(torch.isnan(ny)))
        assert(not torch.any(torch.isnan(self.dot(x, y))))
        print(torch.abs(self.dot(x, y)) / denom)
        assert(not torch.any(torch.abs(self.dot(x, y) / denom) < 1))
        return acosh(torch.abs(self.dot(x, y)) / denom)
