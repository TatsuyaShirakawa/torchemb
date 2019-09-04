import torch
import torch.nn as nn
import torch.nn.functional as F


EuclideanEmbedding = nn.Embedding


class EuclideanSpace(object):

    def __init__(self):
        pass

    def distance(self, x, y):
        return torch.norm(x - y, dim=-1)

    def norm(self, x):
        return torch.norm(x, dim=-1)

    def dot(self, x, y):
        return (x * y).sum(-1)


if __name__ == '__main__':

    embs = EuclideanEmbedding(10, 2)
