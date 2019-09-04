'''
A rouch implementation of the following paper

M. Nickel+, "Poincaré Embeddings for Learning Hierarchical Representations", NeurIPS2017
https://arxiv.org/pdf/1705.08039.pdf
'''
import argparse
from pathlib import Path
import itertools
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (IterableDataset,
                              RandomSampler,
                              DataLoader)
from gensim.corpora import Dictionary
from tqdm import tqdm
from torchemb.poincare_ball import PoincareBallEmbedding, PoincareBall


parser = argparse.ArgumentParser('Poincare Embeddings')
parser.add_argument('data_file', type=Path)
parser.add_argument('result_file', type=Path)
parser.add_argument('-e', '--embedding_dim', default=10, type=int)
parser.add_argument('-m', '--max_epoch', default=200, type=int)
parser.add_argument('-s', '--samples_per_iter', default=10000, type=int)
parser.add_argument('-b', '--batch_size', default=512, type=int)
parser.add_argument('-n', '--neg_size', default=10, type=int)
parser.add_argument('-c', default=1, type=float)
parser.add_argument('--lr', default=1.0e-3, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--seed', default=0, type=int)
args = parser.parse_args()
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)


def load_data(data_file):
    data_file = Path(data_file)
    pairs = []
    with data_file.open() as fin:
        for line in fin:
            a, b = line.strip().split('\t')
            pairs.append((a, b))
    d = Dictionary(pairs)
    pairs = np.asarray([d.doc2idx(pair) for pair in pairs])
    return d, pairs


dictionary, pairs = load_data(args.data_file)
print(len(dictionary), len(pairs))


class Dataset(IterableDataset):
    def __init__(self, pairs, neg_size):
        self.pairs = pairs
        self.neg_size = neg_size
        self.pair_sampler = RandomSampler(list(range(len(self.pairs))), replacement=True)

    def __iter__(self):
        pair_iter = itertools.cycle(iter(self.pair_sampler))
        while True:
            idx = next(pair_iter)
            x, y = self.pairs[idx]
            ys = [y] + [self.pairs[next(pair_iter)][1] for _ in range(self.neg_size - 1)]
            yield x, torch.LongTensor(ys)


data = DataLoader(Dataset(pairs, args.neg_size),
                  batch_size=args.batch_size)

embeddings = PoincareBallEmbedding(len(dictionary), args.embedding_dim, c=args.c)
manifold = PoincareBall(c=args.c)

loss = nn.CrossEntropyLoss()
warmup_rate = 100
lr = args.lr *  (args.batch_size ** 0.5) / (args.embedding_dim ** 0.5)
optimizer = optim.Adam(embeddings.parameters(), lr=lr)
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                gamma=math.exp(math.log(0.01) / args.max_epoch))


def train(embeddings, loss, optimizer, data, samples_per_iter):
    embeddings.train()
    data_iter = iter(data)
    avg_loss_ = 0
    N = samples_per_iter // args.batch_size
    for i in tqdm(range(N)):
        idx1, idx2 = next(data_iter)
        x, ys = embeddings(idx1), embeddings(idx2)
        assert(not torch.any(torch.isnan(x)))
        assert(not torch.any(torch.isnan(ys)))        
        ds = manifold.distance(x[:, None, :].expand_as(ys), ys)
        logits = -ds
        loss_ = loss(logits, torch.zeros(len(logits), dtype=torch.long))
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()
        avg_loss_ += loss_.item()
    avg_loss_ /= N
    print('train loss: {:.5f}'.format(avg_loss_))


def save(embeddings, dictionary, result_file):
    embeddings.eval()
    result_file = Path(result_file)
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        with result_file.open('w') as fout:
            for i, c in sorted(dictionary.dfs.items()):
                e = embeddings(torch.LongTensor([i]))[0]
                print(dictionary[i], *[_.item() for _ in e], sep='\t', file=fout)


for epoch in range(args.max_epoch):
    print('epoch:', epoch + 1, '/', args.max_epoch)
    train(embeddings, loss, optimizer, data, args.samples_per_iter)
    lr_scheduler.step()
    save(embeddings, dictionary, args.result_file)
