import torch
import torch.nn as nn
import torch.nn.functional as F
from torchemb.utils import acosh, atanh, EPS


class PoincareBallModifyGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, c):
        ctx.save_for_backward(x, c)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        '''
        first-order approximation of the Riemannian gradient descent
        '''
        x, c = ctx.saved_tensors
        x2 = (torch.norm(x, dim=-1, keepdim=True) / c) ** 2
        m = torch.clamp(1 - x2, EPS) / 2
        modified_grad_output =  grad_output * (m ** 2)
        return modified_grad_output, None


class PoincareBallEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 c=1, scale_grad_by_freq=False,
                 sparse=False, _weight=None):
        self.c = c  #  preset c here as floating point because Embedding.__init__ calls reset_parameters but parameter c cannot be set before __init__ is called
        super(PoincareBallEmbedding, self).__init__(num_embeddings=num_embeddings,
                                                    embedding_dim=embedding_dim,
                                                    padding_idx=padding_idx,
                                                    max_norm=(1 - 1.0e-7) / c, norm_type=2,
                                                    scale_grad_by_freq=scale_grad_by_freq,
                                                    sparse=sparse, _weight=_weight)
        self.c = nn.Parameter(torch.FloatTensor([c]), requires_grad=False)  # reset c

    def reset_parameters(self, mean=0.0, std=1.0e-2):
        std = std / (self.embedding_dim ** 0.5) / self.c
        nn.init.normal_(self.weight, mean, std)
        self.weight
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)
    
    def forward(self, input):
        x = super(PoincareBallEmbedding, self).forward(input)
        return PoincareBallModifyGradient.apply(x, self.c)


class PoincareBall(object):

    def __init__(self, c):
        self.c = c

    def distance(self, x, y):
        '''
        1 + 2 * ||x - y||^2 / (1 - ||x||^2) / (1 - ||y||^2)
        = (1 - ||x||^2 - ||y||^2 + ||x||^2 ||y||^2 + 2(||x||^2 - 2 ||x|| ||y|| + ||y||^2)
        = 1 + ||x||^2 + ||y||^2 + ||x||^2 ||y||^2 - 4 ||x|| ||y||
        '''
        c2 = self.c ** 2
        x2 = torch.norm(x, dim=-1) ** 2 *  c2
        y2 = torch.norm(y, dim=-1) ** 2 *  c2
        x_y = torch.norm(x - y, dim=-1) ** 2 * c2
        z = 1 + 2 * x_y / torch.clamp(1 - x2, 1.0e-6) / torch.clamp(1 - y2, 1.0e-6)
        assert(not torch.any(torch.isnan(z)))
        return acosh(z)

    def norm(self, x):
        return self.distance(torch.zeros_like(x), x)
        return 2 * atanh(torch.norm(x, dim=-1))  # = distance(self, 0, x)

    def scalar_mult(self, r, x):
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        xx = x / x_norm
        return torch.tanh(r * atanh(x_norm * self.c)) / self.c * x / x_norm

    def gyro_add(self, x, y):
        c2 = self.c ** 2
        x2 = torch.norm(x, dim=-1, keepdim=True) ** 2 * c2
        y2 = torch.norm(y, dim=-1, keepdim=True) ** 2 * c2
        xy = (x * y).sum(dim=-1, keepdim=True) * c2
        denom = torch.clamp(1 + 2 * xy + x2 * y2, 1.0e-6)
        a = (1 + 2 * xy + y2) / denom
        b = torch.clamp(1 - x2, EPS) / denom
        assert(not torch.any(torch.isnan(a)))
        assert(not torch.any(torch.isnan(b)))
        assert(a.abs().max() < 1000)
        assert(b.abs().max() < 1000)        
        return a * x + b * y

    def gyro_sub(self, x, y):
        return self.gyro_add(x, -y)

    def gyro_neg(self, x):
        return self.gyro_sub(torch.zeros_like(x), x)

    def gyro_coadd(self, x, y):
        c2 = self.c ** 2
        x2 = torch.norm(x, dim=-1, keepdim=True) ** 2 * c2
        y2 = torch.norm(y, dim=-1, keepdim=True) ** 2 * c2
        denom = torch.clamp(1 - x2 * y2, 1.0e-6)
        a = torch.clamp(1 - y2, EPS) / denom
        b = torch.clamp(1 - x2, EPS) / denom
        return a * x + b * y

    def gyro_cosub(self, x, y):
        return self.gyro_coadd(x, self.gyro_neg(y))


if __name__ == '__main__':

    c = 1
    embs = PoincareBallEmbedding(5, 3, c=c)
    print(embs.weight)

    idx1 = torch.LongTensor([0, 1, 3])
    idx2 = torch.LongTensor([0, 2, 4])

    print('distance', PoincareBall(c).distance(embs(idx1), embs(idx2)))    
    print('add', PoincareBall(c).gyro_add(embs(idx1), embs(idx2)).norm(dim=1))
    print('coadd', PoincareBall(c).gyro_coadd(embs(idx1), embs(idx2)).norm(dim=1))

    print(embs(torch.LongTensor([[0, 1], [2, 3]])).shape)

    print(PoincareBall(c).distance(torch.FloatTensor([0.3, 0]),
                                   torch.FloatTensor([0.999, 0])))
    x = embs(idx1)
    y = embs(idx2)
    print('x', x)
    print('y', y)
    print(PoincareBall(c).distance(x, y),
          PoincareBall(c).distance(y, x))

    print(PoincareBall(c).distance(torch.FloatTensor([0.0029, 0.0096]),
                                   torch.FloatTensor([0.0038, 0.0077])))
