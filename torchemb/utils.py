import torch


EPS = 1.0e-15


def asinh(x):
    return torch.log(x + (x ** 2 + 1) ** 0.5)


def acosh(x):
    return torch.log(x + torch.clamp(x ** 2 - 1, EPS) ** 0.5)


def atanh(x):
    return 0.5 * torch.log((1 + x) / torch.clamp(1 - x, EPS))
