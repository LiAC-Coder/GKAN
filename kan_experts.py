import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

def bspline_basis(x, grid, order):

    x = x.unsqueeze(-1)
    basis = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
    for k in range(1, order + 1):
        left = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)])
        right = (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k])
        basis = left * basis[:, :, :-1] + right * basis[:, :, 1:]
    return basis


class KANBSpline(nn.Module):


    def __init__(self, dim, resolution=5, spline_order=3):
        super().__init__()
        self.order = spline_order
        knots = torch.linspace(-1, 1, resolution + 1)
        step = (knots[-1] - knots[0]) / resolution
        for _ in range(spline_order):
            knots = torch.cat((knots[:1] - step, knots, knots[-1:] + step))
        self.register_buffer("grid", knots.repeat(dim, 1))


        noise = (torch.rand(resolution + 1, dim, dim) - 0.5) * 0.5 / resolution
        points = self.grid[:, spline_order:-spline_order].T if spline_order else self.grid.T
        design = bspline_basis(points, self.grid, spline_order).permute(1, 0, 2)
        design = design[:, None].expand(dim, dim, resolution + 1, resolution + spline_order)
        target = noise.permute(1, 2, 0).unsqueeze(-1)
        coef = torch.linalg.lstsq(design, target).solution[..., 0]
        self.coefficients = nn.Parameter(coef.permute(1, 0, 2).contiguous())
        self.base_weight = nn.Parameter(((torch.rand(dim, dim) * 2 - 1) / math.sqrt(dim)).T.contiguous())
        self.spline_scale = nn.Parameter(torch.ones(dim, dim) / math.sqrt(dim))

    def forward(self, x):
        bases = bspline_basis(x, self.grid, self.order)
        return torch.einsum("nit,oit->no", bases, self.coefficients * self.spline_scale.unsqueeze(-1)) + F.linear(F.silu(x), self.base_weight)


class KANFourier(nn.Module):
    def __init__(self, dim, resolution=5):
        super().__init__()
        self.register_buffer("harmonics", torch.arange(1, resolution + 1).float())
        self.coefficients = nn.Parameter(torch.randn(2, dim, dim, resolution)
                                         / math.sqrt(dim * resolution))

    def forward(self, x):
        angle = x.unsqueeze(-1) * self.harmonics
        return (torch.einsum("nit,oit->no", angle.cos(), self.coefficients[0])
                + torch.einsum("nit,oit->no", angle.sin(), self.coefficients[1]))


class KANRadial(nn.Module):
    def __init__(self, dim, resolution=5):
        super().__init__()
        self.centers = nn.Parameter(torch.linspace(-1, 1, resolution).repeat(dim, 1))
        self.log_width = nn.Parameter(torch.zeros(dim, resolution))
        self.coefficients = nn.Parameter(torch.randn(dim, dim, resolution)
                                         / math.sqrt(dim * resolution))

    def forward(self, x):
        width = F.softplus(self.log_width) + 1e-6
        bases = torch.exp(-0.5 * ((x.unsqueeze(-1) - self.centers) / width).square())
        return torch.einsum("nit,oit->no", bases, self.coefficients)


class KANBernstein(nn.Module):
    def __init__(self, dim, resolution=5):
        super().__init__()
        self.order = resolution
        self.register_buffer("powers", torch.arange(resolution + 1).float())
        self.register_buffer("binomial", torch.tensor(
            [math.comb(resolution, i) for i in range(resolution + 1)]).float())
        self.coefficients = nn.Parameter(torch.randn(dim, dim, resolution + 1)
                                         / math.sqrt(dim * (resolution + 1)))

    def forward(self, x):
        value = x.sigmoid().unsqueeze(-1)
        bases = self.binomial * value.pow(self.powers) * (1 - value).pow(self.order - self.powers)
        return torch.einsum("nit,oit->no", bases, self.coefficients)


class KANExpertNetwork(nn.Module):


    def __init__(self, dim, resolution=5, spline_order=3, activation_checkpoint=True):
        super().__init__()
        self.experts = nn.ModuleList([
            KANBSpline(dim, resolution, spline_order), KANFourier(dim, resolution),
            KANRadial(dim, resolution), KANBernstein(dim, resolution),
        ])
        self.attention = nn.Linear(dim, 4, bias=False)
        self.activation_checkpoint = activation_checkpoint

    def forward(self, x):
        weights = self.attention(x).softmax(dim=-1)
        output = torch.zeros_like(x)
        for index, expert in enumerate(self.experts):
            if self.training and self.activation_checkpoint and torch.is_grad_enabled():
                value = checkpoint(expert, x, use_reentrant=False)
            else:
                value = expert(x)
            output = output + weights[:, index:index + 1] * value
        return output
