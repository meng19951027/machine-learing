"""通用多尺度傅里叶特征映射 PINN（PyTorch）实现。

这个模块提供：
1) MultiScaleFourierFeatures: 多尺度傅里叶特征编码层；
2) MLP: 可配置全连接主干网络；
3) MultiScaleFourierPINN: PINN 网络封装；
4) autograd 工具函数：用于计算一阶/二阶偏导，方便构建 PDE 残差。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

import torch
from torch import Tensor, nn


class MultiScaleFourierFeatures(nn.Module):
    """多尺度傅里叶特征映射。

    给定输入 x ∈ R^{N x in_dim}，输出为：
        [x, sin(2π x B_1), cos(2π x B_1), ..., sin(2π x B_k), cos(2π x B_k)]

    其中 B_i 是每个尺度的随机投影矩阵，并按 scale_i 进行缩放。
    """

    def __init__(
        self,
        in_dim: int,
        map_size: int = 64,
        scales: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
        include_input: bool = True,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError("in_dim must be > 0")
        if map_size <= 0:
            raise ValueError("map_size must be > 0")
        if len(scales) == 0:
            raise ValueError("scales must not be empty")

        self.in_dim = in_dim
        self.map_size = map_size
        self.include_input = include_input
        self.register_buffer("_two_pi", torch.tensor(2.0 * torch.pi), persistent=False)

        proj_mats: List[nn.Parameter] = []
        for s in scales:
            if s <= 0:
                raise ValueError(f"all scales must be > 0, got {s}")
            mat = torch.randn(in_dim, map_size) * float(s)
            p = nn.Parameter(mat, requires_grad=trainable)
            proj_mats.append(p)

        self.proj_mats = nn.ParameterList(proj_mats)

    @property
    def out_dim(self) -> int:
        base = self.in_dim if self.include_input else 0
        return base + 2 * self.map_size * len(self.proj_mats)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or x.size(-1) != self.in_dim:
            raise ValueError(f"x shape should be [N, {self.in_dim}], got {tuple(x.shape)}")

        feats: List[Tensor] = [x] if self.include_input else []
        for mat in self.proj_mats:
            phase = self._two_pi * (x @ mat)
            feats.extend((torch.sin(phase), torch.cos(phase)))
        return torch.cat(feats, dim=-1)


class MLP(nn.Module):
    """标准 MLP 主干，可选激活函数与层归一化。"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Sequence[int] = (128, 128, 128, 128),
        activation: Optional[Callable[[], nn.Module]] = nn.Tanh,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must not be empty")

        dims = [in_dim, *hidden_dims, out_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == len(dims) - 2
            if not is_last:
                if layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                if activation is not None:
                    layers.append(activation())

        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MultiScaleFourierPINN(nn.Module):
    """通用的多尺度傅里叶特征 PINN。"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        map_size: int = 64,
        scales: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
        hidden_dims: Sequence[int] = (128, 128, 128, 128),
        activation: Optional[Callable[[], nn.Module]] = nn.Tanh,
        include_input: bool = True,
        trainable_fourier: bool = False,
    ) -> None:
        super().__init__()
        self.ffm = MultiScaleFourierFeatures(
            in_dim=in_dim,
            map_size=map_size,
            scales=scales,
            include_input=include_input,
            trainable=trainable_fourier,
        )
        self.backbone = MLP(
            in_dim=self.ffm.out_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(self.ffm(x))


@dataclass
class PINNBatch:
    """PINN 训练一个 batch 的常用张量容器。"""

    interior: Tensor
    boundary_x: Optional[Tensor] = None
    boundary_y: Optional[Tensor] = None
    supervised_x: Optional[Tensor] = None
    supervised_y: Optional[Tensor] = None


def grad(outputs: Tensor, inputs: Tensor, create_graph: bool = True) -> Tensor:
    """计算 dy/dx，其中 outputs 为标量场（N x 1）或可求和张量。"""

    g = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=create_graph,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return g


def second_grad(outputs: Tensor, inputs: Tensor, dim: int) -> Tensor:
    """计算二阶偏导 d²y / d x_dim²。"""

    first = grad(outputs, inputs, create_graph=True)
    second = grad(first[:, dim : dim + 1], inputs, create_graph=True)[:, dim : dim + 1]
    return second


def burgers_1d_residual(model: nn.Module, xt: Tensor, nu: float) -> Tensor:
    """示例：1D Burgers 方程残差。

    PDE: u_t + u u_x - nu u_xx = 0
    xt: [x, t]
    """

    xt = xt.requires_grad_(True)
    u = model(xt)  # [N, 1]

    du = grad(u, xt)
    u_x = du[:, 0:1]
    u_t = du[:, 1:2]
    u_xx = second_grad(u, xt, dim=0)
    return u_t + u * u_x - nu * u_xx


def pinn_loss(
    model: nn.Module,
    batch: PINNBatch,
    pde_residual_fn: Callable[[nn.Module, Tensor], Tensor],
    w_pde: float = 1.0,
    w_bc: float = 1.0,
    w_sup: float = 1.0,
) -> Tensor:
    """通用 PINN 损失组合：PDE + 边界条件 + 监督数据。"""

    loss = torch.tensor(0.0, device=batch.interior.device)

    r = pde_residual_fn(model, batch.interior)
    loss = loss + w_pde * torch.mean(r.square())

    if batch.boundary_x is not None and batch.boundary_y is not None:
        pred_bc = model(batch.boundary_x)
        loss = loss + w_bc * torch.mean((pred_bc - batch.boundary_y).square())

    if batch.supervised_x is not None and batch.supervised_y is not None:
        pred_sup = model(batch.supervised_x)
        loss = loss + w_sup * torch.mean((pred_sup - batch.supervised_y).square())

    return loss


if __name__ == "__main__":
    # 快速自检
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiScaleFourierPINN(in_dim=2, out_dim=1).to(device)

    xt = torch.rand(256, 2, device=device)
    nu = 0.01 / torch.pi

    def residual_fn(m: nn.Module, x: Tensor) -> Tensor:
        return burgers_1d_residual(m, x, nu=nu)

    batch = PINNBatch(interior=xt)
    loss = pinn_loss(model, batch, residual_fn)
    loss.backward()
    print(f"sanity check loss = {loss.item():.6f}")
