# machine-learing

一个包含机器学习示例的仓库。这里新增了一个 **通用的多尺度傅里叶特征映射 PINN（PyTorch）** 实现。

## 文件说明

- `pinn/multiscale_fourier_pinn.py`
  - `MultiScaleFourierFeatures`：多尺度傅里叶编码层；
  - `MultiScaleFourierPINN`：编码层 + MLP 主干；
  - `grad / second_grad`：自动微分工具；
  - `pinn_loss`：PDE / 边界 / 监督混合损失；
  - `burgers_1d_residual`：1D Burgers 方程残差示例。

## 快速使用

```python
import torch
from pinn import MultiScaleFourierPINN, PINNBatch, pinn_loss, burgers_1d_residual

model = MultiScaleFourierPINN(
    in_dim=2,           # 例如 (x, t)
    out_dim=1,          # 例如 u(x,t)
    map_size=64,
    scales=(1.0, 2.0, 4.0, 8.0),
    hidden_dims=(128, 128, 128, 128),
)

xt = torch.rand(1024, 2)
nu = 0.01 / torch.pi

batch = PINNBatch(interior=xt)
loss = pinn_loss(
    model,
    batch,
    pde_residual_fn=lambda m, x: burgers_1d_residual(m, x, nu=nu),
)
loss.backward()
```

## 自检

```bash
python pinn/multiscale_fourier_pinn.py
```

如果运行成功，会打印类似：

```text
sanity check loss = 0.xxxxxx
```
