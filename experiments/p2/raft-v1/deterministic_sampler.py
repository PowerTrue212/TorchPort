"""Pilot RAFT bilinear sampling with deterministic gather backward on CUDA."""
import torch
import torch.nn.functional as F


def bilinear_sampler(img, coords, mode="bilinear", mask=False):
    if mode != "bilinear":
        raise ValueError(mode)
    batch, channels, height, width = img.shape
    x, y = coords[..., 0], coords[..., 1]
    x0, y0 = torch.floor(x), torch.floor(y)
    x1, y1 = x0 + 1, y0 + 1
    flat = img.reshape(batch, channels, height * width)

    def corner(xi, yi):
        valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
        index = (yi.clamp(0, height - 1).long() * width + xi.clamp(0, width - 1).long())
        index = index.reshape(batch, 1, -1).expand(-1, channels, -1)
        value = torch.gather(flat, 2, index).reshape(batch, channels, *x.shape[1:])
        return value * valid[:, None]

    a = corner(x0, y0)
    b = corner(x1, y0)
    c = corner(x0, y1)
    d = corner(x1, y1)
    wx, wy = x - x0, y - y0
    output = (a * (1 - wx)[:, None] * (1 - wy)[:, None]
              + b * wx[:, None] * (1 - wy)[:, None]
              + c * (1 - wx)[:, None] * wy[:, None]
              + d * wx[:, None] * wy[:, None])
    if mask:
        inside = (x > 0) & (x < width - 1) & (y > 0) & (y < height - 1)
        return output, inside[:, None].float()
    return output


class _OfficialForwardDeterministicBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img, coords):
        height, width = img.shape[-2:]
        xgrid = 2 * coords[..., 0:1] / (width - 1) - 1
        ygrid = 2 * coords[..., 1:2] / (height - 1) - 1
        grid = torch.cat([xgrid, ygrid], dim=-1)
        ctx.save_for_backward(img, coords)
        return F.grid_sample(img, grid, align_corners=True)

    @staticmethod
    def backward(ctx, grad_output):
        img, coords = ctx.saved_tensors
        with torch.enable_grad():
            x = img.detach().requires_grad_(True)
            xy = coords.detach().requires_grad_(True)
            output = bilinear_sampler(x, xy)
            grad_img, grad_coords = torch.autograd.grad(output, (x, xy), grad_output)
        return grad_img, grad_coords


def official_forward_deterministic_backward(img, coords, mode="bilinear", mask=False):
    if mode != "bilinear":
        raise ValueError(mode)
    output = _OfficialForwardDeterministicBackward.apply(img, coords)
    if mask:
        height, width = img.shape[-2:]
        x, y = coords[..., 0], coords[..., 1]
        inside = (x > 0) & (x < width - 1) & (y > 0) & (y < height - 1)
        return output, inside[:, None].float()
    return output
