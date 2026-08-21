from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


RUNS = {
    "unet": "0357-unet-align-1787206427",
    "resnet": "0363-resnet-align-1787207350",
    "senet": "0369-senet-align-1787208241",
    "mae": "0387-mae-align-0372",
    "superres": "0390-superres-align-0373",
    "swin": "0395-swin-align-0376",
    "vit": "0399-vitp-align-0378",
    "vae": "0406-vae-align-0383",
}


def pair(root: Path, case: str) -> tuple[np.lib.npyio.NpzFile, np.lib.npyio.NpzFile]:
    folder = root / RUNS[case] / "align"
    return np.load(folder / "source.npz"), np.load(folder / "target.npz")


def fmap(value: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(value[0]), axis=0).astype(np.float32)


def tokenmap(value: np.ndarray) -> np.ndarray:
    tokens = value[0]
    size = round(tokens.shape[0] ** 0.5)
    return np.mean(np.abs(tokens), axis=-1).reshape(size, size).astype(np.float32)


def classmap(value: np.ndarray, size: int) -> np.ndarray:
    return value[0].mean(axis=0)[0, 1:].reshape(size, size).astype(np.float32)


def winmap(value: np.ndarray) -> np.ndarray:
    scores = value.mean(axis=(0, 1, 2))
    size = round(scores.shape[0] ** 0.5)
    return scores.reshape(size, size).astype(np.float32)


def unpatch(value: np.ndarray) -> np.ndarray:
    tokens = value[0, 1:]
    grid = round(tokens.shape[0] ** 0.5)
    patch = round((tokens.shape[1] / 3) ** 0.5)
    image = tokens.reshape(grid, grid, patch, patch, 3)
    return image.transpose(0, 2, 1, 3, 4).reshape(grid * patch, grid * patch, 3).astype(np.float32)


def detail(value: np.ndarray) -> np.ndarray:
    dx = np.diff(value, axis=1, append=value[:, -1:])
    dy = np.diff(value, axis=0, append=value[-1:, :])
    return np.sqrt(dx * dx + dy * dy).astype(np.float32)


def collect(root: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}

    src, tgt = pair(root, "vit")
    for tag, data in (("src", src), ("tgt", tgt)):
        out[f"vit_attn_{tag}"] = classmap(data["transformer.layers.0.0.attend"], 4)
        out[f"vit_feat_{tag}"] = tokenmap(data["to_patch_embedding"])

    src, tgt = pair(root, "mae")
    for tag, data in (("src", src), ("tgt", tgt)):
        out[f"mae_rec_{tag}"] = unpatch(data["decoder_pred"])
        out[f"mae_attn_{tag}"] = classmap(data["decoder_blocks.7.attn.attn_drop"], 14)

    src, tgt = pair(root, "swin")
    for tag, data in (("src", src), ("tgt", tgt)):
        out[f"swin_attn_{tag}"] = winmap(data["layers.0.blocks.0.attn.attn_drop"])
        out[f"swin_feat_{tag}"] = tokenmap(data["layers.1.blocks.0"])

    src, tgt = pair(root, "resnet")
    for tag, data in (("src", src), ("tgt", tgt)):
        out[f"resnet_early_{tag}"] = fmap(data["layer1.0"])
        out[f"resnet_late_{tag}"] = fmap(data["layer3.2"])

    src, tgt = pair(root, "senet")
    for tag, data in (("src", src), ("tgt", tgt)):
        out[f"senet_gate_{tag}"] = data["layer3.0.se.fc.3"][0].astype(np.float32)
        out[f"senet_feat_{tag}"] = fmap(data["layer3.0"])

    src, tgt = pair(root, "unet")
    for tag, data in (("src", src), ("tgt", tgt)):
        out[f"unet_enc_{tag}"] = fmap(data["down3"])
        out[f"unet_dec_{tag}"] = fmap(data["up1"])
        out[f"unet_mask_{tag}"] = data["output"][0, 1].astype(np.float32)

    src, tgt = pair(root, "superres")
    for tag, data in (("src", src), ("tgt", tgt)):
        image = data["output"][0, 0].astype(np.float32)
        out[f"superres_out_{tag}"] = image
        out[f"superres_edge_{tag}"] = detail(image)

    src, tgt = pair(root, "vae")
    for tag, data in (("src", src), ("tgt", tgt)):
        out[f"vae_rec_{tag}"] = data["output.0"][0].reshape(28, 28).astype(np.float32)
        mean = data["output.1"][0]
        std = np.exp(0.5 * data["output.2"][0])
        out[f"vae_latent_{tag}"] = np.stack((mean, std)).astype(np.float32)

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **collect(args.runs))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
