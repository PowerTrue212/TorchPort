"""Capture controller-only CLIP target forward stages on the V100."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jittor as jt
import numpy as np
from evaluation_entry import load_train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    jt.flags.use_cuda = 1
    model = load_train_model()
    with np.load(args.weights, allow_pickle=False) as weights:
        model.model.load_parameters({name: weights[name] for name in weights.files})
        loaded = len(weights.files)
    model.train()
    for parameter in model.parameters():
        parameter.start_grad()
    clip = model.model
    images = jt.array(np.array(np.load(args.images, mmap_mode="r")[0]))
    tokens = jt.array(np.array(np.load(args.tokens, mmap_mode="r")[0]))
    arrays: dict[str, np.ndarray] = {}

    def save(name: str, value) -> None:
        arrays[name] = value.numpy()

    visual = clip.visual
    image = visual.conv1(images)
    save("image.conv1", image)
    image = image.reshape(image.shape[0], image.shape[1], -1).transpose(1, 2)
    cls = visual.class_embedding.reshape(1, 1, -1).broadcast(
        [image.shape[0], 1, image.shape[-1]]
    )
    image = jt.concat([cls, image], dim=1)
    image = image + visual.positional_embedding.reshape(
        1, image.shape[1], image.shape[-1]
    )
    save("image.position", image)
    image_mean = jt.mean(image, dims=[-1], keepdims=1)
    save("image.ln_pre.mean", image_mean)
    image_mean_mm = jt.matmul(
        image.reshape(-1, image.shape[-1]),
        jt.ones((image.shape[-1], 1), dtype="float32") / image.shape[-1],
    ).reshape(image.shape[0], image.shape[1], 1)
    save("image.ln_pre.mean_matmul", image_mean_mm)
    image_centered_mm = image - image_mean_mm
    image_var_mm = jt.matmul(
        (image_centered_mm * image_centered_mm).reshape(-1, image.shape[-1]),
        jt.ones((image.shape[-1], 1), dtype="float32") / image.shape[-1],
    ).reshape(image.shape[0], image.shape[1], 1)
    save("image.ln_pre.var_matmul", image_var_mm)
    image_var = jt.mean(
        (image - image_mean) * (image - image_mean), dims=[-1], keepdims=1
    )
    save("image.ln_pre.var", image_var)
    image = visual.ln_pre(image)
    save("image.ln_pre", image)
    image = image.transpose(0, 1)
    for index, block in enumerate(visual.transformer.resblocks):
        image = block(image)
        save(f"image.block.{index}", image)
    image = image.transpose(0, 1)
    image = visual.ln_post(image[:, 0, :])
    save("image.ln_post", image)
    image = jt.matmul(image, visual.proj)
    save("image.projected", image)

    text = clip.token_embedding(tokens).float32()
    save("text.embedding", text)
    text = text + clip.positional_embedding
    save("text.position", text)
    text_mean_mm = jt.matmul(
        text.reshape(-1, text.shape[-1]),
        jt.ones((text.shape[-1], 1), dtype="float32") / text.shape[-1],
    ).reshape(text.shape[0], text.shape[1], 1)
    save("text.ln_1_0.mean_matmul", text_mean_mm)
    text = text.transpose(0, 1)
    save("text.ln_1_0", clip.transformer.resblocks[0].ln_1(text))
    for index, block in enumerate(clip.transformer.resblocks):
        text = block(text)
        save(f"text.block.{index}", text)
    text = text.transpose(0, 1)
    text = clip.ln_final(text)
    save("text.ln_final", text)
    eot = jt.argmax(tokens, dim=-1)[0]
    index = eot.reshape((-1, 1, 1)).broadcast([text.shape[0], 1, text.shape[-1]])
    text = jt.gather(text, 1, index).reshape((text.shape[0], text.shape[-1]))
    text = jt.matmul(text, clip.text_projection)
    save("text.projected", text)

    np.savez(args.output, **arrays)
    print(json.dumps({"loaded": loaded, "stages": len(arrays)}))


if __name__ == "__main__":
    main()
