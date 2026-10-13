"""Build fixed single-image .npy inputs (numpy + PIL only; no torch/cv2 needed).

cv2 is not installed in any local interpreter, so BiRefNet's cv2.resize
(INTER_LINEAR) is re-implemented in numpy (half-pixel centers, no antialias).
Images are read straight from the zips (no unzip).
Run: E:\\Anaconda\\python.exe make_inputs.py
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
COD10K_ZIP = ROOT / "migrations" / "CamoFormer" / "data" / "TestDataset.zip"
COD10K_MEMBER = "TestDataset/COD10K/Image/COD10K-CAM-1-Aquatic-1-BatFish-2.jpg"
PLANT_ZIP = ROOT / "heldout" / "plantcamo" / "data" / "PlantCAMO1250.zip"
PLANT_MEMBER = "PlantCAMO1250/test/rgb/PC_Abronia_latifolia_1042.jpg"
IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
VSCODE_COD = ([0.447, 0.443, 0.364], [0.192, 0.185, 0.180])  # dataset_test.py transformCOD_RGB


def read(zip_path: Path, member: str) -> Image.Image:
    with zipfile.ZipFile(zip_path) as z:
        return Image.open(io.BytesIO(z.read(member))).convert("RGB")


def to_tensor(img: Image.Image, mean, std) -> np.ndarray:
    """torchvision ToTensor + Normalize in float32."""
    x = np.asarray(img, dtype=np.uint8).astype(np.float32) / np.float32(255.0)
    x = (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None]).astype(np.float32)


def cv2_linear(img: np.ndarray, h: int, w: int) -> np.ndarray:
    src = img.astype(np.float64)
    H, W = src.shape[:2]

    def idx(n_out, n_in):
        s = np.maximum((np.arange(n_out) + 0.5) * n_in / n_out - 0.5, 0.0)
        i0 = np.minimum(np.floor(s).astype(int), n_in - 1)
        i1 = np.minimum(i0 + 1, n_in - 1)
        return i0, i1, s - i0

    y0, y1, wy = idx(h, H)
    x0, x1, wx = idx(w, W)
    rows = src[y0] * (1 - wy)[:, None, None] + src[y1] * wy[:, None, None]
    out = rows[:, x0] * (1 - wx)[None, :, None] + rows[:, x1] * wx[None, :, None]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def main() -> None:
    cod = read(COD10K_ZIP, COD10K_MEMBER)
    # BiRefNet smoke/run_torch.py: cv2.resize(1024, LINEAR) -> RGB -> ToTensor -> ImageNet norm
    b = cv2_linear(np.asarray(cod), 1024, 1024)
    np.save(OUT / "birefnet-cod10k-batfish2-1024.npy", to_tensor(Image.fromarray(b), *IMAGENET))
    # VSCode-v2 dataset_test.py: trans.Scale((352,352)) (PIL bilinear) -> ToTensor -> COD mean/std
    v = cod.resize((352, 352), Image.BILINEAR)
    np.save(OUT / "vscode-v2-cod10k-batfish2-352.npy", to_tensor(v, *VSCODE_COD))
    # PlantCamo utils/dataloader.py My_test_dataset: exif_transpose -> Resize((704,704)) -> ImageNet norm
    p = ImageOps.exif_transpose(read(PLANT_ZIP, PLANT_MEMBER)).resize((704, 704), Image.BILINEAR)
    np.save(OUT / "plantcamo-abronia1042-704.npy", to_tensor(p, *IMAGENET))
    for f in sorted(OUT.glob("*.npy")):
        a = np.load(f)
        print(f.name, a.shape, a.dtype, float(a.min()), float(a.max()))


if __name__ == "__main__":
    main()
