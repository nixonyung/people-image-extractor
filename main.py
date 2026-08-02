from pathlib import Path
from typing import cast

import torch
import torchvision.transforms.v2.functional as TF
from myutils.libreyolo import YOLORunner, expand_bounding_box
from myutils.pathlib import rglob_with_suffices
from torchcodec.decoders import decode_image, decode_jpeg  # pyright: ignore[reportPrivateImportUsage]
from torchcodec.encoders import JpegEncoder  # pyright: ignore[reportPrivateImportUsage]
from tqdm import tqdm

INPUTS_DIR = Path("inputs")
CROPPED_OUTPUTS_DIR = Path("outputs/cropped")
ENHANCED_OUTPUTS_DIR = Path("outputs/enhanced")

INPUT_MIN_WIDTH = 150
INPUT_MIN_HEIGHT = 150
EXPANSION_FACTOR_X = 0.1
EXPANSION_FACTOR_Y = 0.03
OUTPUT_MIN_WIDTH = 1080
OUTPUT_MIN_HEIGHT = 1920

BATCH_SIZE = 32


def save_jpg(path: Path, img: torch.Tensor, quality: int = 85):
    path.parent.mkdir(parents=True, exist_ok=True)
    JpegEncoder(img).to_file(path, quality=quality)


def crop_images(paths: list[Path]):
    person_cls = YOLO_RUNNER.get_cls("person")

    for path, idx, xyxy, cls in YOLO_RUNNER.extract(
        tqdm(paths, "extracting bounding boxes"),
        batch=BATCH_SIZE,
    ):
        if cls != person_cls:
            continue
        if xyxy[2] - xyxy[0] < INPUT_MIN_WIDTH or xyxy[3] - xyxy[1] < INPUT_MIN_HEIGHT:
            continue

        xyxy = expand_bounding_box(xyxy, (EXPANSION_FACTOR_X, EXPANSION_FACTOR_Y))

        img = decode_image(path).to(device=DEVICE)
        img = img[:, xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]

        save_jpg(
            CROPPED_OUTPUTS_DIR / path.relative_to(INPUTS_DIR).parent / f"{path.stem}_{idx}.jpg",
            img,
        )


def enhance_images(paths: list[Path]):
    for path in tqdm(paths, "enhancing images"):
        img = cast(torch.Tensor, decode_jpeg(path, device=DEVICE))

        _, h, w = img.shape
        scale = max(OUTPUT_MIN_WIDTH / w, OUTPUT_MIN_HEIGHT / h)
        img = TF.resize(img, [round(h * scale), round(w * scale)])

        img = TF.adjust_sharpness(img, 1.25)

        save_jpg(
            ENHANCED_OUTPUTS_DIR
            / path.relative_to(CROPPED_OUTPUTS_DIR).parent
            / f"{path.stem}.jpg",
            img,
        )


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    YOLO_RUNNER = YOLORunner()

    crop_images(list(rglob_with_suffices(INPUTS_DIR, [".jpg", "jpeg", ".png"])))
    enhance_images(list(rglob_with_suffices(CROPPED_OUTPUTS_DIR, [".jpg"])))
