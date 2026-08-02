# pyright: reportMissingTypeStubs=false

import os
from itertools import batched
from pathlib import Path
from typing import cast

from libreyolo.models import LibreYOLO
from libreyolo.models.base.model import BaseModel, Results
from PIL import Image, ImageFilter
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

INPUT_DIR = Path("inputs")
CROPPED_OUTPUT_DIR = Path("outputs/cropped")
ENHANCED_OUTPUT_DIR = Path("outputs/enhanced")

MODEL = cast(BaseModel, LibreYOLO("models/LibreYOLO9c.pt"))
PERSON_CLASS_ID = next((id, name) for id, name in MODEL.names.items() if name == "person")[0]
BATCH_SIZE = 32

EXPANSION_FACTOR_X = 0.1
EXPANSION_FACTOR_Y = 0.03
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920


def crop_image(img: Image.Image, x1: float, y1: float, x2: float, y2: float):
    w = x2 - x1
    h = y2 - y1
    dx = w * EXPANSION_FACTOR_X / 2
    dy = h * EXPANSION_FACTOR_Y / 2

    return img.crop(
        (
            max(0, x1 - dx),
            max(0, y1 - dy),
            min(img.width, x2 + dx),
            min(img.height, y2 + dy),
        )
    )


def upscale_image(img: Image.Image):
    w, h = img.size
    scale = max(OUTPUT_WIDTH / w, OUTPUT_HEIGHT / h)

    return img.resize(
        (int(w * scale), int(h * scale)),
        resample=Image.Resampling.LANCZOS,
    )


def sharpen_image(img: Image.Image):
    return img.filter(
        ImageFilter.UnsharpMask(
            radius=1.5,
            percent=120,
            threshold=4,
        )
    )


def enhance_image(path: Path):
    img = Image.open(path)

    img = upscale_image(img)
    img = sharpen_image(img)

    return path, img


if __name__ == "__main__":
    for batch in batched(
        tqdm(
            [
                path
                for path in INPUT_DIR.rglob("*")
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ],
            desc="extracting people",
        ),
        BATCH_SIZE,
    ):
        results = MODEL(batch)
        for input_path, result in zip(
            batch,
            cast(list[Results], MODEL(batch)),
        ):
            if result.boxes is None:
                continue

            img = Image.open(input_path)
            for i, (x1, y1, x2, y2, confidence, class_id) in enumerate(result.boxes.data):
                if int(class_id) != PERSON_CLASS_ID:
                    continue
                if x2 - x1 < 200 or y2 - y1 < 200:
                    continue

                cropped_img = crop_image(img, float(x1), float(y1), float(x2), float(y2))
                output_path = (
                    CROPPED_OUTPUT_DIR
                    / input_path.relative_to(INPUT_DIR).parent
                    / f"{input_path.stem}_{i}.jpg"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cropped_img.save(output_path)

    for path, img in process_map(
        enhance_image,
        [
            path
            for path in CROPPED_OUTPUT_DIR.rglob("*")
            if path.is_file() and path.suffix == ".jpg"
        ],
        max_workers=cpu_count // 2 if (cpu_count := os.cpu_count()) is not None else 1,
        chunksize=1,
        desc="enhancing images",
    ):
        output_path = (
            ENHANCED_OUTPUT_DIR / path.relative_to(CROPPED_OUTPUT_DIR).parent / f"{path.stem}.jpg"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
