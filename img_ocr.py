# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Functions to do image OCR
__author__ = "Nina Belyavskaya"

# from doctr.io import decode_img_as_tensor
from doctr.models import ocr_predictor
from PIL import Image
import numpy as np
import torch

from img_proc import BaseLineSplitter

# Load the model once (CPU or GPU)
predictor = ocr_predictor(
    det_arch="db_resnet50",      # very good at detecting handwritten lines
    reco_arch="crnn_vgg16_bn",   # we only use detection, recognition is ignored
    pretrained=True,
    detect_orientation=True,     # automatically rotates the page correctly
    straighten_pages=True,       # fixes slight skew
).to("cuda" if torch.cuda.is_available() else "cpu")


def split_image_into_lines(image: np.ndarray) -> list[Image.Image]:
    """
    Returns a list of PIL Images, each containing one text line,
    in correct reading order (left→right, top→bottom).
    """
    # Convert grayscale to RGB if necessary
    if image.ndim == 2:
        image = Image.fromarray(image).convert("RGB")
        image = np.asarray(image)

    # Run detection only (much faster than full OCR)
    result = predictor([image])
    result.show()

    return result.pages


def crop_lines_from_pages(image, pages, padding=3) -> list[Image.Image]:
    """ Crops line images from docTR result pages. """
    line_crops = []
    for page in pages:
        h, w = page.dimensions  # original image height, width
        print(f'Page dimensions: width={w}, height={h}')
        print(f'Image dimensions: shape={image.shape}')

        for block in page.blocks:
            for line in block.lines:
                # line.geometry = ((x_min, y_min), (x_max, y_max)) in relative coords [0,1]
                (x1, y1), (x2, y2) = line.geometry

                # Convert to absolute pixel coordinates
                left = int(x1 * w)
                top = int(y1 * h)
                right = int(x2 * w)
                bottom = int(y2 * h)

                # Add a small padding (helps TrOCR a lot)
                if padding > 0:
                    left = max(0, left - padding)
                    top = max(0, top - padding)
                    right = min(w, right + padding)
                    bottom = min(h, bottom + padding)

                mask = np.zeros((bottom-top, right-left), dtype=np.uint8)
                for word in line.words:
                    (bx1, by1), (bx2, by2) = word.geometry
                    # Convert to absolute pixel coordinates within the line crop
                    bx1 = int((bx1 * w))
                    by1 = int((by1 * h))
                    bx2 = int((bx2 * w))
                    by2 = int((by2 * h))
                    mask[by1 - top:by2 - top, bx1 - left:bx2 - left] = 255

                # Crop the original image
                line_crop = image[top:bottom, left:right]
                line_crop = np.where(mask > 0, line_crop, 255)
                line_crops.append(line_crop)
    return line_crops


class DocTrLineSplitter(BaseLineSplitter):
    """ Line splitter using docTR OCR model. """

    def split_image_into_lines(self, image: np.ndarray) -> list[np.ndarray]:
        pages = split_image_into_lines(image)
        line_crops = crop_lines_from_pages(image, pages, padding=3)
        return line_crops
