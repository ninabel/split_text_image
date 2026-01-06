# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# OCR handwritten text from images with Tesseract
__author__ = "Nina Belyavskaya"

from base import BaseOCR
import pytesseract
import numpy as np


class TesseractOCR(BaseOCR):
    """ OCR processor using Tesseract engine. """
    def __init__(self, lang='eng'):
        self.lang = lang

    def recognize(self, line_image: np.ndarray) -> str:
        """
        Recognizes text from an image using Tesseract.
        """
        text = pytesseract.image_to_string(line_image, lang=self.lang)
        return text.strip()
