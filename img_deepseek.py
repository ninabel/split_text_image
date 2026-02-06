# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Functions to do image OCR with DeepSeek library
__author__ = "Nina Belyavskaya"

from deepseek_ocr import DeepSeekOCR
import os
from base import BaseOCR

class DeepSeekOCRModel(BaseOCR):
    """ OCR processor using DeepSeek library. """
    def __init__(self):
        self.ocr = DeepSeekOCR()
       

    def recognize_file(self, image_path: str) -> str:
        """ Recognizes text from a line image using DeepSeek. """
        # DeepSeek expects a file path, so we need to save the line image temporarily
        return self.ocr.recognize(image_path)
