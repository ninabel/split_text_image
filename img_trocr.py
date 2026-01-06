# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Functions to do image OCR with TrOCR library
__author__ = "Nina Belyavskaya"


from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import numpy as np
from base import BaseOCRwithSplitter, BaseLineSplitter


class TrOCRModel(BaseOCRwithSplitter):
    """ OCR processor using TrOCR model from HuggingFace. """
    def __init__(self, line_splitter: BaseLineSplitter, model_name='microsoft/trocr-base-handwritten', device=None):
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.device = device
        super().__init__(line_splitter)

    def recognize_line(self, line_image: np.ndarray) -> str:
        """ Recognizes text from a line image using TrOCR. """
        # Preprocess the image
        pixel_values = self.processor(images=line_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        # Generate text ids
        generated_ids = self.model.generate(pixel_values)

        # Decode the generated ids to text
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription.strip()
