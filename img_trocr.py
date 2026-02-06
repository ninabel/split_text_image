# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Functions to do image OCR with TrOCR library
__author__ = "Nina Belyavskaya"


from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    AutoTokenizer,
    AutoFeatureExtractor,
)
from config import hf_token as HUGGINGFACE_API_TOKEN
import torch
import numpy as np
from base import BaseOCR, BaseLineSplitter
import requests
from PIL import Image
import io
from huggingface_hub import InferenceClient

HUGGINGFACE_API_URL = "https://router.huggingface.co/hf-inference/models/"

class TrOCRModel(BaseOCR):
    """ OCR processor using TrOCR model from HuggingFace. """
    def __init__(self, line_splitter: BaseLineSplitter, model_name='microsoft/trocr-base-handwritten', device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            try:
                self.processor = TrOCRProcessor.from_pretrained(model_name)
            except KeyError:
                print(f"Model '{model_name}' not found, falling back to 'microsoft/trocr-base-handwritten'")
                self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            self.model.to(device)
            self.device = device
            super().__init__(line_splitter)
        except Exception as e:
            raise ValueError(f"Failed to load TrOCR model '{model_name}': {e}. Check transformers version and internet.") from e

    def recognize(self, line_image: np.ndarray) -> str:
        """ Recognizes text from a line image using TrOCR. """
        # Preprocess the image
        pixel_values = self.processor(images=line_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        # Generate text ids
        generated_ids = self.model.generate(pixel_values)

        # Decode the generated ids to text
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription.strip()


class TrOCROnlineModel(BaseOCR):
    """ OCR processor using TrOCR model from HuggingFace with online line splitting. """
    def __init__(self, line_splitter, model_name='microsoft/trocr-base-handwritten', device=None):
        self.API_URL = HUGGINGFACE_API_URL + model_name
        print(f"Initialized TrOCROnlineModel with API URL: {self.API_URL}")
        super().__init__(line_splitter)
    
    def recognize(self, line_image: np.ndarray) -> str:
        """ Recognizes text from a line image using TrOCR via HuggingFace API. """

        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(line_image)

        # Prepare the payload for the API request
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()

        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
            "Content-Type": "application/octet-stream",
            }

        retries = 3
        while retries > 0:
            retries -= 1
            response = requests.post(self.API_URL, headers=headers, files={"file": img_bytes})
            if response.status_code == 200:
                result = response.json()
                return result[0]['generated_text'].strip()
        
        raise Exception(f"API request {self.API_URL} failed with status code {response.status_code}: {response.text}")