# ! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Compare different OCR models for handwritten text
__author__ = "Nina Belyavskaya"

import jiwer
from ocr_handwritten.img_doctr import DocTrOCR, DocTrLineSplitter
from ocr_handwritten.img_trocr import TrOCRModel
from ocr_handwritten.img_tesseract import TesseractOCR
from ocr_handwritten.img_proc import ContourBasedLineSplitter
# from huggingface_hub import snapshot_download, list_models
from pathlib import Path
import logging

# local config to set path to images and ground truth
import config


def compute_cer_wer(true_texts, pred_texts):
    """ Computes CER and WER between true and predicted texts. """
    total_cer = jiwer.cer(true_texts, pred_texts)
    total_wer = jiwer.wer(true_texts, pred_texts)
    return total_cer, total_wer


def compare_ocr_models(image, true_text):
    """ Compares different OCR models on the given image and true text. """
    # Initialize line splitters
    line_splitters = [None, DocTrLineSplitter(), ContourBasedLineSplitter()]

    # Prepare TrOCR models
    # pipeline_tag=image-to-text&sort=downloads&search=handwritten or sort=trending?
    trocr_models = ['microsoft/trocr-base-handwritten',
                    'microsoft/trocr-large-handwritten',
                    'DunnBC22/trocr-base-handwritten-OCR-handwriting_recognition_v2',
                    'deepseek-ai/DeepSeek-OCR',
                    'Qwen/Qwen2.5-VL-72B-Instruct',
                    'mistralai/Mistral-OCR',
                    'PaddlePaddle/PaddleOCR',
                    'allenai/olmOCR',
                    'Gustavosta/Trocr-Handwritten-OCR',
                    'Xenova/trocr-base-handwritten',
                    ]

    # Initialize OCR models
    OCR_tools = []
    for splitter in line_splitters:
        splitter_name = type(splitter).__name__ if splitter is not None else "NoSplitter"

        doctr_ocr = DocTrOCR(line_splitter=splitter)
        OCR_tools.append((f"docTR {splitter_name}", doctr_ocr))

        # Add TrOCR models
        for model_name in trocr_models:
            trocr_ocr = TrOCRModel(line_splitter=splitter, model_name=model_name)
            OCR_tools.append((f"TrOCR-{model_name} {splitter_name}", trocr_ocr))

        tesseract_ocr = TesseractOCR()
        OCR_tools.append((f"Tesseract {splitter_name}", tesseract_ocr))
    results = {}
    # Recognize text with each model
    for model_name, ocr_tool in OCR_tools:
        pred_text = ocr_tool.recognize_text(image)
        results[model_name] = pred_text

    # Compute CER and WER for each model
    for model_name, pred_text in results.items():
        cer, wer = compute_cer_wer(true_text, pred_text)
        results[model_name] = {"CER": cer, "WER": wer, "Predicted Text": pred_text}

    return results


if __name__ == "__main__":
    image_path = config.image_path
    true_text_path = config.ground_truth_text_path
    file_template = config.image_file_template
    logging.basicConfig(level=logging.INFO)

    # read image files with names corresponding to template *_clean.*
    for image_file in Path(image_path).glob(file_template):
        logging.info(f"Processing image: {image_file}")
        # read ground truth text file
        true_text_file = Path(true_text_path) / f"{image_file.stem}.txt"
        with open(true_text_file, 'r') as f:
            true_text = f.read()
        results = compare_ocr_models(image_file, true_text)
        for model_name, metrics in results.items():
            logging.info(f"Model: {model_name}, CER: {metrics['CER']}, WER: {metrics['WER']}")
