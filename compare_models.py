# ! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Compare different OCR models for handwritten text
__author__ = "Nina Belyavskaya"

from importlib.resources import path
import jiwer
from img_doctr import DocTrOCR, DocTrLineSplitter
from img_trocr import TrOCRModel
from img_tesseract import TesseractOCR
from img_proc import ContourBasedLineSplitter, read_image
from img_deepseek import DeepSeekOCRModel
# from huggingface_hub import snapshot_download, list_models
from pathlib import Path
import logging
import numpy as np

# local config to set path to images and ground truth
import config


def compute_cer_wer(true_texts, pred_texts):
    """ Computes CER and WER between true and predicted texts. """
    total_cer = jiwer.cer(true_texts, pred_texts)
    total_wer = jiwer.wer(true_texts, pred_texts)
    return total_cer, total_wer


if __name__ == "__main__":
    image_path = config.image_path
    true_text_path = config.ground_truth_text_path
    file_template = config.image_file_template
    # log to output stream
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logging.info("Starting OCR model comparison...")

    # Initialize line splitters
    line_splitters = [None, DocTrLineSplitter(), ContourBasedLineSplitter()]

    # Prepare TrOCR models
    # pipeline_tag=image-to-text&sort=downloads&search=handwritten or sort=trending?
    trocr_models = ['microsoft/trocr-base-handwritten',
                    'microsoft/trocr-large-handwritten',
                    'DunnBC22/trocr-base-handwritten-OCR-handwriting_recognition_v2',
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
    OCR_tools.append(("DeepSeek", DeepSeekOCRModel()))

    summary_results = {}
    # read image files with names corresponding to template *_clean.*
    for image_file in Path(image_path).glob(file_template):
        logging.info(f"Processing image: {image_file}")
        # read ground truth text file
        true_text_file = Path(true_text_path) / f"{image_file.stem}.txt"
        logging.info(f"Reading ground truth from: {true_text_file}")    
        with open(true_text_file, 'r', encoding="utf-8") as f:
            true_text = f.read()
        results = {}
        for model_name, model in OCR_tools:
            logging.info(f"Evaluating model: {model_name}")
            if model_name == "DeepSeek":
                pred_text = model.recognize_file(str(image_file))
            else:
                image = read_image(image_file)
                pred_text = model.recognize_text(image)
            results[model_name] = pred_text

            # Compute CER and WER for the model
            cer, wer = compute_cer_wer(true_text, pred_text)
            model_result = results.get(model_name, [])
            model_result.append({"CER": cer, "WER": wer, "Predicted Text": pred_text})
            results[model_name] = model_result
        for model_name, metrics_list in results.items():
            logging.info(f"Model: {model_name}, CER: {metrics_list[0]['CER']}, WER: {metrics_list[0]['WER']}")
            model_results = summary_results.get(model_name, {"CER": [], "WER": []})
            model_results["CER"].append(metrics["CER"])
            model_results["WER"].append(metrics["WER"])
            summary_results[model_name] = model_results

    # Print final summary
    logging.info("Final Summary:")
    for model_name, metrics in summary_results.items():
        median_cer = np.median(metrics["CER"])
        median_wer = np.median(metrics["WER"])
        logging.info(f"Model: {model_name}, Median CER: {median_cer}, Median WER: {median_wer}")
