# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Base classes for OCR handwritten text processing
__author__ = "Nina Belyavskaya"

import numpy as np


class BaseLineSplitter:
    """ Base class for line splitters. """
    def split_into_lines(self, image: np.ndarray) -> list[np.ndarray]:
        """
        Args:
            image (np.ndarray): Input image as a NumPy array.
        Returns:
            list[np.ndarray]: List of NumPy arrays, each containing one text line.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class BaseOCR():
    """ Base class combining line splitting and OCR processing. """
    def __init__(self, line_splitter: BaseLineSplitter = None):
        self.line_splitter = line_splitter

    def recognize(self, line_image: np.ndarray) -> str:
        """
        Recognizes text from a single line image.
        Args:
            line_image (np.ndarray): Input line image as a NumPy array.
        Returns:
            str: Recognized text from the line image.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def recognize_text(self, image: np.ndarray) -> str:
        """ Recognizes text from the entire image by splitting into lines first. """
        if self.line_splitter is None:
            return self.recognize(image)
        lines = self.line_splitter.split_into_lines(image)
        return "\n".join([self.recognize(line) for line in lines])
