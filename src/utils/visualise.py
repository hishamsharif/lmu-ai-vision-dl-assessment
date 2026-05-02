"""
visualise.py  —  Shared Visualisation Helpers
----------------------------------------------
Bounding box drawing, image grid display, and plot utilities
shared across Section A and Section B notebook cells.

Implement:
    draw_boxes(image, boxes, class_ids, scores, score_threshold, figsize)
    load_image_as_array(image_path, target_size)
    plot_image_grid(images, titles, cols, figsize, output_path)
"""

import numpy as np


def draw_boxes(image: np.ndarray,
               boxes: np.ndarray,
               class_ids: np.ndarray,
               scores: np.ndarray,
               score_threshold: float = 0.3,
               figsize: tuple = (10, 10)):
    pass


def load_image_as_array(image_path: str,
                         target_size: tuple = None) -> np.ndarray:
    pass


def plot_image_grid(images: list,
                    titles: list,
                    cols: int = 3,
                    figsize: tuple = (15, 5),
                    output_path: str = None):
    pass
