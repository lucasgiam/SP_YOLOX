"""
Draws bounding boxes over detected objects.
"""

import cv2
import numpy as np
from cv2 import FONT_HERSHEY_SIMPLEX, LINE_AA
from typing import Any, Dict, List, Tuple

from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.nodes.draw.utils.constants import (
    BLACK,
    CHAMPAGNE,
    FILLED,
    NORMAL_FONTSCALE,
    POINT_RADIUS,
    PRIMARY_PALETTE,
)
from peekingduck.pipeline.nodes.draw.utils.constants import (
    PRIMARY_PALETTE_LENGTH as TOTAL_COLORS,
)
from peekingduck.pipeline.nodes.draw.utils.constants import THICK, VERY_THICK
from peekingduck.pipeline.nodes.draw.utils.general import (
    get_image_size,
    project_points_onto_original_image,
)


def draw_bboxes(
    frame: np.ndarray,
    bboxes: List[List[float]],
    bbox_labels: List[str],
    show_labels: bool,
    color_choice: Tuple[int, int, int] = None,
) -> None:
    """Draws bboxes onto an image frame.
    Args:
        frame (np.ndarray): Image of current frame.
        bboxes (List[List[float]]): Bounding box coordinates.
        color (Tuple[int, int, int]): Color used for bounding box.
        bbox_labels (List[str]): Labels of object detected.
    """
    image_size = get_image_size(frame)
    # Get unique label color indexes
    color_idx = {label: idx for idx, label in enumerate(set(bbox_labels))}

    for i, bbox in enumerate(bboxes):
        if color_choice:
            color = color_choice[i]
        else:
            color = PRIMARY_PALETTE[color_idx[bbox_labels[i]] % TOTAL_COLORS]
        if show_labels:
            _draw_bbox(frame, bbox, image_size, color, bbox_labels[i])
        else:
            _draw_bbox(frame, bbox, image_size, color)


def _draw_bbox(
    frame: np.ndarray,
    bbox: np.ndarray,
    image_size: Tuple[int, int],
    color: Tuple[int, int, int],
    bbox_label: str = None,
) -> None:
    """Draws a single bounding box."""
    top_left, bottom_right = project_points_onto_original_image(bbox, image_size)
    cv2.rectangle(
        frame,
        (top_left[0], top_left[1]),
        (bottom_right[0], bottom_right[1]),
        color,
        VERY_THICK,
    )

    if bbox_label:
        _draw_label(frame, top_left, bbox_label, color, BLACK)


def _draw_label(
    frame: np.ndarray,
    top_left: Tuple[int, int],
    bbox_label: str,
    bg_color: Tuple[int, int, int],
    text_color: Tuple[int, int, int],
) -> None:
    """Draws bbox label at top left of bbox."""
    # get label size
    (text_width, text_height), baseline = cv2.getTextSize(
        bbox_label, FONT_HERSHEY_SIMPLEX, NORMAL_FONTSCALE, THICK
    )
    # put filled text rectangle
    cv2.rectangle(
        frame,
        (top_left[0], top_left[1]),
        (top_left[0] + text_width, top_left[1] - text_height - baseline),
        bg_color,
        FILLED,
    )

    # put text above rectangle
    bbox_label = bbox_label[:1].capitalize() + bbox_label[1:]
    cv2.putText(
        frame,
        bbox_label,
        (top_left[0], top_left[1] - 6),
        FONT_HERSHEY_SIMPLEX,
        NORMAL_FONTSCALE,
        text_color,
        THICK,
        LINE_AA,
    )


def draw_tags(
    frame: np.ndarray,
    bboxes: np.ndarray,
    tags: List[str],
    color: Tuple[int, int, int],
) -> None:
    """Draw tags above bboxes.
    Args:
        frame (np.ndarray): Image of current frame.
        bboxes (np.ndarray): Bounding box coordinates.
        tags (Union[List[str], List[int]]): Tag associated with bounding box.
        color (Tuple[int, int, int]): Color of text.
    """
    image_size = get_image_size(frame)
    for idx, bbox in enumerate(bboxes):
        _draw_tag(frame, bbox, tags[idx], image_size, color)


def _draw_tag(
    frame: np.ndarray,
    bbox: np.ndarray,
    tag: str,
    image_size: Tuple[int, int],
    color: Tuple[int, int, int],
) -> None:
    """Draws a tag above a single bounding box."""
    top_left, btm_right = project_points_onto_original_image(bbox, image_size)

    # Find offset to centralize text
    (text_width, _), baseline = cv2.getTextSize(
        tag, FONT_HERSHEY_SIMPLEX, NORMAL_FONTSCALE, THICK
    )
    bbox_width = btm_right[0] - top_left[0]
    offset = int((bbox_width - text_width) / 2)
    position = (top_left[0] + offset, top_left[1] - baseline)

    cv2.putText(
        frame, tag, position, FONT_HERSHEY_SIMPLEX, NORMAL_FONTSCALE, color, VERY_THICK
    )


def draw_pts(frame: np.ndarray, pts: List[Tuple[float]]) -> None:
    """Draw pts of selected object onto frame.
    Args:
        frame (np.array): Image of current frame.
        pts (List[Tuple[float]]): Bottom midpoints of bboxes.
    """
    for point in pts:
        cv2.circle(frame, point, POINT_RADIUS, CHAMPAGNE, -1)


class Node(AbstractNode):
    """Draws bounding boxes on image.
    The :mod:`draw.bbox` node uses :term:`bboxes` and, optionally,
    :term:`bbox_labels` from the model predictions to draw the bbox predictions
    onto the image.
    Inputs:
        |img_data|
        |bboxes_data|
        |bbox_labels_data|
    Outputs:
        |none_output_data|
    Configs:
        show_labels (:obj:`bool`): **default = False**. |br|
            If ``True``, shows class label, e.g., "person", above the bounding
            box.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        color_ppe_status = []
        for status in inputs["bbox_labels"]:
            if status == 'all ppe':
                color = [0, 255, 0]
            elif status == "no mask & vest" or status == "no helmet & vest" or status == "no helmet & mask":
                color = [0, 100, 255]
            elif status == "no helmet" or status == "no vest" or status == "no mask":
                color = [0, 200, 255]
            else:
                color = [0, 0, 255]
            color_ppe_status.append(color)
        draw_bboxes(
            inputs["img"], inputs["bboxes"], inputs["bbox_labels"], self.show_labels, color_ppe_status
        )
        return {}

    def _get_config_types(self) -> Dict[str, Any]:
        """Returns dictionary mapping the node's config keys to respective types."""
        return {"show_labels": bool}
