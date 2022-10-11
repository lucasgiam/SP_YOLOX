"""
Custom node to show object detection scores
"""

from typing import Any, Dict, List, Tuple
import cv2
from peekingduck.pipeline.nodes.node import AbstractNode


def map_bbox_to_image_coords(
    bbox: List[float], image_size: Tuple[int, int]
) -> List[int]:
    """This is a helper function to map bounding box coords (relative) to
    image coords (absolute).
    Bounding box coords ranges from 0 to 1
    where (0, 0) = image top-left, (1, 1) = image bottom-right.

    Args:
       bbox (List[float]): List of 4 floats x1, y1, x2, y2
       image_size (Tuple[int, int]): Width, Height of image

    Returns:
       List[int]: x1, y1, x2, y2 in integer image coords
    """
    width, height = image_size[0], image_size[1]
    x1, y1, x2, y2 = bbox
    x1 *= width
    x2 *= width
    y1 *= height
    y2 *= height
    return int(x1), int(y1), int(x2), int(y2)


class Node(AbstractNode):
    """This is a template class of how to write a node for PeekingDuck,
       using AbstractNode as the parent class.
       This node draws scores on objects detected.

    Args:
       config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        """Node initializer

        Since we do not require any special setup, it only calls the __init__
        method of its parent class.
        """
        super().__init__(config, node_path=__name__, **kwargs)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This method implements the display score function.
        As PeekingDuck iterates through the CV pipeline, this 'run' method
        is called at each iteration.

        Args:
              inputs (dict): Dictionary with keys "img", "bboxes", "bbox_scores"

        Returns:
              outputs (dict): Empty dictionary
        """

        # extract pipeline inputs and compute image size in (width, height)
        img = inputs["img"]
        bboxes = inputs["bboxes"]
        scores = inputs["bbox_scores"]
        labels = inputs["bbox_labels"]
        img_size = (img.shape[1], img.shape[0])  # width, height

        # assign conditional colors to text
        color_score_status = []
        for status in labels:
            if status == 'all ppe':
                score_color = [0, 255, 0]
            elif status == "no mask & vest" or status == "no helmet & vest" or status == "no helmet & mask":
                score_color = [0, 100, 255]
            elif status == "no helmet" or status == "no vest" or status == "no mask":
                score_color = [0, 200, 255]
            else:
                score_color = [0, 0, 255]
            color_score_status.append(score_color)

        for i, bbox in enumerate(bboxes):
            # for each bounding box:
            #   - compute (x1, y1) top-left, (x2, y2) bottom-right coordinates
            #   - convert score into a two decimal place numeric string
            #   - draw score string onto image using opencv's putText()
            #     (see opencv's API docs for more info)
            x1, y1, x2, y2 = map_bbox_to_image_coords(bbox, img_size)
            score = scores[i]
            score_str = f"{score:0.2f}"
            color_status = color_score_status[i]
            cv2.putText(
                img=img,
                text=score_str,
                org=(x1, y2),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=color_status,
                thickness=3,
            )

        return {}               # node has no outputs
