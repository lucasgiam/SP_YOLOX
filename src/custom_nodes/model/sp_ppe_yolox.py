from typing import Any, Dict

import numpy as np
import torch

from peekingduck.pipeline.nodes.node import AbstractNode

from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import postprocess


class Node(AbstractNode):

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.exp = get_exp(self.exp_path)
        self.test_size = self.exp.test_size
        self.device: str
        self.fp16: bool
        self.num_classes = self.exp.num_classes
        self.confthre: float
        self.nmsthre: float
        self.model = self.load_model()

    def load_model(self):
        model = self.exp.get_model()
        ckpt = torch.load(self.ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if self.device == "gpu":
            model.cuda()
        model.eval()
        return model

    def preprocess(self, img):
        img_info = {"id": 0}
        img_info["file_name"] = None
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(
            self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1]
        )
        img_info["ratio"] = ratio

        img, _ = ValTransform(legacy=False)(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16
        return img, img_info

    def postprocess(self, predictions, img_info):
        predictions = postprocess(
            predictions,
            self.num_classes,
            self.confthre,
            self.nmsthre,
            class_agnostic=True,
        )
        ratio = img_info["ratio"]
        img = img_info["raw_img"]

        predictions = predictions[0]   # Removed outer list

        if predictions is None:
            return np.empty((0, 4)), np.empty(0), np.empty(0)
        bboxes = predictions[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio
        # print(predictions)
        class_id = predictions[:, 6]
        scores = predictions[:, 4] * predictions[:, 5]

        bboxes = bboxes.cpu().detach().numpy()
        class_id = class_id.cpu().detach().numpy()
        scores = scores.cpu().detach().numpy()

        # print("unnormalised bboxes", bboxes)

        bboxes[..., [0, 2]] = bboxes[..., [0, 2]] / img_info["width"]
        bboxes[..., [1, 3]] = bboxes[..., [1, 3]] / img_info["height"]

        return bboxes, class_id, scores

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Reads the image input and returns the predicted class label and
        confidence score.

        Args:
              inputs (dict): Dictionary with key "img".

        Returns:
              outputs (dict): Dictionary with keys "pred_label" and "pred_score".
        """
        img = inputs["img"]
        img, img_info = self.preprocess(img)
        with torch.no_grad():
            predictions = self.model(img)
            bboxes, class_ids, scores = self.postprocess(predictions, img_info)
            class_labels = [self.voc_classes[class_id] for class_id in class_ids]
        outputs = {"bboxes": bboxes, "bbox_labels": class_labels, "bbox_scores": scores}
        return outputs
