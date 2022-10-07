"""
Writes the output video to images.
"""

import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from collections import deque
import cv2
import os
import numpy as np

from peekingduck.pipeline.nodes.abstract_node import AbstractNode


class Node(AbstractNode):

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.num_img_to_store: int
        self.output_dir = Path(self.output_dir)  # type: ignore
        self._file_name: Optional[str] = None
        self._file_path_with_timestamp: Optional[str] = None
        self.writer = None
        self._prepare_directory(self.output_dir)
        self._fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.logger.info(f"Output directory used is: {self.output_dir}")
        self.img_filepaths = deque([])
        self.frame_counter = 0


    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Writes media information to filepath."""
        # reset and terminate when there are no more data
        if inputs["pipeline_end"]:
            if self.writer:  # images automatically releases writer
                self.writer.release()
            return {}
        images = os.listdir(self.output_dir)
        if self.frame_counter == 0 and len(images) != 0:
            [os.remove(os.path.join(self.output_dir, image)) for image in images]
        self._prepare_writer(inputs["filename"])
        self._write(inputs["img"])
        filename = str(Path(self._file_path_with_timestamp).stem + ".jpg")
        output = {"filename": filename}
        self._delete()
        self.frame_counter += 1
        return output


    def _write(self, img: np.ndarray) -> None:
        self._file_path_with_timestamp = Path(self._file_path_with_timestamp).with_suffix(".jpg")
        cv2.imwrite(str(self._file_path_with_timestamp), img)


    def _delete(self):
        self.img_filepaths.append(self._file_path_with_timestamp)
        if len(self.img_filepaths) >= self.num_img_to_store:
            img_to_delete = self.img_filepaths.popleft()
            os.remove(str(img_to_delete))


    def _prepare_writer(self, filename: str) -> None:
        self._file_path_with_timestamp = self._append_datetime_filename(filename)


    @staticmethod
    def _prepare_directory(output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

    def _append_datetime_filename(self, filename: str) -> str:
        self._file_name = filename
        current_time = datetime.datetime.now()
        # output as 'YYYYMMDD_hhmmss'
        time_str = current_time.strftime("%y%m%d_%H%M%S_%f")

        # append timestamp to filename before extension Format: filename_timestamp.extension
        filename_with_timestamp = f"_{time_str}.".join(filename.split(".")[-2:])
        file_path_with_timestamp = self.output_dir / filename_with_timestamp

        return str(file_path_with_timestamp)