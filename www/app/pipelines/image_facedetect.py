from typing import Any

from app.pipelines.singleton_instance import SingletonInstance
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import numpy as np

class FaceDetect(SingletonInstance):
    def build(self):
        # https://www.modelscope.cn/models/damo/cv_hrnetv2w32_body-2d-keypoints_image/summary
        return pipeline(
            task=Tasks.face_detection,
            model='damo/cv_manual_face-detection_tinymog',
            model_revision='v2.0.2',  # 20230130
        )

    def handle(self, image1: Any):
        raw_result = self.instance()(image1)
        print(raw_result)
        if raw_result:
            output = str(raw_result)
        return output
