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
            task=Tasks.face_recognition,
            model='damo/cv_ir101_facerecognition_cfglint',
            model_revision='v2.0.2',  # 20230130
        )

    def handle(self, image1: Any,image2:Any):
        print(image1)
        print(image2)
        emb1 = self.instance()(image1)[OutputKeys.IMG_EMBEDDING]
        emb2 = self.instance()(image2)[OutputKeys.IMG_EMBEDDING]
        sim = np.dot(emb1[0], emb2[0])
        print(sim)
        if sim:
            output = str(sim)
        return output
