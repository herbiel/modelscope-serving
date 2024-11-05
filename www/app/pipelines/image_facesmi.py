from typing import Any

from app.pipelines.singleton_instance import SingletonInstance
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import numpy as np

class FaceSim(SingletonInstance):
    def build(self):
        # https://www.modelscope.cn/models/damo/cv_hrnetv2w32_body-2d-keypoints_image/summary
        return pipeline(
            task=Tasks.face_recognition,
            model='damo/cv_ir_face-recognition-ood_rts',
        )

    def handle(self, image1: Any,image2:Any):

        if image1 is None or image2 is None:
            # If either embedding is None, return an error response
            output = {
                "code": 400,
                "error": "Failed to get images",
                "score": None
            }

        try:
            emb1 = self.instance()(image1)[OutputKeys.IMG_EMBEDDING]
            emb2 = self.instance()(image2)[OutputKeys.IMG_EMBEDDING]
            sim = np.dot(emb1[0], emb2[0])
            print("{image1} with {image2} sim is {str(sim)}")
            return {
                "code": 200,
                "error": None,
                "score": str(sim)
            }
        except Exception as e:
            # Handle any unexpected errors during the similarity calculation
            return {
                "code": 500,
                "error": str(e),
                "sim": None
            }