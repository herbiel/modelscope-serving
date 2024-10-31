from app.internal.utils import base64_to_numpy,url_to_numpy
# from app.pipelines.instances import body_2d_keypoint
from app.pipelines.model_manager import get_model_manager, ModelManager
from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel
# https://www.modelscope.cn/docs/%E9%83%A8%E7%BD%B2EAS
router = APIRouter()

@router.post('/facedetect')
def post_facedetect(
        image1: str = Body(embed=True,alias="image1", min_length=10),
        model_manager: ModelManager = Depends(get_model_manager)
):
    if not image1:
        raise HTTPException(status_code=422, detail="Request Error, invalid image")
    result = model_manager.facedetect.handle(image1)

    return result
