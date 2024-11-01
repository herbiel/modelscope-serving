from app.routers.predict import image_facedetect,image_body,image_facesmi
from app.routers.server import router as server_router
from fastapi import APIRouter

predict_routers = APIRouter()
#predict_routers.include_router(image_cartoon.router)
#predict_routers.include_router(image_segment.router)
#predict_routers.include_router(word_segmentation.router)
#predict_routers.include_router(speech_asr.router)
#predict_routers.include_router(speech_tts.router)
#predict_routers.include_router(image_ocr.router)
#predict_routers.include_router(image_body.router)
predict_routers.include_router(image_facedetect.router)
predict_routers.include_router(image_facesmi.router)

routers = APIRouter(prefix="/api")
routers.include_router(predict_routers, prefix="/predict")
routers.include_router(server_router, prefix="/server")

__all__ = ["routers"]
