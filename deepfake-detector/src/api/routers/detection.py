from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from src.api.services.detector_service import predict_deepfake

router = APIRouter(prefix="/detect", tags=["Deepfake Detection"])

@router.post("/")
async def detect_video(file: UploadFile = File(...)):
    """
    Endpoint that calls the deepfake detector.
    """
    file_bytes = await file.read()
    result = predict_deepfake(file_bytes)
    return JSONResponse(result)
