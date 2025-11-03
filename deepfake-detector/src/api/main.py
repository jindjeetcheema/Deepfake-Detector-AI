from fastapi import FastAPI
from src.api.routers import detection

# create FastAPI instance
app = FastAPI(title="Deepfake Detection API")

# include routes
app.include_router(detection.router)

# simple health check route
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Deepfake API is running"}
    