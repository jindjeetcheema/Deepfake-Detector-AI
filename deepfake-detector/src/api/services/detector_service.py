import torch
import torchvision.transforms as transforms
import cv2
import tempfile
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image

# Path to your trained model (update if different)
MODEL_PATH = "models/final_model.pth"

# -----------------------------------------------------
# Load model
# -----------------------------------------------------
def load_model():
    """
    Loads the trained PyTorch model once and keeps it in memory.
    """
    from src.models.xception import Xception  # or EfficientNet if you used that

    model = Xception(num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


# Create global singleton
_model = None
_mtcnn = None


# -----------------------------------------------------
# Frame pre-processing and inference
# -----------------------------------------------------
def predict_deepfake(file_bytes: bytes):
    """
    Accepts raw video bytes, extracts faces from frames,
    runs inference, and returns confidence score.
    """
    global _model, _mtcnn
    if _model is None:
        _model = load_model()
    if _mtcnn is None:
        _mtcnn = MTCNN(keep_all=False, device="cpu")

    # Save uploaded video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(file_bytes)
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, frame_count - 1, num=10, dtype=int)  # 10 frames

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    scores = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = _mtcnn(frame_rgb)

        if face is None:
            continue

        img_tensor = transform(face).unsqueeze(0)
        with torch.no_grad():
            output = _model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            fake_prob = probs[0][1].item()
            scores.append(fake_prob)

    cap.release()

    if len(scores) == 0:
        return {"result": "No face detected", "confidence": 0.0}

    avg_conf = float(np.mean(scores))
    label = "Fake" if avg_conf > 0.5 else "Real"
    return {"result": label, "confidence": round(avg_conf, 3)}
