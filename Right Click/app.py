from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import onnxruntime as ort
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

app = FastAPI(title="Space Debris Detection API")

# Load YOLO and MobileNet ONNX models
yolo_sess = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])
mobilenet_sess = ort.InferenceSession("mobilenet_classifier.onnx", providers=["CPUExecutionProvider"])

@app.get("/")
def home():
    return {"message": "Space Debris Detection API is running 🚀"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Load image
    image = Image.open(BytesIO(await file.read())).convert("RGB")
    img_np = np.array(image)

    # Preprocess for YOLO (resize to 640x640)
    img_resized = cv2.resize(img_np, (640, 640))
    img_input = img_resized.transpose(2, 0, 1)[np.newaxis, :] / 255.0
    img_input = img_input.astype(np.float32)

    # Run YOLO model
    yolo_input_name = yolo_sess.get_inputs()[0].name
    yolo_output_name = yolo_sess.get_outputs()[0].name
    yolo_outputs = yolo_sess.run([yolo_output_name], {yolo_input_name: img_input})[0]

    # Draw fake bounding boxes (you’ll replace this with proper postprocessing)
    for det in yolo_outputs[:5]:  # example: first 5 detections
        x1, y1, x2, y2, conf, cls = map(int, det[:6])
        cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Run MobileNet classifier (optional per detection)
    mobilenet_input_name = mobilenet_sess.get_inputs()[0].name
    mobilenet_output_name = mobilenet_sess.get_outputs()[0].name
    mobilenet_result = mobilenet_sess.run([mobilenet_output_name], {mobilenet_input_name: img_input})

    # Convert back to image
    _, img_encoded = cv2.imencode('.jpg', img_resized)
    return StreamingResponse(BytesIO(img_encoded.tobytes()), media_type="image/jpeg")
