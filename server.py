# server.py
import io
import os
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from ultralytics import YOLO

# CONFIG
MODEL_PATH = os.environ.get("MODEL_PATH", "sdd.weights.pt")  # default filename used previously
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# if your model uses custom classes, you can provide labels.txt in the image below folder or in container

# load model
# Ultralytics YOLO loads a model and will move to cuda if available
model = YOLO(MODEL_PATH)
try:
    model.to(DEVICE)
except Exception:
    # some ultralytics versions set device during init; ignore if not supported
    pass

app = FastAPI(title="YOLO CV Model API (Unity)")

# Allow CORS from Unity editor / builds (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for production restrict to your domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}

@app.post("/predict")
async def predict(file: UploadFile = File(...), conf_threshold: float = 0.25, iou_threshold: float = 0.45):
    """
    Accepts an image file and returns YOLO detections in JSON.
    - file: multipart/form-data field "file"
    - conf_threshold: minimum confidence (0..1)
    - iou_threshold: nms iou threshold (0..1)
    """
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open image: {e}")

    # run model (Ultralytics returns a Results object or list of Results)
    # we pass deviceless PIL image directly; ultralytics handles preprocessing
    results = model(image, conf=conf_threshold, iou=iou_threshold)

    output_list = []
    # results may be iterable (one per image); here we only sent one image
    for res in results:
        # res.boxes is an ultralytics Boxes object
        # iterate boxes and extract fields
        boxes = getattr(res, "boxes", None)
        if boxes is None:
            continue

        # boxes.xyxy, boxes.conf, boxes.cls are tensors; iterate per box
        # support both older and newer ultralytics APIs
        try:
            xyxys = boxes.xyxy.cpu().tolist()
        except Exception:
            # fallback: some versions expose .xyxy directly as list
            xyxys = [b.tolist() for b in boxes.xyxy]

        # confs and classes may be tensors or attributes on each box item
        try:
            confs = boxes.conf.cpu().numpy().tolist()
        except Exception:
            # older API may place conf/cls inside box objects
            confs = []
            for b in boxes:
                try:
                    confs.append(float(b.conf))
                except Exception:
                    confs.append(None)

        try:
            clss = boxes.cls.cpu().numpy().astype(int).tolist()
        except Exception:
            clss = []
            for b in boxes:
                try:
                    clss.append(int(b.cls[0]))
                except Exception:
                    clss.append(None)

        # Combine into output entries
        for xyxy, conf, cls in zip(xyxys, confs, clss):
            output_list.append({
                "xyxy": [float(x) for x in xyxy],       # [x1, y1, x2, y2]
                "confidence": float(conf) if conf is not None else None,
                "class": int(cls) if cls is not None else None
            })

    return {"predictions": output_list}
