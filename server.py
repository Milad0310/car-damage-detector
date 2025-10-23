import os, io, torch
from ultralytics import YOLO
from torch_snippets import P, makedir
from sdd import SDD
from PIL import Image
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO("sdd.weights.pt")

server_root = P('/tmp')
templates = './templates'
static = server_root/'server/static'
files = server_root/'server/files'
for fldr in [static,files]: makedir(fldr)

app = FastAPI()
app.mount("/static", StaticFiles(directory=static), name="static")
app.mount("/files", StaticFiles(directory=files), name="files")
templates = Jinja2Templates(directory=templates)

@app.get("/")
async def read_item(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post('/uploaddata/')
async def upload_file(request: Request, file:UploadFile=File(...)):
    print(request)
    content = file.file.read()
    saved_filepath = f'{files}/{file.filename}'
    with open(saved_filepath, 'wb') as f:
        f.write(content)
    output = model.predict_from_path(saved_filepath)
    payload = {
        'request': request, 
        "filename": file.filename, 
        'output': output
    }
    return templates.TemplateResponse("home.html", payload)


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    content = file.file.read()
    image = Image.open(io.BytesIO(content))

    results = model(image)  # YOLOv8 results object
    output_list = []

    for result in results:
        for box in result.boxes:  # iterate over detected boxes
            output_list.append({
                "xyxy": box.xyxy.tolist(),           # bounding box coordinates
                "confidence": float(box.conf),       # confidence score
                "class": int(box.cls[0])             # take the first element of the tensor
            })

    return {"predictions": output_list}


