import os
import sys
import base64
import time
import tempfile
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ["KERAS_BACKEND"] = "torch"
import moonshine

from pipeline import pipeline_instance
from App.select_final import selectFinal
from App.include50_pipeline import word_pipeline_instance, WORD_CATALOG

app = FastAPI(title="Indian Sign Language Translator", version="1.0.0")

# Mount static and templates
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "web", "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "web", "static")), name="static")

@app.on_event("startup")
async def startup_event():
    print("[Server] Initializing Indian Sign Language Pipeline...")
    pipeline_instance.initialize()
    print("[Server] Pipeline is ready to serve predictions!")
    
    print("[Server] Initializing INCLUDE-50 Word-Level Pipeline...")
    try:
        word_pipeline_instance.initialize()
        print("[Server] INCLUDE-50 Word-Level Pipeline is ready!")
    except Exception as e:
        print("[Server] Note initializing INCLUDE-50 Pipeline:", e)

    print("[Server] Pre-warming Moonshine base STT model...")
    try:
        dummy_audio = os.path.join(str(moonshine.ASSETS_DIR), 'beckett.wav')
        if os.path.exists(dummy_audio):
            _ = moonshine.transcribe(dummy_audio, 'moonshine/base')
        print("[Server] Moonshine base STT model is ready!")
    except Exception as e:
        print("[Server] Moonshine warmup note:", e)

class FrameData(BaseModel):
    image: str # base64 encoded data url or raw base64
    detect_face_only: Optional[bool] = False

class WordFrameData(BaseModel):
    image: str
    reset: Optional[bool] = False
    return_annotated: Optional[bool] = False

class WordClipData(BaseModel):
    images: List[str]

def decode_base64_image(b64_str: str) -> np.ndarray:
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    img_data = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image data")
    return img

@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "classes": pipeline_instance.class_map,
            "total_classes": len(pipeline_instance.class_map),
            "word_classes": word_pipeline_instance.display_map,
            "total_word_classes": len(word_pipeline_instance.display_map),
            "word_catalog": WORD_CATALOG
        }
    )

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "pipeline_ready": pipeline_instance.initialized,
        "word_pipeline_ready": word_pipeline_instance.initialized,
        "classes": pipeline_instance.class_map,
        "word_classes": word_pipeline_instance.display_map,
        "models": {
            "yolo": "cross-hands YOLOv3 (Darknet)",
            "sign_recognition": "SqueezeNet CNN (Transfer Learning)",
            "segmentation": "HSV + YCbCr Skin Filter + Watershed",
            "word_recognition": "AI4Bharat INCLUDE-50 BiLSTM + MediaPipe Landmarks"
        }
    }

@app.get("/api/classes")
async def get_classes():
    return pipeline_instance.class_map

@app.get("/api/words/classes")
async def get_word_classes():
    return {
        "classes": word_pipeline_instance.label_map,
        "display_map": word_pipeline_instance.display_map,
        "total": len(word_pipeline_instance.label_map)
    }

@app.post("/api/predict-word-frame")
async def predict_word_frame(data: WordFrameData):
    try:
        if data.reset:
            word_pipeline_instance.reset_buffer()
        frame = decode_base64_image(data.image)
        res = word_pipeline_instance.process_frame(
            frame, 
            add_to_buffer=True, 
            return_annotated=bool(data.return_annotated)
        )
        return res
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/api/predict-word-clip")
async def predict_word_clip(data: WordClipData):
    try:
        frames = [decode_base64_image(img_b64) for img_b64 in data.images if img_b64]
        res = word_pipeline_instance.process_clip(frames)
        return {"result": res, "total_frames": len(frames)}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/api/reset-word-buffer")
async def reset_word_buffer():
    word_pipeline_instance.reset_buffer()
    return {"status": "reset", "buffer_length": 0}

@app.post("/api/detect-face")
async def detect_face(data: FrameData):
    try:
        frame = decode_base64_image(data.image)
        has_face, boxes = pipeline_instance.detect_face(frame)
        return {
            "face_detected": has_face,
            "face_count": len(boxes),
            "boxes": boxes
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/predict-image")
async def predict_image(file: Optional[UploadFile] = File(None), data: Optional[FrameData] = None):
    try:
        if file is not None:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Could not decode uploaded file")
        elif data is not None and data.image:
            frame = decode_base64_image(data.image)
        else:
            raise ValueError("No image provided")

        result = pipeline_instance.process_frame(frame, return_images=True)
        return result
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/api/predict-stream-frame")
async def predict_stream_frame(data: FrameData):
    try:
        frame = decode_base64_image(data.image)
        # Fast inference returning only classification + bounding boxes, no intermediate base64 images
        result = pipeline_instance.process_frame(frame, return_images=False)
        return result
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/api/process-video")
async def process_video(file: UploadFile = File(...)):
    try:
        suffix = os.path.splitext(file.filename)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            os.remove(tmp_path)
            raise ValueError("Could not read frames from video file")

        # Sample frames (similar to original thesis: ~20% of frames or up to 10 sample frames)
        sample_count = min(12, max(5, total_frames // 5))
        step = max(1, total_frames // sample_count)
        indices_to_sample = [i * step for i in range(sample_count) if i * step < total_frames]

        frame_results = []
        vote_counter = {}

        current_idx = 0
        while cap.isOpened() and len(frame_results) < len(indices_to_sample):
            ret, frame = cap.read()
            if not ret:
                break
            if current_idx in indices_to_sample:
                res = pipeline_instance.process_frame(frame, return_images=True)
                sign = res["predicted_sign"]
                vote_counter[sign] = vote_counter.get(sign, 0) + 1
                frame_results.append({
                    "frame_index": current_idx,
                    "predicted_sign": sign,
                    "confidence": res["confidence"],
                    "hand_detected": res["hand_detected"],
                    "images": res.get("images", {})
                })
            current_idx += 1

        cap.release()
        os.remove(tmp_path)

        if not vote_counter:
            raise ValueError("No frames processed from video")

        consensus_sign = selectFinal.select_max(vote_counter)
        total_votes = sum(vote_counter.values())
        consensus_confidence = round((vote_counter[consensus_sign] / total_votes) * 100, 1)

        return {
            "consensus_sign": consensus_sign,
            "consensus_confidence": consensus_confidence,
            "vote_distribution": vote_counter,
            "total_frames_analyzed": len(frame_results),
            "frame_results": frame_results
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/api/transcribe-speech")
async def transcribe_speech(file: UploadFile = File(...)):
    tmp_path = None
    wav_path = None
    try:
        suffix = os.path.splitext(file.filename or "")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            if not content:
                raise ValueError("Empty audio payload")
            tmp.write(content)
            tmp_path = tmp.name

        wav_path = tmp_path
        # If not standard wav, convert with ffmpeg to 16kHz mono WAV
        if not tmp_path.lower().endswith(".wav"):
            wav_path = tmp_path + ".converted.wav"
            ret = os.system(f"/opt/homebrew/bin/ffmpeg -y -i '{tmp_path}' -ar 16000 -ac 1 '{wav_path}' >/dev/null 2>&1")
            if ret != 0 or not os.path.exists(wav_path):
                wav_path = tmp_path

        res = moonshine.transcribe(wav_path, "moonshine/base")
        text = " ".join([t.strip() for t in res if t.strip()])
        return {"text": text, "model": "moonshine/base"}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    finally:
        for p in [tmp_path, wav_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8050, reload=False)
