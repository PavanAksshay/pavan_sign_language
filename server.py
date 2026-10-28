import os
import sys
import base64
import time
import tempfile
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Form
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
from phase18_collector import collection_engine, CONDITION_PRESETS, ALLOWED_SIGNERS

app = FastAPI(title="Indian Sign Language Translator", version="1.0.0")

# Mount static and templates
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "web", "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "web", "static")), name="static")

@app.on_event("startup")
async def startup_event():
    print("[Server] Initializing Indian Sign Language Pipeline...")
    pipeline_instance.initialize()
    print(f"[Server] Pipeline is ready to serve predictions! (Lap-hand suppression: {'ENABLED' if pipeline_instance.suppress_lap_hand else 'DISABLED'})")
    
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
    session_id: Optional[str] = "default_session"

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
    detector_name = "MediaPipe HandLandmarker" if pipeline_instance.backend_detector == "mediapipe" else "cross-hands YOLOv3 (Darknet)"
    if pipeline_instance.recognition_arch == "model_c":
        classifier_name = "Model C (280d Occlusion-Aware Geometry Random Forest)"
    elif pipeline_instance.recognition_arch == "late_fusion":
        classifier_name = "Late-Fusion Ensemble (MediaPipe SqueezeNet v1 + Landmark Random Forest, alpha=0.25)"
    elif pipeline_instance.backend_detector == "mediapipe":
        classifier_name = "MediaPipe Fine-Tuned SqueezeNet v1"
    else:
        classifier_name = "SqueezeNet CNN (Transfer Learning)"

    models_info = {
        "detector": detector_name,
        "yolo": detector_name,
        "sign_recognition": classifier_name,
        "segmentation": "HSV + YCbCr Skin Filter + Watershed",
        "word_recognition": "AI4Bharat INCLUDE-50 BiLSTM + MediaPipe Landmarks"
    }
    if pipeline_instance.shadow_mode_enabled:
        models_info["shadow_recognizer"] = "Model C (280d Occlusion-Aware Geometry Random Forest)"

    return {
        "status": "healthy",
        "backend_detector": pipeline_instance.backend_detector,
        "recognition_arch": pipeline_instance.recognition_arch,
        "shadow_mode_enabled": pipeline_instance.shadow_mode_enabled,
        "temporal_smoothing_enabled": pipeline_instance.temporal_smoothing_enabled,
        "mediapipe_delegate": pipeline_instance.mediapipe_delegate,
        "model_c_input_resolution": getattr(pipeline_instance, "model_c_input_resolution", "native"),
        "pipeline_ready": pipeline_instance.initialized,
        "word_pipeline_ready": word_pipeline_instance.initialized,
        "diagnostics_enabled": pipeline_instance.diagnostics_enabled,
        "effective_config": {
            "backend_detector": pipeline_instance.backend_detector,
            "recognition_arch": pipeline_instance.recognition_arch,
            "shadow_mode_enabled": pipeline_instance.shadow_mode_enabled,
            "shadow_async": pipeline_instance.shadow_async,
            "mediapipe_delegate": pipeline_instance.mediapipe_delegate,
            "model_c_input_resolution": getattr(pipeline_instance, "model_c_input_resolution", "native"),
            "temporal_smoothing_enabled": pipeline_instance.temporal_smoothing_enabled,
            "suppress_lap_hand": pipeline_instance.suppress_lap_hand,
            "lap_hand_min_y_ratio": pipeline_instance.lap_hand_min_y_ratio,
            "lap_hand_max_active_y_ratio": pipeline_instance.lap_hand_max_active_y_ratio,
            "direct_resize": pipeline_instance.direct_resize,
            "square_crop": pipeline_instance.square_crop,
            "filter_edge_detections": pipeline_instance.filter_edge_detections,
            "conservative_face_clamp": pipeline_instance.conservative_face_clamp,
            "improved_two_hand_merge": pipeline_instance.improved_two_hand_merge
        },
        "classes": pipeline_instance.class_map,
        "word_classes": word_pipeline_instance.display_map,
        "models": models_info
    }

@app.get("/api/backend/status")
async def get_backend_status():
    if pipeline_instance.recognition_arch == "model_c":
        clf_model = "Model C (280d Occlusion-Aware Geometry Random Forest)"
    elif pipeline_instance.recognition_arch == "late_fusion":
        clf_model = "Late-Fusion Ensemble (alpha=0.25)"
    elif pipeline_instance.backend_detector == "mediapipe":
        clf_model = "MediaPipe Fine-Tuned SqueezeNet v1"
    else:
        clf_model = "SqueezeNet CNN (Transfer Learning)"

    return {
        "backend_detector": pipeline_instance.backend_detector,
        "recognition_arch": pipeline_instance.recognition_arch,
        "shadow_mode_enabled": pipeline_instance.shadow_mode_enabled,
        "shadow_async": pipeline_instance.shadow_async,
        "mediapipe_delegate": pipeline_instance.mediapipe_delegate,
        "model_c_input_resolution": getattr(pipeline_instance, "model_c_input_resolution", "native"),
        "temporal_smoothing_enabled": pipeline_instance.temporal_smoothing_enabled,
        "detector_model": "MediaPipe HandLandmarker" if pipeline_instance.backend_detector == "mediapipe" else "cross-hands YOLOv3 (Darknet)",
        "classifier_model": clf_model
    }

@app.post("/api/backend/set")
async def set_backend(request: Request):
    try:
        body = await request.json() if "application/json" in request.headers.get("content-type", "") else {}
        backend = body.get("backend")
        if not backend:
            raise HTTPException(status_code=400, detail="Missing 'backend' field in request body")
        pipeline_instance.set_backend(backend)
        return {
            "status": "ok",
            "backend_detector": pipeline_instance.backend_detector,
            "recognition_arch": pipeline_instance.recognition_arch
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/recognition/status")
async def get_recognition_status():
    res = {
        "recognition_arch": pipeline_instance.recognition_arch,
        "shadow_mode_enabled": pipeline_instance.shadow_mode_enabled,
        "shadow_async": pipeline_instance.shadow_async,
        "mediapipe_delegate": pipeline_instance.mediapipe_delegate,
        "model_c_input_resolution": getattr(pipeline_instance, "model_c_input_resolution", "native"),
        "temporal_smoothing_enabled": pipeline_instance.temporal_smoothing_enabled,
        "alpha": pipeline_instance.late_fusion_alpha if pipeline_instance.recognition_arch == "late_fusion" else None
    }
    if pipeline_instance.model_c_recognizer is not None:
        res["model_c_metadata"] = pipeline_instance.model_c_recognizer.get_public_metadata()
    return res

@app.post("/api/recognition/set")
async def set_recognition(request: Request):
    try:
        body = await request.json() if "application/json" in request.headers.get("content-type", "") else {}
        arch = body.get("architecture") or body.get("arch")
        if not arch:
            raise HTTPException(status_code=400, detail="Missing 'architecture' field in request body")
        pipeline_instance.set_recognition_arch(arch)
        if "delegate" in body:
            pipeline_instance.set_mediapipe_delegate(body["delegate"])
        if "resolution" in body:
            pipeline_instance.set_model_c_input_resolution(body["resolution"])
        return {
            "status": "ok",
            "recognition_arch": pipeline_instance.recognition_arch,
            "mediapipe_delegate": pipeline_instance.mediapipe_delegate,
            "model_c_input_resolution": getattr(pipeline_instance, "model_c_input_resolution", "native")
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/delegate/set")
async def set_delegate_endpoint(request: Request):
    try:
        body = await request.json() if "application/json" in request.headers.get("content-type", "") else {}
        delegate = body.get("delegate")
        if not delegate:
            raise HTTPException(status_code=400, detail="Missing 'delegate' field in request body")
        pipeline_instance.set_mediapipe_delegate(delegate)
        return {
            "status": "ok",
            "mediapipe_delegate": pipeline_instance.mediapipe_delegate
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/resolution/set")
async def set_resolution_endpoint(request: Request):
    try:
        body = await request.json() if "application/json" in request.headers.get("content-type", "") else {}
        resolution = body.get("resolution")
        if not resolution:
            raise HTTPException(status_code=400, detail="Missing 'resolution' field in request body")
        pipeline_instance.set_model_c_input_resolution(resolution)
        return {
            "status": "ok",
            "model_c_input_resolution": getattr(pipeline_instance, "model_c_input_resolution", "native")
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/shadow-mode/status")
async def get_shadow_mode_status():
    return {
        "shadow_mode_enabled": pipeline_instance.shadow_mode_enabled,
        "shadow_async": pipeline_instance.shadow_async,
        "model_c_loaded": pipeline_instance.model_c_recognizer.is_loaded if pipeline_instance.model_c_recognizer else False,
        "telemetry": pipeline_instance.get_shadow_telemetry()
    }

@app.post("/api/shadow-mode/set")
async def set_shadow_mode_endpoint(request: Request):
    try:
        body = await request.json() if "application/json" in request.headers.get("content-type", "") else {}
        if "enabled" in body:
            enabled = bool(body["enabled"])
        elif "shadow_mode" in body:
            enabled = bool(body["shadow_mode"])
        else:
            raise HTTPException(status_code=400, detail="Missing 'enabled' field in request body")
        async_mode = body.get("async_mode")
        pipeline_instance.set_shadow_mode(enabled, async_mode=async_mode)
        return {
            "status": "ok",
            "shadow_mode_enabled": pipeline_instance.shadow_mode_enabled,
            "shadow_async": pipeline_instance.shadow_async
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/temporal-smoothing/set")
async def set_temporal_smoothing(request: Request):
    try:
        body = await request.json() if "application/json" in request.headers.get("content-type", "") else {}
        if "enabled" not in body:
            raise HTTPException(status_code=400, detail="Missing 'enabled' field in request body")
        pipeline_instance.set_temporal_smoothing(bool(body["enabled"]))
        return {
            "status": "ok",
            "temporal_smoothing_enabled": pipeline_instance.temporal_smoothing_enabled
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/temporal-smoothing/reset")
async def reset_temporal_smoothing():
    pipeline_instance.temporal_smoother.reset()
    return {"status": "ok", "buffer": "reset"}

@app.post("/api/diagnostics/toggle")
async def toggle_diagnostics(request: Request):
    try:
        body = await request.json() if "application/json" in request.headers.get("content-type", "") else {}
    except Exception:
        body = {}
    if "enable" in body:
        pipeline_instance.diagnostics_enabled = bool(body["enable"])
    else:
        pipeline_instance.diagnostics_enabled = not pipeline_instance.diagnostics_enabled
    return {
        "status": "ok",
        "diagnostics_enabled": pipeline_instance.diagnostics_enabled,
        "frame_counter": pipeline_instance.frame_counter
    }

@app.get("/api/diagnostics/status")
async def get_diagnostics_status():
    return {
        "diagnostics_enabled": pipeline_instance.diagnostics_enabled,
        "frame_counter": pipeline_instance.frame_counter
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
async def predict_image(request: Request, file: Optional[UploadFile] = File(None)):
    try:
        frame = None
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body = await request.json()
            b64_str = body.get("image")
            if not b64_str:
                raise ValueError("No image provided in JSON")
            frame = decode_base64_image(b64_str)
        elif file is not None:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Could not decode uploaded file")
        else:
            form = await request.form()
            if "file" in form and hasattr(form["file"], "read"):
                contents = await form["file"].read()
                nparr = np.frombuffer(contents, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            elif "image" in form:
                frame = decode_base64_image(str(form["image"]))
            else:
                raise ValueError("No image provided")

        if frame is None:
            raise ValueError("Could not decode image frame")

        session_id = request.headers.get("X-Session-ID") or "default_session"
        result = pipeline_instance.process_frame(frame, return_images=True, session_id=session_id)
        return result
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/api/predict-stream-frame")
async def predict_stream_frame(data: FrameData):
    try:
        frame = decode_base64_image(data.image)
        session_id = data.session_id or "default_session"
        # Fast inference returning only classification + bounding boxes, no intermediate base64 images
        result = pipeline_instance.process_frame(frame, return_images=False, session_id=session_id)
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

# -------------------------------------------------------------
# Phase 18: ISL Multi-Signer Data Collection Studio Endpoints
# -------------------------------------------------------------

class CommitRequest(BaseModel):
    staging_id: str
    action: str  # "accept" or "reject"
    rejection_note: Optional[str] = None

@app.get("/collect", response_class=HTMLResponse)
async def collect_page(request: Request):
    """Renders the Phase 18/34A ISL Multi-Signer Data Collection Studio."""
    from phase18_collector import SIGNER_ROLE_MAPPING
    return templates.TemplateResponse(
        request=request,
        name="collect.html",
        context={
            "signers": ALLOWED_SIGNERS,
            "conditions": CONDITION_PRESETS,
            "roles": SIGNER_ROLE_MAPPING,
            "classes": [chr(65 + i) for i in range(26)]
        }
    )

@app.get("/api/collect/progress")
async def get_collect_progress():
    """Returns the multi-signer collection progress matrix."""
    try:
        return collection_engine.get_progress()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/collect/stage")
async def stage_recording_endpoint(
    file: UploadFile = File(...),
    signer_id: str = Form(...),
    letter: str = Form(...),
    condition_id: str = Form(...),
    session_id: Optional[str] = Form(None),
    consent_confirmed: bool = Form(False),
    sampling_interval: int = Form(3),
    operator_notes: Optional[str] = Form(""),
    take_number: int = Form(1),
    actual_pitch_deg: Optional[float] = Form(None),
    actual_distance_cm: Optional[float] = Form(None),
    actual_yaw_deg: Optional[float] = Form(None),
    distance_measurement_method: Optional[str] = Form("nominal")
):
    """
    Stages an incoming video recording, extracts frames at the configurable interval,
    runs blur, brightness, duplicate, resolution, and landmark audits, and returns review previews.
    """
    try:
        content = await file.read()
        print(f"[Stage Endpoint] Received '{file.filename}', size: {len(content)} bytes, signer: {signer_id}, letter: {letter}, condition: {condition_id}, take: {take_number}, consent: {consent_confirmed}", flush=True)
        if not content:
            raise ValueError("Uploaded video payload is empty (0 bytes).")
        res = collection_engine.stage_recording(
            video_bytes=content,
            filename=file.filename or "recording.webm",
            signer_id=signer_id,
            letter=letter,
            condition_id=condition_id,
            session_id=session_id,
            consent_confirmed=consent_confirmed,
            sampling_interval=sampling_interval,
            operator_notes=operator_notes or "",
            take_number=take_number,
            actual_pitch_deg=actual_pitch_deg,
            actual_distance_cm=actual_distance_cm,
            actual_yaw_deg=actual_yaw_deg,
            distance_measurement_method=distance_measurement_method
        )
        print(f"[Stage Endpoint SUCCESS] Staged '{res.get('staging_id')}' with {res.get('usable_count')} usable frames.", flush=True)
        return res
    except ValueError as ve:
        import traceback
        traceback.print_exc()
        print(f"[Stage Endpoint ValueError] {ve}", flush=True)
        return JSONResponse(status_code=400, content={"error": str(ve)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Stage Endpoint Exception] {e}", flush=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/collect/commit")
async def commit_recording_endpoint(data: CommitRequest):
    """
    Operator action: Accepts and commits usable frames into the appropriate partition,
    or discards and permanently deletes the staged recording.
    """
    try:
        res = collection_engine.commit_recording(
            staging_id=data.staging_id,
            action=data.action,
            rejection_note=data.rejection_note
        )
        return res
    except FileNotFoundError as fnf:
        return JSONResponse(status_code=404, content={"error": str(fnf)})
    except ValueError as ve:
        return JSONResponse(status_code=400, content={"error": str(ve)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.delete("/api/collect/recording/{recording_id}")
async def delete_recording_endpoint(recording_id: str):
    """
    Voluntary participant right-to-deletion: permanently removes a recording and its frames.
    """
    try:
        res = collection_engine.delete_recording(recording_id)
        return res
    except FileNotFoundError as fnf:
        return JSONResponse(status_code=404, content={"error": str(fnf)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/collect/split-audit")
async def get_split_audit():
    """
    Verifies that all collected samples adhere strictly to partition isolation boundaries.
    """
    try:
        return collection_engine.audit_split_isolation()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/collect/summary/{signer_id}")
async def get_signer_summary_endpoint(signer_id: str):
    """
    Returns the comprehensive collection, retake, and readiness summary for a specific signer.
    """
    try:
        return collection_engine.get_signer_summary(signer_id)
    except ValueError as ve:
        return JSONResponse(status_code=400, content={"error": str(ve)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/collect/session/{signer_id}")
async def get_signer_session_endpoint(signer_id: str):
    """
    Returns session recovery metadata, active target letter, and take slot after interruption.
    """
    try:
        return collection_engine.get_signer_summary(signer_id)
    except ValueError as ve:
        return JSONResponse(status_code=400, content={"error": str(ve)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/collect/preflight")
async def get_collect_preflight(
    signer_id: str,
    condition_id: str,
    letter: str,
    take_number: int = 1,
    consent_confirmed: bool = False,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[float] = None
):
    """
    Returns session preflight readiness checklist and protocol validation.
    """
    try:
        det_res = {"width": width, "height": height} if (width and height) else None
        return collection_engine.get_preflight_checklist(
            signer_id=signer_id,
            condition_id=condition_id,
            letter=letter,
            take_number=take_number,
            consent_confirmed=consent_confirmed,
            detected_resolution=det_res,
            observed_fps=fps
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8050, reload=False)
