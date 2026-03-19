"""
Bangala-Hindi-English Sentiment Analysis Dashboard
Run: uvicorn main:app --reload --port 8000
Then open: http://localhost:8000
"""

import os
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL   = "xlm-roberta-base"
MAX_LENGTH   = 192
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
ADAPTER_PATH = Path("./models/bhe_lora_adapter")

LABEL2ID = {"Positive": 0, "Negative": 1, "Neutral": 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Sentiment Analysis Dashboard")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model (loaded once on first request) ─────────────────────────────────────
_tokenizer = None
_model     = None

def get_model():
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model

    print(f"Loading model on {DEVICE}...")

    if ADAPTER_PATH.exists():
        print(f"  Found adapter at {ADAPTER_PATH} — loading fine-tuned model")
        _tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_PATH))
        base = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
        )
        _model = PeftModel.from_pretrained(base, str(ADAPTER_PATH)).merge_and_unload()
    else:
        print(f"  No adapter found — running in DEMO MODE (untuned base model)")
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        _model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
        )

    _model.eval().to(DEVICE)
    print("  Model ready.")
    return _tokenizer, _model

# ── Schema ────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":        "ok",
        "device":        DEVICE,
        "adapter_found": ADAPTER_PATH.exists(),
    }

@app.post("/predict")
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty.")

    tokenizer, model = get_model()

    inputs = tokenizer(
        req.text,
        return_tensors="pt",
        max_length=MAX_LENGTH,
        truncation=True,
        padding=True,
    ).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0]

    pred = probs.argmax().item()

    return {
        "sentiment":     ID2LABEL[pred],
        "confidence":    round(probs[pred].item(), 4),
        "probabilities": {ID2LABEL[i]: round(probs[i].item(), 4) for i in range(3)},
        "demo_mode":     not ADAPTER_PATH.exists(),
    }

# ── Serve the frontend ────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def root():
    html_path = Path("index.html")
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h2>index.html not found next to main.py</h2>", status_code=404)