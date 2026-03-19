"""
Code-Switched Sentiment Analysis Dashboard
  - Bangla-Hindi-English (BHE)  : LoRA PEFT on xlm-roberta-base
  - Arabic-English (AR)         : LoRA PEFT on xlm-roberta-base

Run: uvicorn main:app --reload --port 8000
Then open: http://localhost:8000
"""

from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL = "xlm-roberta-base"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# BHE: Positive=0, Negative=1, Neutral=2
BHE_LABEL2ID = {"Positive": 0, "Negative": 1, "Neutral": 2}
BHE_ID2LABEL = {v: k for k, v in BHE_LABEL2ID.items()}
BHE_MAX_LEN  = 192

# Arabic: negative=0, neutral=1, positive=2  (from preprocessing notebook)
AR_LABEL2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
AR_ID2LABEL = {v: k for k, v in AR_LABEL2ID.items()}
AR_MAX_LEN  = 128

MODEL_PATHS = {
    "bhe":    Path("./models/bhe_lora_adapter"),
    "arabic": Path("./models/arabic_lora_adapter"),
}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Code-Switched Sentiment Analysis")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model cache ───────────────────────────────────────────────────────────────
_tokenizers: dict = {}
_models:     dict = {}

def get_model(key: str):
    if key in _models:
        return _tokenizers[key], _models[key]

    path     = MODEL_PATHS[key]
    id2label = BHE_ID2LABEL if key == "bhe" else AR_ID2LABEL
    label2id = BHE_LABEL2ID if key == "bhe" else AR_LABEL2ID

    print(f"\nLoading [{key}] model on {DEVICE}...")

    if path.exists():
        print(f"  Found LoRA adapter — loading fine-tuned {key.upper()} model")
        tokenizer = AutoTokenizer.from_pretrained(str(path))
        base = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=3, id2label=id2label, label2id=label2id
        )
        model = PeftModel.from_pretrained(base, str(path)).merge_and_unload()
    else:
        print(f"  No adapter found — {key.upper()} running in DEMO MODE")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=3, id2label=id2label, label2id=label2id
        )

    model.eval().to(DEVICE)
    _tokenizers[key] = tokenizer
    _models[key]     = model
    _models[key + "_max_len"]  = BHE_MAX_LEN if key == "bhe" else AR_MAX_LEN
    _models[key + "_id2label"] = id2label
    print(f"  [{key}] model ready.")
    return tokenizer, model

# ── Schema ────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    model: Literal["bhe", "arabic"]
    text:  str

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":  "ok",
        "device":  DEVICE,
        "models":  {k: v.exists() for k, v in MODEL_PATHS.items()},
    }

@app.post("/predict")
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty.")

    tokenizer, model = get_model(req.model)
    max_len  = _models[req.model + "_max_len"]
    id2label = _models[req.model + "_id2label"]

    inputs = tokenizer(
        req.text,
        return_tensors="pt",
        max_length=max_len,
        truncation=True,
        padding=True,
    ).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0]

    pred = probs.argmax().item()

    return {
        "sentiment":     id2label[pred],
        "confidence":    round(probs[pred].item(), 4),
        "probabilities": {id2label[i]: round(probs[i].item(), 4) for i in range(3)},
        "demo_mode":     not MODEL_PATHS[req.model].exists(),
    }

# ── Serve the frontend ────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def root():
    html_path = Path("index.html")
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h2>index.html not found next to main.py</h2>", status_code=404)