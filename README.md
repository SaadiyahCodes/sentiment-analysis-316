# BHE Sentiment Analysis Dashboard
**CSCI 316 — Project 2**

A web dashboard for real-time sentiment analysis of Bangla-Hindi-English (BHE) code-switched text, using a LoRA fine-tuned XLM-RoBERTa model served via FastAPI.

---

## Project Structure

```
Code-Switched Text Sentiment Analysis/
├── notebooks/  
    ├── N1_PreProcessing.ipynb
    ├── N2_Full_Fine_Tuning.ipynb
    ├── N3_LoRA_PEFT.ipynb
    └── N4_Full_Fine_Tuning_Native_PyTorch.ipynb
    └── N5_LoRA_Fine_Tuning_Native_PyTorch.ipynb
├── main.py              # FastAPI backend
├── index.html           # Frontend (http://localhost:8000)
├── requirements.txt     # Python dependencies
├── README.md
└── models/
    └── bhe_lora_adapter/
        ├── adapter_config.json
        ├── adapter_model.safetensors
        ├── tokenizer.json
        └── tokenizer_config.json
```

---

## Setup

### 1. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the server
```bash
uvicorn main:app --reload --port 8000
```

### 4. Open the dashboard
Go to **http://localhost:8000** in your browser.

---

## Model Details

| | |
|---|---|
| Base model | `xlm-roberta-base` (278M parameters) |
| Fine-tuning method | LoRA PEFT |
| Trainable parameters | < 1% of total |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| Target modules | query, key, value, dense |
| Max token length | 192 |
| Labels | Positive · Negative · Neutral |

---

## Dependencies

| Package | Purpose |
|---|---|
| `fastapi` | Web framework / API |
| `uvicorn` | ASGI server |
| `torch` | Model inference |
| `transformers` | XLM-RoBERTa tokenizer & model |
| `peft` | Load LoRA adapter weights |
| `sentencepiece` | XLM-RoBERTa tokenizer requirement |

---

## Run with Docker
```bash
docker build -t sentiment316 .
docker run -p 8000:8000 sentiment316
```

Open: http://localhost:8000

> **Windows users:** If port 8000 is unavailable (common with Docker Desktop + WSL2),
> run `docker run -p 9000:8000 sentiment316` and open http://localhost:9000 instead.