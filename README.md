# Code-Switched Text Sentiment Analysis Dashboard
**CSCI 316 — Project 2**

A web dashboard for real-time sentiment analysis of code-switched text across two language pairs, using LoRA fine-tuned XLM-RoBERTa models served via FastAPI.

- **Bangla-Hindi-English (BHE)** — LoRA PEFT on xlm-roberta-base
- **Arabic-English (AR)** — LoRA PEFT on xlm-roberta-base

---

## Project Structure

```
Code-Switched Text Sentiment Analysis/
├── notebooks/
│   ├── Arabic-English/
│   │   ├── ARA1_datapreprocessing.ipynb
│   │   ├── ARA2_FineTuning.ipynb
│   │   ├── ARA3_PEFT.ipynb
│   │   └── ARA4_Framework_Pytorch.ipynb
│   └── Bangla-Hindi-English/
│       ├── N1_PreProcessing.ipynb
│       ├── N2_Full_Fine_Tuning.ipynb
│       ├── N3_LoRA_PEFT.ipynb
│       ├── N4_Full_Fine_Tuning_Native_PyTorch.ipynb
│       └── N5_LoRA_Fine_Tuning_Native_PyTorch.ipynb
├── main.py              # FastAPI backend
├── index.html           # Frontend (served at http://localhost:8000)
├── requirements.txt     # Python dependencies
├── README.md
└── models/
    ├── arabic_lora_adapter/
    │   ├── adapter_config.json
    │   ├── adapter_model.safetensors
    │   ├── tokenizer.json
    │   └── tokenizer_config.json
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

## Usage

1. Select a language pair tab — **Bangla · Hindi · English** or **Arabic · English**
2. Type or paste a code-switched sentence, or click an example chip
3. Click **Analyse Sentiment** (or press **Ctrl+Enter**)
4. The model returns **Positive**, **Negative**, or **Neutral** with a confidence score and probability breakdown

---

## Model Details

| | Bangla-Hindi-English | Arabic-English |
|---|---|---|
| Base model | `xlm-roberta-base` | `xlm-roberta-base` |
| Fine-tuning | LoRA PEFT | LoRA PEFT |
| LoRA rank | 8 | 8 |
| LoRA alpha | 16 | 16 |
| Target modules | query, key, value, dense | query, value |
| Max token length | 192 | 128 |
| Trainable params | < 1% of total | < 1% of total |
| Labels | Positive · Negative · Neutral | Positive · Negative · Neutral |
| Label mapping | Positive=0, Negative=1, Neutral=2 | Negative=0, Neutral=1, Positive=2 |

---

## API Endpoints
Interactive API docs: **http://localhost:8000/docs**

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
