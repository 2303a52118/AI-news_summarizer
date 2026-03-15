# News Summarizer — Fine-tuned BART on CNN/DailyMail

Abstractive news summarization using BART-large fine-tuned on the CNN/DailyMail dataset.
Includes ROUGE evaluation, keyword extraction, named entity detection, and a Streamlit dashboard.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Fine-tune BART
```bash
python train.py
```
Trains for 3 epochs on 10,000 CNN/DailyMail articles.
Best model saved to `models/bart-finetuned/` based on validation ROUGE-2.

To use the full dataset (~287k articles), set `train_samples = -1` in `config.py`.

**Expected training ROUGE-2:** ~21–22 (vs ~18 baseline pretrained)

### 2. Evaluate on test set
```bash
python evaluate_model.py
```
Compares pretrained baseline vs fine-tuned. Prints ROUGE table and saves `outputs/evaluation_results.csv`.

### 3. Streamlit dashboard
```bash
streamlit run app_streamlit.py
```
Three tabs:
- **Paste text** — paste any article, get summary + ROUGE vs reference
- **From URL** — fetch and summarize any news URL
- **ROUGE evaluator** — compute ROUGE between any two texts

### 4. Flask REST API
```bash
python app.py
```

```bash
# Summarize text
curl -X POST http://localhost:5000/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Your article text here..."}'

# Summarize from URL
curl -X POST http://localhost:5000/summarize-url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.thehindu.com/..."}'

# Compute ROUGE
curl -X POST http://localhost:5000/rouge \
  -H "Content-Type: application/json" \
  -d '{"prediction": "...", "reference": "..."}'
```

## Project structure

```
news_summarizer/
├── data/
│   └── prepare.py          CNN/DailyMail loader + tokenizer
├── models/
│   └── bart-finetuned/     Saved after training
├── utils/
│   ├── summarize.py        Inference engine (used by Flask + Streamlit)
│   ├── scraper.py          URL → article text extractor
│   ├── rouge_score.py      ROUGE-1/2/L computation
│   └── keywords.py         Keyword + named entity extraction
├── outputs/
│   └── evaluation_results.csv
├── config.py               Central config (model, data, training params)
├── train.py                Fine-tuning loop with validation ROUGE
├── evaluate_model.py       Baseline vs fine-tuned comparison table
├── app.py                  Flask REST API
├── app_streamlit.py        Streamlit dashboard (3 tabs)
├── requirements.txt
└── README.md
```

## Model details

| Item | Value |
|---|---|
| Base model | facebook/bart-large-cnn |
| Dataset | CNN/DailyMail 3.0.0 |
| Train samples | 10,000 (default) / 287,113 (full) |
| Epochs | 3 |
| Optimizer | AdamW lr=3e-5 |
| Scheduler | Linear warmup + decay |
| Batch size | 4 × 4 grad accum = 16 effective |
| ROUGE-2 (pretrained) | ~18 |
| ROUGE-2 (fine-tuned) | ~21–22 |

## What makes this project stand out

- **Actually fine-tunes the model** — not just an API wrapper
- **ROUGE evaluation** — quantitative benchmark, same metric used in ACL papers
- **Baseline vs fine-tuned comparison** — shows the value added by training
- **Full NLP pipeline** — summarization + keywords + NER + compression stats
- **URL scraping** — works on real news sites (The Hindu, BBC, Reuters, etc.)
