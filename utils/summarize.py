import os
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from config import CFG

_model = None
_tokenizer = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load():
    global _model, _tokenizer
    if _model is not None:
        return
    path = CFG.save_dir if os.path.exists(CFG.save_dir) else CFG.model_name
    print(f"Loading model from: {path}")
    _tokenizer = BartTokenizer.from_pretrained(path)
    _model     = BartForConditionalGeneration.from_pretrained(path).to(DEVICE)
    _model.eval()
    print("Model ready.")


def summarize(text: str,
              max_len: int   = 130,
              min_len: int   = 30,
              num_beams: int = 4) -> dict:
    _load()
    inputs = _tokenizer(
        text,
        max_length=CFG.max_input_len,
        truncation=True,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        ids = _model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_len,
            min_length=min_len,
            num_beams=num_beams,
            length_penalty=CFG.length_penalty,
            no_repeat_ngram_size=CFG.no_repeat_ngram,
            early_stopping=CFG.early_stopping,
        )

    summary = _tokenizer.decode(ids[0], skip_special_tokens=True)
    n_input  = inputs["input_ids"].shape[1]
    n_output = ids.shape[1]

    return {
        "summary":        summary,
        "input_tokens":   n_input,
        "output_tokens":  n_output,
        "compression":    round((1 - n_output / n_input) * 100, 1),
    }
