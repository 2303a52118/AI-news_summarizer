"""
Evaluate the fine-tuned model on the full test split.
Compares: pretrained BART baseline vs fine-tuned BART.
Outputs: ROUGE scores table + sample predictions CSV.
"""
import os
import csv
import torch
import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate

from config import CFG
from data.prepare import load_and_tokenize

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
rouge  = evaluate.load("rouge")


def load_model(path):
    tok   = BartTokenizer.from_pretrained(path)
    model = BartForConditionalGeneration.from_pretrained(path).to(DEVICE)
    model.eval()
    return model, tok


def run_eval(model, tokenizer, loader, label="Model"):
    preds, refs = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=label):
            ids   = batch["input_ids"].to(DEVICE)
            mask  = batch["attention_mask"].to(DEVICE)

            gen = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=CFG.max_target_len,
                min_length=CFG.min_target_len,
                num_beams=CFG.num_beams,
                length_penalty=CFG.length_penalty,
                no_repeat_ngram_size=CFG.no_repeat_ngram,
                early_stopping=CFG.early_stopping,
            )
            decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
            labels  = tokenizer.batch_decode(
                [[l for l in lab if l != -100] for lab in batch["labels"]],
                skip_special_tokens=True,
            )
            preds.extend(decoded)
            refs.extend(labels)

    scores = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    return {k: round(v * 100, 2) for k, v in scores.items()}, preds, refs


if __name__ == "__main__":
    _, _, test_tok, _, test_raw = load_and_tokenize()
    test_loader = DataLoader(test_tok, batch_size=8)

    # ── Baseline: pretrained (no fine-tuning) ─────────────────
    print("\nEvaluating baseline (pretrained BART)...")
    base_model, base_tok = load_model(CFG.model_name)
    base_scores, base_preds, refs = run_eval(base_model, base_tok,
                                             test_loader, "Baseline")

    # ── Fine-tuned ────────────────────────────────────────────
    if os.path.exists(CFG.save_dir):
        print("\nEvaluating fine-tuned model...")
        ft_model, ft_tok = load_model(CFG.save_dir)
        ft_scores, ft_preds, _ = run_eval(ft_model, ft_tok,
                                          test_loader, "Fine-tuned")
    else:
        print("Fine-tuned model not found. Run train.py first.")
        ft_scores = ft_preds = None

    # ── Print comparison table ────────────────────────────────
    metrics = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    print(f"\n{'─'*55}")
    print(f"{'Metric':<15} {'Baseline':>12} {'Fine-tuned':>12} {'Delta':>10}")
    print(f"{'─'*55}")
    for m in metrics:
        b = base_scores[m]
        f = ft_scores[m] if ft_scores else 0
        d = f - b if ft_scores else 0
        sign = "+" if d >= 0 else ""
        print(f"{m:<15} {b:>12.2f} {f:>12.2f} {sign+str(round(d,2)):>10}")
    print(f"{'─'*55}")

    # ── Save sample predictions ───────────────────────────────
    articles = [test_raw[i]["article"][:300] for i in range(len(refs))]
    rows = []
    for i, (art, ref, bp) in enumerate(zip(articles, refs, base_preds)):
        row = {
            "article_snippet": art,
            "reference":        ref,
            "baseline_pred":    bp,
        }
        if ft_preds:
            row["finetuned_pred"] = ft_preds[i]
        rows.append(row)

    os.makedirs("outputs", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv("outputs/evaluation_results.csv", index=False)
    print("\nSample predictions saved to: outputs/evaluation_results.csv")
