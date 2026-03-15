import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
import evaluate

from config import CFG
from data.prepare import load_and_tokenize

os.makedirs(CFG.save_dir, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Data ──────────────────────────────────────────────────────
train_tok, val_tok, _, tokenizer, val_raw = load_and_tokenize()

train_loader = DataLoader(train_tok, batch_size=CFG.batch_size, shuffle=True)
val_loader   = DataLoader(val_tok,   batch_size=CFG.batch_size)

# ── Model ─────────────────────────────────────────────────────
model = BartForConditionalGeneration.from_pretrained(CFG.model_name).to(DEVICE)

# ── Optimiser + Scheduler ─────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=CFG.learning_rate,
                  weight_decay=CFG.weight_decay)

total_steps = (len(train_loader) // CFG.grad_accum) * CFG.epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=CFG.warmup_steps,
    num_training_steps=total_steps,
)

scaler = torch.cuda.amp.GradScaler(enabled=CFG.fp16 and DEVICE == "cuda")
rouge  = evaluate.load("rouge")


def generate_summaries(loader, n=100):
    model.eval()
    preds, refs = [], []
    count = 0
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=CFG.max_target_len,
                min_length=CFG.min_target_len,
                num_beams=CFG.num_beams,
                length_penalty=CFG.length_penalty,
                no_repeat_ngram_size=CFG.no_repeat_ngram,
                early_stopping=CFG.early_stopping,
            )
            decoded_preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(
                [[l for l in lab if l != -100] for lab in batch["labels"]],
                skip_special_tokens=True,
            )
            preds.extend(decoded_preds)
            refs.extend(decoded_labels)
            count += len(decoded_preds)
            if count >= n:
                break

    scores = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    return {k: round(v * 100, 2) for k, v in scores.items()}


def train_epoch(epoch):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                desc=f"Epoch {epoch}")

    for step, batch in pbar:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        with torch.cuda.amp.autocast(enabled=CFG.fp16 and DEVICE == "cuda"):
            out  = model(input_ids=input_ids,
                         attention_mask=attention_mask,
                         labels=labels)
            loss = out.loss / CFG.grad_accum

        scaler.scale(loss).backward()
        total_loss += loss.item() * CFG.grad_accum

        if (step + 1) % CFG.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        pbar.set_postfix(loss=f"{total_loss/(step+1):.4f}")

    return total_loss / len(train_loader)


# ── Training loop ─────────────────────────────────────────────
best_rouge = 0.0

for epoch in range(1, CFG.epochs + 1):
    print(f"\n{'═'*55}")
    print(f"  Epoch {epoch}/{CFG.epochs}")
    print(f"{'═'*55}")

    avg_loss = train_epoch(epoch)

    print(f"\nValidation ROUGE (on 100 samples)...")
    scores = generate_summaries(val_loader, n=100)

    print(f"  Loss:     {avg_loss:.4f}")
    print(f"  ROUGE-1:  {scores['rouge1']:.2f}")
    print(f"  ROUGE-2:  {scores['rouge2']:.2f}")
    print(f"  ROUGE-L:  {scores['rougeL']:.2f}")
    print(f"  ROUGE-Lsum:{scores['rougeLsum']:.2f}")

    if scores["rouge2"] > best_rouge:
        best_rouge = scores["rouge2"]
        model.save_pretrained(CFG.save_dir)
        tokenizer.save_pretrained(CFG.save_dir)
        print(f"  ✓ Saved best model (ROUGE-2={best_rouge:.2f})")

print(f"\nTraining complete. Best ROUGE-2: {best_rouge:.2f}")
print(f"Model saved to: {CFG.save_dir}")
