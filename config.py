from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # ── Model ─────────────────────────────────────────────────
    model_name: str = "facebook/bart-large-cnn"   # pretrained checkpoint
    save_dir: str   = "models/bart-finetuned"     # where to save fine-tuned

    # ── Data ──────────────────────────────────────────────────
    dataset_name: str    = "cnn_dailymail"
    dataset_version: str = "3.0.0"
    train_samples: int   = 10000   # set to -1 for full dataset (~287k)
    val_samples: int     = 1000
    test_samples: int    = 500

    # ── Tokenisation ──────────────────────────────────────────
    max_input_len: int   = 1024
    max_target_len: int  = 128
    min_target_len: int  = 30

    # ── Training ──────────────────────────────────────────────
    epochs: int          = 3
    batch_size: int      = 4       # increase to 8–16 if you have a GPU
    grad_accum: int      = 4       # effective batch = batch_size × grad_accum
    learning_rate: float = 3e-5
    warmup_steps: int    = 200
    weight_decay: float  = 0.01
    fp16: bool           = True    # set False on CPU or MPS

    # ── Generation ────────────────────────────────────────────
    num_beams: int       = 4
    length_penalty: float = 2.0
    no_repeat_ngram: int  = 3
    early_stopping: bool  = True


CFG = Config()
