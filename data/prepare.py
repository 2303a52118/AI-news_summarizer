from datasets import load_dataset
from transformers import BartTokenizer
from config import CFG


def load_and_tokenize():
    print("Loading CNN/DailyMail dataset...")
    raw = load_dataset(CFG.dataset_name, CFG.dataset_version)

    tokenizer = BartTokenizer.from_pretrained(CFG.model_name)

    def tokenize(batch):
        # Tokenise articles (input)
        model_inputs = tokenizer(
            batch["article"],
            max_length=CFG.max_input_len,
            padding="max_length",
            truncation=True,
        )
        # Tokenise highlights (target summaries)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["highlights"],
                max_length=CFG.max_target_len,
                padding="max_length",
                truncation=True,
            )

        # Replace padding token id with -100 so loss ignores padding
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Optionally slice for faster experimentation
    def slice_split(split, n):
        return raw[split].select(range(n)) if n > 0 else raw[split]

    train = slice_split("train",      CFG.train_samples)
    val   = slice_split("validation", CFG.val_samples)
    test  = slice_split("test",       CFG.test_samples)

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print("Tokenising...")

    cols = ["article", "highlights", "id"]
    train_tok = train.map(tokenize, batched=True, remove_columns=cols)
    val_tok   = val.map(tokenize,   batched=True, remove_columns=cols)
    test_tok  = test.map(tokenize,  batched=True, remove_columns=cols)

    train_tok.set_format("torch")
    val_tok.set_format("torch")
    test_tok.set_format("torch")

    print("Done.")
    return train_tok, val_tok, test_tok, tokenizer, test
