from rouge_score import rouge_scorer


_scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"], use_stemmer=True
)


def score(prediction: str, reference: str) -> dict:
    """
    Returns ROUGE-1, ROUGE-2, ROUGE-L F1 scores (0–100).
    """
    scores = _scorer.score(reference, prediction)
    return {
        "rouge1": round(scores["rouge1"].fmeasure * 100, 2),
        "rouge2": round(scores["rouge2"].fmeasure * 100, 2),
        "rougeL": round(scores["rougeL"].fmeasure * 100, 2),
    }


def interpret(rouge2: float) -> str:
    if rouge2 >= 20:
        return "Excellent"
    elif rouge2 >= 14:
        return "Good"
    elif rouge2 >= 8:
        return "Fair"
    else:
        return "Poor"
