"""Small evaluation metrics shared by decision and LLM reports."""


def tag_micro_metrics(
    predicted: list[set[str]], gold: list[set[str]]
) -> tuple[float, float, float]:
    """Return set-based micro precision, recall, and F1 across articles."""
    true_positive = sum(len(predicted_set & gold_set) for predicted_set, gold_set in zip(predicted, gold))
    predicted_total = sum(len(predicted_set) for predicted_set in predicted)
    gold_total = sum(len(gold_set) for gold_set in gold)
    precision = true_positive / predicted_total if predicted_total else 0.0
    recall = true_positive / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1
