"""Compare two keyword-tag hypothesis wordings on DEV true positives only."""

import argparse
import asyncio
import json
from pathlib import Path

from adverse_media.decision.nli_classifier import HFNLIClassifier
from adverse_media.ingestion.dedup import deduplicate
from adverse_media.ingestion.sources import LocalDatasetSource, fetch_candidates

ROOT = Path(__file__).parents[1]
KEYWORDS = ["corruption", "fraud", "bribery", "money laundering"]
COMPANY = "Acme GmbH"
# Alternative wording under evaluation; it is intentionally not part of the cascade.
T1 = "{company} committed {keyword}."
T2 = "This article is about {keyword}."


def micro_metrics(
    scores: list[dict[str, float]],
    gold: list[set[str]],
    selector,
) -> tuple[float, float, float]:
    true_positive = false_positive = false_negative = 0
    for article_scores, gold_categories in zip(scores, gold):
        predicted = set(selector(article_scores))
        true_positive += len(predicted & gold_categories)
        false_positive += len(predicted - gold_categories)
        false_negative += len(gold_categories - predicted)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def print_metrics(scores: list[dict[str, float]], gold: list[set[str]]) -> None:
    selectors = [
        ("p>=0.30", lambda values: [k for k, p in values.items() if p >= 0.30]),
        ("p>=0.50", lambda values: [k for k, p in values.items() if p >= 0.50]),
        ("p>=0.70", lambda values: [k for k, p in values.items() if p >= 0.70]),
        ("p>=0.90", lambda values: [k for k, p in values.items() if p >= 0.90]),
        ("top-1", lambda values: [max(values, key=values.get)]),
        ("top-2", lambda values: sorted(values, key=values.get, reverse=True)[:2]),
    ]
    for delta in (0.05, 0.10, 0.20, 0.30):
        selectors.append(
            (
                f"max-delta-{delta:.2f}",
                lambda values, delta=delta: [
                    keyword for keyword, score in values.items()
                    if score >= max(values.values()) - delta
                ],
            )
        )
    print("  selection              precision  recall  f1")
    for name, selector in selectors:
        precision, recall, f1 = micro_metrics(scores, gold, selector)
        print(f"  {name:<21} {precision:.3f}      {recall:.3f}   {f1:.3f}")
    print(f"  n_articles={len(scores)} n_gold_labels={sum(len(categories) for categories in gold)}")


async def load_dev_true_positives() -> tuple[list, list[set[str]]]:
    labels = {
        label["id"]: label
        for label in json.loads((ROOT / "data" / "gold_labels.json").read_text(encoding="utf-8"))
    }
    sources = [
        LocalDatasetSource(ROOT / "data" / "articles.json", "newswire"),
        LocalDatasetSource(ROOT / "data" / "articles.json", "web_archive"),
    ]
    candidates = await fetch_candidates(sources, COMPANY, KEYWORDS, "en", 5)
    articles, _ = deduplicate(candidates)
    selected = [
        article
        for article in articles
        if labels[article.id]["split"] == "dev"
        and labels[article.id]["is_hit"]
    ]
    return selected, [set(labels[article.id]["categories"]) for article in selected]


async def run(model_name: str) -> None:
    articles, gold = await load_dev_true_positives()
    classifier = HFNLIClassifier(model_name)
    classifier.load()
    premises = [f"{article.title}. {article.snippet}" for article in articles]
    wordings = {
        "T1": T1,
        "T2": T2,
    }
    for name, template in wordings.items():
        scores: list[dict[str, float]] = []
        flat_scores = await classifier.aentailment_probabilities(
            [premise for premise in premises for _ in KEYWORDS],
            [
                template.format(company=COMPANY, keyword=keyword)
                for _ in premises
                for keyword in KEYWORDS
            ],
        )
        for offset, article in enumerate(articles):
            scores.append(
                dict(zip(KEYWORDS, flat_scores[offset * len(KEYWORDS) : (offset + 1) * len(KEYWORDS)]))
            )
        print(f"\n{name}: {template}")
        print("  id     language  gold_categories                         corruption fraud bribery money laundering")
        for article, categories, article_scores in zip(articles, gold, scores):
            print(
                f"  {article.id:<6} {article.language:<9} {','.join(sorted(categories)):<38} "
                f"{article_scores['corruption']:.3f}      {article_scores['fraud']:.3f}  "
                f"{article_scores['bribery']:.3f}   {article_scores['money laundering']:.3f}"
            )
        print_metrics(scores, gold)
    print("\nNote: with 8 articles and about 13 gold labels, differences under about 0.05 F1 are noise.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    )
    asyncio.run(run(parser.parse_args().model))


if __name__ == "__main__":
    main()
