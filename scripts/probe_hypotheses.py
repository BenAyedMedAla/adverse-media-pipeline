"""Probe fixed NLI hypotheses on the development split only."""

import argparse
import asyncio
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch

from adverse_media.decision.entity_gate import EntityGate
from adverse_media.decision.hypotheses import join_risks
from adverse_media.decision.nli_classifier import HFNLIClassifier
from adverse_media.ingestion.dedup import deduplicate
from adverse_media.ingestion.sources import LocalDatasetSource, fetch_candidates

ROOT = Path(__file__).parents[1]
KINDS = (
    "true_positive",
    "fp_victim",
    "fp_training",
    "fp_denied_rumor",
    "fp_other_entity",
    "fp_mention_only",
    "noise",
)
# Retained here as the record of the DEV hypothesis comparison.
G1 = "{company} is involved in wrongdoing, such as {risks}."
G2 = "{company} is accused of or under investigation for {risks}."
G3 = "{company} committed {risks}."
V = "{company} is a victim of crime or was cleared of wrongdoing."
HYPOTHESES = (("G1", G1), ("G2", G2), ("G3", G3), ("V", V))


def auc(scores: list[float], hits: list[bool]) -> float:
    """Return the pairwise hit-vs-non-hit ranking score, with ties worth 0.5."""
    hit_scores = [score for score, hit in zip(scores, hits) if hit]
    non_hit_scores = [score for score, hit in zip(scores, hits) if not hit]
    if not hit_scores or not non_hit_scores:
        return float("nan")
    total = sum(
        1.0 if hit_score > non_hit_score else 0.5 if hit_score == non_hit_score else 0.0
        for hit_score in hit_scores
        for non_hit_score in non_hit_scores
    )
    return total / (len(hit_scores) * len(non_hit_scores))


def _print_stats(name: str, scores: list[float], kinds: list[str]) -> None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for score, kind in zip(scores, kinds):
        grouped[kind].append(score)
    print(f"\n{name}")
    for kind in KINDS:
        values = grouped[kind]
        print(
            f"  {kind}: count={len(values)}, min={min(values):.4f}, "
            f"median={statistics.median(values):.4f}, max={max(values):.4f}"
        )


async def _load_dev_articles() -> tuple[list, dict[str, dict]]:
    raw_labels = json.loads((ROOT / "data" / "gold_labels.json").read_text(encoding="utf-8"))
    labels = {label["id"]: label for label in raw_labels}
    sources = [
        LocalDatasetSource(ROOT / "data" / "articles.json", "newswire"),
        LocalDatasetSource(ROOT / "data" / "articles.json", "web_archive"),
    ]
    candidates = await fetch_candidates(sources, "Acme GmbH", list(RISKS), "en", 5)
    articles, _ = deduplicate(candidates)
    gate = EntityGate("Acme GmbH")
    return [article for article in articles if labels[article.id]["split"] == "dev" and gate.matches(article)], labels


RISKS = ("corruption", "fraud", "bribery", "money laundering")


async def run(model_name: str, quantize: bool) -> None:
    articles, labels = await _load_dev_articles()
    premises = [f"{article.title}. {article.snippet}" for article in articles]
    kinds = [labels[article.id]["kind"] for article in articles]
    hits = [labels[article.id]["is_hit"] for article in articles]
    classifier = HFNLIClassifier(model_name, quantize=quantize)
    classifier.load()
    risks = join_risks(list(RISKS))
    all_scores: dict[str, list[float]] = {}
    started = time.perf_counter()
    for name, template in HYPOTHESES:
        hypothesis = template.format(company="Acme GmbH", risks=risks)
        scores = classifier.entailment_probabilities(
            premises, [hypothesis] * len(premises)
        )
        all_scores[name] = scores
        _print_stats(name, scores, kinds)
        if name != "V":
            print(
                f"  AUC: {auc(scores, hits):.4f}; "
                f"n_hits={sum(hits)}, n_non_hits={len(hits) - sum(hits)}"
            )
    elapsed_ms = (time.perf_counter() - started) * 1000
    pairs = len(premises) * len(HYPOTHESES)
    print(
        f"\nmodel={model_name}; quantize={quantize}; torch={torch.__version__}; "
        f"threads={torch.get_num_threads()}; mean_ms_per_pair={elapsed_ms / pairs:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args.model, args.quantize))


if __name__ == "__main__":
    main()
