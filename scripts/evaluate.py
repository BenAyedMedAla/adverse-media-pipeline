"""Tune and evaluate the decision cascade without any LLM or API wiring."""

import argparse
import asyncio
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from adverse_media.config import Settings
from adverse_media.decision.cascade import DecisionCascade, Zone, select_tag_categories
from adverse_media.decision.entity_gate import EntityGate
from adverse_media.decision.hypotheses import (
    GATE_HYPOTHESIS,
    TAG_HYPOTHESIS,
    VICTIM_HYPOTHESIS,
    join_risks,
)
from adverse_media.decision.nli_classifier import HFNLIClassifier
from adverse_media.ingestion.dedup import deduplicate
from adverse_media.ingestion.sources import LocalDatasetSource, fetch_candidates
from adverse_media.metrics import tag_micro_metrics

ROOT = Path(__file__).parents[1]
KEYWORDS = ["corruption", "fraud", "bribery", "money laundering"]
KINDS = (
    "true_positive",
    "fp_victim",
    "fp_training",
    "fp_denied_rumor",
    "fp_other_entity",
    "fp_mention_only",
    "noise",
)


def f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def binary_metrics(predicted: list[bool], actual: list[bool]) -> tuple[float, float, float]:
    tp = sum(p and a for p, a in zip(predicted, actual))
    fp = sum(p and not a for p, a in zip(predicted, actual))
    fn = sum(not p and a for p, a in zip(predicted, actual))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall, f1(precision, recall)


class CachedClassifier:
    """Cache pairwise model scores so tuning and reporting do not rerun the model."""

    def __init__(self, classifier: HFNLIClassifier) -> None:
        self.classifier = classifier
        self.cache: dict[tuple[str, str], float] = {}

    async def aentailment_probabilities(
        self, premises: list[str], hypotheses: list[str]
    ) -> list[float]:
        missing = [
            (premise, hypothesis)
            for premise, hypothesis in zip(premises, hypotheses)
            if (premise, hypothesis) not in self.cache
        ]
        if missing:
            scores = self.classifier.entailment_probabilities(
                [pair[0] for pair in missing], [pair[1] for pair in missing]
            )
            self.cache.update(zip(missing, scores))
        return [self.cache[(premise, hypothesis)] for premise, hypothesis in zip(premises, hypotheses)]


async def load_articles(split: str) -> tuple[list, dict[str, dict]]:
    labels = {
        label["id"]: label
        for label in json.loads((ROOT / "data" / "gold_labels.json").read_text(encoding="utf-8"))
    }
    sources = [
        LocalDatasetSource(ROOT / "data" / "articles.json", "newswire"),
        LocalDatasetSource(ROOT / "data" / "articles.json", "web_archive"),
    ]
    candidates = await fetch_candidates(sources, "Acme GmbH", KEYWORDS, "en", 5)
    articles, _ = deduplicate(candidates)
    selected = [
        article for article in articles if split == "all" or labels[article.id]["split"] == split
    ]
    return selected, labels


async def collect_scores(
    articles: list, labels: dict[str, dict], classifier: CachedClassifier
) -> tuple[list[dict], dict[str, dict[str, float]], dict[str, float], dict[str, float]]:
    gate = EntityGate("Acme GmbH")
    survivors = [article for article in articles if gate.matches(article)]
    premises = [f"{article.title}. {article.snippet}" for article in survivors]
    risks = join_risks(KEYWORDS)
    gate_scores = await classifier.aentailment_probabilities(
        premises, [GATE_HYPOTHESIS.format(company="Acme GmbH", risks=risks)] * len(survivors)
    )
    non_rejected_scores = [
        (article, score) for article, score in zip(survivors, gate_scores) if score >= 0.50
    ]
    non_rejected = [article for article, _ in non_rejected_scores]
    non_rejected_premises = [f"{article.title}. {article.snippet}" for article in non_rejected]
    victim_scores = await classifier.aentailment_probabilities(
        non_rejected_premises,
        [VICTIM_HYPOTHESIS.format(company="Acme GmbH")] * len(non_rejected),
    )
    tag_scores: dict[str, dict[str, float]] = defaultdict(dict)
    tag_articles = non_rejected
    tag_pairs = [
        (article, keyword)
        for article in tag_articles
        for keyword in KEYWORDS
    ]
    tag_values = await classifier.aentailment_probabilities(
        [f"{article.title}. {article.snippet}" for article, _ in tag_pairs],
        [
            TAG_HYPOTHESIS.format(keyword=keyword)
            for _, keyword in tag_pairs
        ],
    )
    for (article, keyword), score in zip(tag_pairs, tag_values):
        tag_scores[article.id][keyword] = score
    rows = [
        {
            "article": article,
            "label": labels[article.id],
            "gate": gate_score,
            "victim": victim_score,
        }
        for (article, gate_score), victim_score in zip(non_rejected_scores, victim_scores)
    ]
    rows.extend(
        {
            "article": article,
            "label": labels[article.id],
            "gate": gate_score,
            "victim": None,
        }
        for article, gate_score in zip(survivors, gate_scores)
        if gate_score < 0.50
    )
    rows.extend(
        {
            "article": article,
            "label": labels[article.id],
            "gate": None,
            "victim": None,
        }
        for article in articles
        if article not in survivors
    )
    return rows, tag_scores, {row["article"].id: row["gate"] for row in rows}, {
        row["article"].id: row["victim"] for row in rows
    }


def print_tuning(rows: list[dict], tag_scores: dict[str, dict[str, float]]) -> tuple[float, float, float]:
    print("\nDEV tuning: reject threshold")
    for threshold in [x / 100 for x in range(30, 66, 5)]:
        rejected = [row for row in rows if row["gate"] is not None and row["gate"] < threshold]
        print(
            f"  threshold={threshold:.2f} true_positives_rejected="
            f"{sum(row['label']['is_hit'] for row in rejected)} "
            f"non_hits_rejected={sum(not row['label']['is_hit'] for row in rejected)} "
            f"passed={len(rows) - len(rejected)}"
        )
    non_victim_non_hits = [
        row["gate"]
        for row in rows
        if row["gate"] is not None and not row["label"]["is_hit"] and row["label"]["kind"] != "fp_victim"
    ]
    true_positive_scores = [row["gate"] for row in rows if row["label"]["is_hit"] and row["gate"] is not None]
    low_edge = max(non_victim_non_hits)
    high_edge = min(true_positive_scores)
    print(f"  gap_edges highest_non_victim_non_hit={low_edge:.4f} lowest_true_positive={high_edge:.4f}")
    print(
        "  selection: recall-first: the lowest table value strictly above the highest "
        "non-victim non-hit score and below the lowest true-positive score."
    )

    print("\nDEV tuning: victim veto threshold")
    for threshold in [x / 100 for x in range(15, 46, 5)]:
        eligible = [row for row in rows if row["gate"] is not None and row["gate"] >= 0.50]
        edge = [row for row in eligible if row["victim"] is not None and row["victim"] >= threshold]
        print(
            f"  threshold={threshold:.2f} true_positives_edge="
            f"{sum(row['label']['is_hit'] for row in edge)} "
            f"non_hits_edge={sum(not row['label']['is_hit'] for row in edge)} "
            f"eligible={len(eligible)}"
        )
    tp_v = [row["victim"] for row in rows if row["label"]["is_hit"] and row["victim"] is not None]
    victim_non_hits = [
        row["victim"] for row in rows
        if not row["label"]["is_hit"] and row["label"]["kind"] == "fp_victim" and row["victim"] is not None
    ]
    print(
        f"  gap_edges highest_true_positive_victim={max(tp_v):.4f} "
        f"lowest_victim_non_hit={min(victim_non_hits):.4f}"
    )
    print("  selection: candidate inside the victim-score gap; victim flags never reject.")

    tag_predictions = {
        row["article"].id: select_tag_categories(
            tag_scores[row["article"].id], KEYWORDS, 0.50, 2
        )
        for row in rows
        if row["label"]["is_hit"]
    }
    precision, recall, tag_f1 = tag_micro_metrics(
        [set(tag_predictions[row["article"].id]) for row in rows if row["label"]["is_hit"]],
        [set(row["label"]["categories"]) for row in rows if row["label"]["is_hit"]],
    )
    print("\nDEV tag result: final rule tag_threshold=0.50, tag_max_labels=2")
    print(
        f"  micro_precision={precision:.4f} micro_recall={recall:.4f} "
        f"micro_f1={tag_f1:.4f} "
        f"n_articles={sum(row['label']['is_hit'] for row in rows)} "
        f"n_gold_labels={sum(len(row['label']['categories']) for row in rows if row['label']['is_hit'])}"
    )
    for row in rows:
        if row["label"]["is_hit"]:
            print(
                f"  id={row['article'].id} language={row['article'].language} "
                f"gold={row['label']['categories']} predicted={tag_predictions[row['article'].id]}"
            )
    eligible_reject_values = [
        threshold
        for threshold in [x / 100 for x in range(30, 66, 5)]
        if threshold > low_edge and threshold < high_edge
    ]
    chosen_reject = min(eligible_reject_values) if eligible_reject_values else 0.50
    print(
        f"  chosen thresholds: gate_reject={chosen_reject:.2f} "
        f"victim_veto=0.30 tag=0.50 (manual default; diagnostic table only)"
    )
    return chosen_reject, 0.30, 0.50


def report(result, labels: dict[str, dict], articles: list) -> None:
    print("\nFINAL CASCADE REPORT")
    for kind in KINDS:
        subset = [record for record in result.records if labels[record.article.id]["kind"] == kind]
        print(
            f"  {kind}: n={len(subset)} entity-rejected="
            f"{sum(r.rejected_stage == 'entity' for r in subset)} "
            f"noise={sum(r.rejected_stage == 'gate' for r in subset)} "
            f"edge={sum(r.zone is Zone.EDGE and r.rejected_stage is None for r in subset)} "
            f"hit={sum(r.zone is Zone.HIT and r.rejected_stage is None for r in subset)}"
        )
    actual = [labels[r.article.id]["is_hit"] for r in result.records]
    not_rejected = [r.rejected_stage is None for r in result.records]
    hit_only = [r.zone is Zone.HIT and r.rejected_stage is None for r in result.records]
    print(f"  not_rejected: precision/recall/f1={binary_metrics(not_rejected, actual)} n={len(actual)}")
    print(f"  hit_only: precision/recall/f1={binary_metrics(hit_only, actual)} n={len(actual)}")
    tp = fp = fn = 0
    for record in result.records:
        if not labels[record.article.id]["is_hit"]:
            continue
        gold = set(labels[record.article.id]["categories"])
        predicted = set(record.matched_categories)
        tp += len(gold & predicted)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
    tag_precision = tp / (tp + fp) if tp + fp else 0.0
    tag_recall = tp / (tp + fn) if tp + fn else 0.0
    print(
        f"  tag_micro_precision={tag_precision:.4f} tag_micro_recall={tag_recall:.4f} "
        f"tag_micro_f1={f1(tag_precision, tag_recall):.4f} "
        f"n_true_positives={sum(actual)}"
    )
    baseline = [
        any(keyword.casefold() in f"{r.article.title} {r.article.snippet}".casefold() for keyword in KEYWORDS)
        for r in result.records
    ]
    print(f"  keyword_baseline: precision/recall/f1={binary_metrics(baseline, actual)} n={len(actual)}")
    print(
        f"  llm_candidates={result.n_edge + result.n_hit} n_deduplicated={len(articles)} "
        f"share={(result.n_edge + result.n_hit) / len(articles):.4f} "
        f"floor_gold_true_positives={sum(actual)}"
    )
    print(
        f"  model_passes={result.model_passes} n={len(result.records)} "
        f"stage_ms={result.stage_ms} total_decision_layer_ms={result.stage_ms['total']:.2f} "
        f"ms_per_article={result.stage_ms['total'] / len(articles):.2f}"
    )
    for record in result.records:
        predicted = record.rejected_stage is None
        if predicted != labels[record.article.id]["is_hit"]:
            print(
                f"  misclassified id={record.article.id} kind={labels[record.article.id]['kind']} "
                f"language={record.article.language} p_gate={record.p_gate} p_victim={record.p_victim}"
            )
    print("  note: n is reported for every metric; the test split is small.")


async def run(split: str) -> None:
    articles, labels = await load_articles(split)
    model = HFNLIClassifier("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
    model.load()
    cached = CachedClassifier(model)
    rows, tag_scores, _, _ = await collect_scores(articles, labels, cached)
    if split == "dev":
        gate_threshold, victim_threshold, tag_threshold = print_tuning(rows, tag_scores)
        settings = Settings(
            gate_reject_threshold=gate_threshold,
            victim_veto_threshold=victim_threshold,
            tag_threshold=tag_threshold,
        )
    else:
        settings = Settings()
    # Warm up one real pair; its cost is excluded from the final cascade timing.
    model.entailment_probabilities(["Warm-up premise."], ["Warm-up hypothesis."])
    started = time.perf_counter()
    result = await DecisionCascade(EntityGate("Acme GmbH"), model, settings).run(
        articles, "Acme GmbH", KEYWORDS
    )
    result.stage_ms["total"] = (time.perf_counter() - started) * 1000
    report(result, labels, articles)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test", "all"), required=True)
    asyncio.run(run(parser.parse_args().split))


if __name__ == "__main__":
    main()
