"""Evaluate the selective LLM stage on one split without API or pipeline wiring."""

import argparse
import asyncio
from urllib.parse import urlsplit

from adverse_media.config import get_settings
from adverse_media.decision.cascade import DecisionCascade, Zone
from adverse_media.decision.entity_gate import EntityGate
from adverse_media.decision.nli_classifier import HFNLIClassifier
from adverse_media.llm.openai_compat import build_backend
from adverse_media.llm.synthesizer import HitSynthesizer
from scripts.evaluate import KEYWORDS, binary_metrics, load_articles
from adverse_media.metrics import tag_micro_metrics


async def run(split: str, limit: int | None, verbose: bool = False) -> None:
    articles, labels = await load_articles(split)
    settings = get_settings()
    classifier = HFNLIClassifier(settings.nli_model_name)
    classifier.load()
    cascade = await DecisionCascade(EntityGate("Acme GmbH"), classifier, settings).run(
        articles, "Acme GmbH", KEYWORDS
    )
    records = [
        record for record in cascade.records if record.rejected_stage is None
    ]
    if limit is not None:
        records = records[:limit]
    backend = build_backend(settings)
    batch = await HitSynthesizer(backend, settings).synthesize(
        records, "Acme GmbH", KEYWORDS, "en"
    )
    if backend is not None:
        await backend.close()
    outcomes_by_id = {outcome.record.article.id: outcome for outcome in batch.outcomes}
    gold = [labels[record.article.id]["is_hit"] for record in cascade.records]
    final = [
        outcomes_by_id[record.article.id].verdict.confirmed
        if record.article.id in outcomes_by_id
        else False
        for record in cascade.records
    ]
    sent = [record.article.id in outcomes_by_id for record in cascade.records]
    print("PER ARTICLE")
    for record in cascade.records:
        outcome = outcomes_by_id.get(record.article.id)
        print(
            f"id={record.article.id} kind={labels[record.article.id]['kind']} "
            f"zone={record.zone.value} confirmed={outcome.verdict.confirmed if outcome else False} "
            f"nli_tags={record.matched_categories} "
            f"llm_categories={outcome.verdict.categories if outcome else []} "
            f"gold_categories={labels[record.article.id]['categories']} "
            f"source={outcome.source if outcome else 'not_sent'}"
        )
        if outcome is not None and outcome.source == "fallback":
            print(
                f"  error={outcome.error} status={outcome.error_status} "
                f"code={outcome.error_code} message={outcome.error_message} "
                f"raw={outcome.raw_reply}"
            )
        if verbose and outcome is not None and outcome.source == "llm":
            article = record.article
            print(f"  language={article.language} sentiment={outcome.verdict.sentiment}")
            print(f"  title={article.title}")
            print(f"  snippet={article.snippet[:300]}")
            print(f"  reason={outcome.verdict.reason[:400]}")
    print(f"FINAL precision/recall/f1={binary_metrics(final, gold)} n={len(gold)}")
    print(f"SENT_TO_LLM precision/recall/f1={binary_metrics(sent, gold)} n={len(gold)}")
    tp_records = [record for record in cascade.records if labels[record.article.id]["is_hit"]]
    tp_outcomes = [item for item in batch.outcomes if item.record in tp_records]
    tag_precision, tag_recall, tag_f1 = tag_micro_metrics(
        [set(item.verdict.categories) for item in tp_outcomes],
        [set(labels[item.record.article.id]["categories"]) for item in tp_outcomes],
    )
    gold_labels = sum(len(labels[record.article.id]["categories"]) for record in tp_records)
    print(
        f"TAG_LLM_VS_NLI precision={tag_precision:.4f} recall={tag_recall:.4f} "
        f"f1={tag_f1:.4f} n_articles={len(tp_records)} n_gold_labels={gold_labels}"
    )
    failures = {}
    for outcome in batch.outcomes:
        if outcome.error:
            error_type = outcome.error.split(":", 1)[0]
            failures[error_type] = failures.get(error_type, 0) + 1
    print(f"FAILURES_BY_TYPE n={sum(failures.values())} counts={failures}")
    print(
        f"llm_invocations={batch.llm_invocations} retries={batch.llm_retries} "
        f"fallbacks={batch.llm_fallbacks} rejected={batch.llm_rejected} "
        f"mean_latency_ms={batch.mean_llm_latency_ms:.2f}"
    )
    print(
        "FINAL_FALSE_POSITIVES",
        [record.article.id for record, is_final in zip(cascade.records, final) if is_final and not labels[record.article.id]["is_hit"]],
    )
    print(
        "DROPPED_TRUE_POSITIVES",
        [item.record.article.id for item in batch.outcomes if item.dropped and labels[item.record.article.id]["is_hit"]],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    print(
        f"provider_host={urlsplit(settings.llm_base_url).netloc} model={settings.llm_model} "
        f"json_mode={settings.llm_json_mode} key_present={'yes' if settings.llm_api_key else 'no'}"
    )
    asyncio.run(run(args.split, args.limit, args.verbose))


if __name__ == "__main__":
    main()
