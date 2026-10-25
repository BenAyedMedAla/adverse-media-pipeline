"""Run the assessment sample request against the real app in-process."""

import argparse
import json
import sys
import textwrap
from pathlib import Path

from starlette.testclient import TestClient

from adverse_media.api import app
from adverse_media.ingestion.dedup import deduplicate
from adverse_media.models import Article
from adverse_media.schemas import AnalyzeResponse

ROOT = Path(__file__).parents[1]
PAYLOAD = {
    "company_name": "Acme GmbH",
    "keywords": ["corruption", "fraud", "bribery", "money laundering"],
    "language": "en",
}


def print_results(response: AnalyzeResponse) -> None:
    print("\nRESULTS")
    if not response.results:
        print("  none")
        return
    for result in response.results:
        print(
            f"  score={result.relevancy_score:.4f} sentiment={result.sentiment} "
            f"categories={result.matched_categories} published_at={result.published_at}"
        )
        print(f"  title={result.title}")
        print(textwrap.fill(f"  reason={result.reason}", width=100, subsequent_indent="    "))


def print_metrics(response: AnalyzeResponse) -> None:
    metrics = response.metrics
    contract = {
        "total_candidates_fetched": metrics.total_candidates_fetched,
        "filtered_out_by_decision_models": metrics.filtered_out_by_decision_models,
        "decision_model_passes": metrics.decision_model_passes,
        "llm_invocations": metrics.llm_invocations,
        "total_latency_ms": metrics.total_latency_ms,
    }
    print("\nMETRICS")
    print(json.dumps(contract, ensure_ascii=False, indent=2))
    print(json.dumps(metrics.breakdown.model_dump(), ensure_ascii=False, indent=2))


def check_invariants(response: AnalyzeResponse) -> bool:
    metrics = response.metrics
    breakdown = metrics.breakdown
    sent_to_llm = metrics.decision_model_passes - metrics.filtered_out_by_decision_models
    checks = {
        "fetched = duplicates_removed + decision_model_passes": (
            metrics.total_candidates_fetched
            == breakdown.duplicates_removed + metrics.decision_model_passes
        ),
        "passes = filtered_out + sent_to_llm": (
            metrics.decision_model_passes
            == metrics.filtered_out_by_decision_models + sent_to_llm
        ),
        "results = sent_to_llm - llm_rejected": (
            len(response.results) == sent_to_llm - breakdown.llm_rejected
        ),
    }
    print("\nCHECKS")
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'} {name}")
    if breakdown.llm_fallbacks or any(
        "NOT verified by an LLM" in result.reason for result in response.results
    ):
        print("  WARNING unverified fallback result(s) are present")
    return all(checks.values())


def print_accuracy(response: AnalyzeResponse) -> None:
    labels = {
        item["id"]: item
        for item in json.loads((ROOT / "data" / "gold_labels.json").read_text(encoding="utf-8"))
    }
    articles = [
        Article.model_validate(item)
        for item in json.loads((ROOT / "data" / "articles.json").read_text(encoding="utf-8"))
    ]
    deduplicated, _ = deduplicate(articles)
    expected_hits = {
        article.source_url for article in deduplicated if labels[article.id]["is_hit"]
    }
    returned_urls = {result.source_url for result in response.results}
    found = expected_hits & returned_urls
    false_positive_titles = [
        result.title for result in response.results if result.source_url not in expected_hits
    ]
    missed_titles = [
        article.title
        for article in deduplicated
        if article.source_url in expected_hits and article.source_url not in returned_urls
    ]
    print("\nACCURACY vs GOLD")
    print(f"  true positives found: {len(found)} / {len(expected_hits)}")
    print(f"  false positives returned (n={len(false_positive_titles)}): {false_positive_titles}")
    print(f"  missed true positives (n={len(missed_titles)}): {missed_titles}")
    print("  The dev split was used for tuning; this is not a held-out number.")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()

    print(
        "Note: loading the model takes a while the first time; on the Groq free tier "
        "(8000 tokens/min) this job takes about a minute."
    )
    print("\nREQUEST")
    print(json.dumps(PAYLOAD, ensure_ascii=False, indent=2))

    try:
        with TestClient(app) as client:
            response = client.post("/analyze", json=PAYLOAD)
    except Exception:
        print(
            "Startup failed. Copy .env.example to .env and set LLM_API_KEY, "
            "then run the demo again."
        )
        return 1

    if response.status_code != 200:
        print(f"HTTP request failed with status {response.status_code}.")
        return 1
    try:
        result = AnalyzeResponse.model_validate(response.json())
    except Exception:
        print("The server response did not validate against the API contract.")
        return 1

    metrics = result.metrics
    sent_to_llm = metrics.decision_model_passes - metrics.filtered_out_by_decision_models
    print("\nFUNNEL")
    print(f"  fetched: {metrics.total_candidates_fetched}")
    print(f"  duplicates removed: {metrics.breakdown.duplicates_removed}")
    print(f"  entered decision layer: {metrics.decision_model_passes}")
    print(
        "  filtered out: "
        f"{metrics.filtered_out_by_decision_models} "
        f"(entity={metrics.breakdown.entity_rejected}, gate={metrics.breakdown.gate_rejected})"
    )
    print(f"  sent to LLM: {sent_to_llm}")
    print(f"  LLM rejected: {metrics.breakdown.llm_rejected}")
    print(f"  final results: {len(result.results)}")
    print_results(result)
    print_metrics(result)
    invariants_ok = check_invariants(result)
    print_accuracy(result)

    if args.raw:
        print("\nRAW RESPONSE")
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    return 0 if invariants_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
