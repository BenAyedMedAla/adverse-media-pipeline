# Architecture

## 1. Flow

```text
sources
  -> deduplication
  -> deterministic entity gate
  -> NLI relevancy gate
  -> victim/cleared-party flag
  -> NLI keyword tags
  -> LLM verification for HIT and EDGE only
  -> Pydantic validation
  -> response and telemetry
```

The decision-first order keeps obvious noise away from the paid, rate-limited stage.

## 2. Decision layer

The decision model is `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`,
a zero-shot NLI model run on CPU in float32. Each pair uses a two-way softmax over
entailment and contradiction. This fits the English, German, and French data, uses
no generated tokens, and makes one auditable score per article/hypothesis pair.

| Rejected option | Reason |
|---|---|
| Dynamic int8 mDeBERTa | Development measurement, script not kept: AUC 0.49–0.56; separation collapsed. |
| MiniLM multilingual NLI | Development measurement, script not kept: about 10x faster but AUC 0.77. |
| Cross-encoder reranker | Relevance scoring does not express risk semantics. |
| GLiClass/GLiNER | No confirmed German checkpoint was available at the time. |
| Embeddings plus classifier/SetFit | More labeled examples are needed than this dataset provides. |
| Keyword rules | Kept only as a baseline; they cannot distinguish context. |
| LLM for every article | It spends tokens on noise and creates avoidable rate-limit exposure. |

The gate, victim, and tag hypotheses are templates built from the request. G3,
`{company} committed {risks}`, was selected in `scripts/probe_hypotheses.py`.
The wording comparison was a small DEV measurement, so differences were within noise.
The unquantized AUC of about 0.91 and quantized value of about 0.56 are also recorded
in the classifier source comment; 110–118 ms/pair is a development measurement,
script not kept.

| Setting | Value | Meaning |
|---|---:|---|
| Gate reject | 0.50 | NOISE below this score; DEV gap was 0.4649–0.6654. |
| Victim veto | 0.30 | EDGE at or above this score; DEV gap was 0.233–0.344. |
| Tag threshold | 0.50 | Keep tags at or above this score. |
| Tag maximum | 2 | If none pass, keep the highest keyword. |

The victim gap rests on only two articles. The veto only flags and never rejects:
this is a recall-first choice. NOISE is rejected, EDGE goes to the LLM with an
adjudication note, and HIT goes to the LLM.

The entity gate normalizes the company name and matches whole tokens; it is a
cheap deterministic filter. Deduplication uses canonical URL first, then
`rapidfuzz` `token_sort_ratio >= 85`. The measured margin is 63.1 for the most
similar distinct stories versus 91.3 for the least similar duplicate, from
`adverse_media/ingestion/dedup.py`. `token_set_ratio` can collapse a title into
a longer word-subset match, and WRatio would have merged two distinct stories
(a001 and a029).

NLI scores are softmax probabilities, not formally calibrated probabilities.
The thresholds come from DEV score gaps; temperature scaling would need more labels.

Sentiment is produced by the LLM in the same call as confirmation, categories,
and reason. A separate decision-model sentiment classifier was deliberately not
built: every confirmed adverse hit is negative by construction, and the fallback
uses fixed `negative`.

## 3. Selective LLM

The configured provider is `openai/gpt-oss-20b` on Groq through a plain
OpenAI-compatible `httpx` backend, not a vendor SDK. It is swappable with
`LLM_BASE_URL`, `LLM_MODEL`, and `LLM_API_KEY`; the free key did not offer
`llama-3.1-8b-instant`. Temperature is zero and JSON mode is requested.
Ollama and vLLM are configuration-compatible, but were not benchmarked.

The prompt gives the exact company, requested categories, language, date, title,
snippet, NLI score, and candidate tags. It asks whether this exact company was
accused, investigated, sanctioned, or convicted, and gives EDGE cases an explicit
victim/cleared-party warning. Pydantic validation restricts categories to the
requested set and requires a category when `confirmed=true`.

Validation or truncation gets one retry. A 429 uses a cumulative 75-second wait
budget, with `Retry-After` or 5 seconds per wait, clamped to 1–30 seconds.
After that, the deterministic fallback is marked `NOT verified by an LLM`.
This favors recall over precision when the provider is unavailable.

Calls are protected by a semaphore. The application default is 4; the free-tier
demo profile sets it to 1 because Groq allows about 8000 tokens/minute.

## 4. Telemetry

| Contract field | Definition |
|---|---|
| `total_candidates_fetched` | Articles returned before deduplication. |
| `filtered_out_by_decision_models` | Entity and gate rejections. |
| `decision_model_passes` | Articles entering the decision layer after deduplication. |
| `llm_invocations` | Every HTTP request, including retries. |
| `total_latency_ms` | Wall-clock time for the complete job. |

The breakdown adds duplicate, entity, gate, EDGE, clear HIT, model-pair, retry,
LLM-rejected, fallback, and per-stage millisecond counts.

The checks are: fetched = duplicates removed + decision passes; passes = filtered
out + sent to LLM; and, without fallbacks, results = sent to LLM - LLM rejected.

## 5. Results

The recorded DEV decision run has n=21 deduplicated articles and n=8 true positives:
not-rejected precision/recall/F1 = 0.8000/1.0000/0.8889, and HIT-only
precision/recall/F1 = 1.0000/1.0000/1.0000. Tag micro F1 is 0.7857 on n=8
articles and 13 gold labels. These DEV numbers are partly in-sample.

The recorded TEST decision run has n=13: not-rejected precision/recall/F1 =
0.6667/1.0000/0.8000, HIT-only precision/recall/F1 = 1.0000/0.7500/0.8571,
and tag micro F1 = 0.5333 on n=4 true positives and 7 gold labels.
HIT-only recall is 0.75 because one true positive is in EDGE and is retained by
the flag-only veto, which is a recall-first choice.

The recorded TEST LLM run has n=13: final precision/recall/F1 =
1.0000/1.0000/1.0000; 6 articles were sent, 2 were rejected, and there were
0 retries or fallbacks. It therefore did not hit the rate limit; retry behavior
is shown by the full demo run, not this evaluation.

The latest clean demo run, recorded in the final demo run, had n=40:
40 fetched, 6 duplicates, 34 decision entries, 18 filtered (5 entity, 13 gate),
16 LLM candidates, 4 LLM rejections, 12 results, 12/12 true positives,
0 false positives, 0 fallbacks, 25 invocations, 9 retries, and 37.6 seconds.
Its stage times were gate 2.7 s, victim 1.5 s, tags 5.4 s, and LLM 27.9 s,
mostly rate-limit waiting. This is not the 892-second stalled run.
The 12/12 result is a one-sided 95% recall lower bound of about 78%.

## 6. Latency and cost horizon

Groq lists `gpt-oss-20b` at $0.075 per 1M input tokens and $0.30 per 1M output
tokens; prices were checked October 2026. See the [Groq model page](https://console.groq.com/docs/model/openai/gpt-oss-20b).
Rate-limit messages showed 954–1048 tokens per request, so 1,000 tokens is used.
That bounds one call from $0.000075 (all input priced) to $0.0003 (all output priced).

| Workload for 1,000,000 articles | Calls | Estimated LLM cost |
|---|---:|---:|
| Pure LLM baseline, one call/article | 1,000,000 | $75–$300 |
| Hybrid, 40% reaching LLM (16/40 sample ratio) | 400,000 | $30–$120 |
| Hybrid, 5% reaching LLM (illustrative assumption) | 50,000 | $3.75–$15 |

The pure-LLM baseline was not implemented or measured; it is a lower bound because
a real agent would also generate tags, sentiment, and scoring. As a latency
estimate, `llm_test.txt` reports 2,401.55 ms/call: sequential latency is the
number of articles times 2.40155 seconds, and parallel calls divide it by concurrency.
In the clean demo,
gate+victim+tags took 9.6 seconds over 34 articles: about 0.282 seconds/article,
or about 78.3 CPU-hours per million articles. CPU has no dollar price assigned.
On this CPU the decision layer is not faster per article than a fast hosted LLM
called in parallel; its advantage is fewer calls, auditability, and less rate-limit
exposure. MiniLM for tags, shorter prompts, a paid tier, and GPU/ONNX are possible
optimizations, but were not implemented or measured.

## 7. Limitations

- The evaluation set has 40 articles; DEV tuning is partly in-sample.
- The victim threshold rests on only two DEV articles.
- The entity gate cannot distinguish `Acme Corp` from `Acme GmbH`; the LLM can.
- Corruption and bribery overlap lowers tag accuracy.
- NLI tags are candidates; the LLM validates final categories, and tag accuracy is the weakest decision-layer part (TEST tag micro F1 0.5333 on n=4 true positives).
- Free-tier rate limits make the demo slower.
- Fallbacks may return unverified articles, explicitly marked.
- There is no live-news connector; the local dataset is deliberate.
