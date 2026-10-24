"""Fixed NLI hypotheses used by the decision-layer probe."""


def join_risks(keywords: list[str]) -> str:
    """Join risk keywords as a readable comma-separated phrase."""
    if not keywords:
        return ""
    if len(keywords) == 1:
        return keywords[0]
    if len(keywords) == 2:
        return f"{keywords[0]} or {keywords[1]}"
    return f"{', '.join(keywords[:-1])} or {keywords[-1]}"


# G3 was best in both DEV probes; wording differences were within noise at n=8/11.
GATE_HYPOTHESIS = "{company} committed {risks}."
# V flags possible victim/cleared cases without rejecting them, preserving recall.
VICTIM_HYPOTHESIS = "{company} is a victim of crime or was cleared of wrongdoing."
# Tagging asks what the article is about, unlike the gate asking whether the company did it.
# T2 won on DEV versus T1; at n=8, differences under about 0.05 F1 are noise.
TAG_HYPOTHESIS = "This article is about {keyword}."
