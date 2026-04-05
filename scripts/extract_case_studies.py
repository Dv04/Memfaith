"""Extract 6 distinct qualitative case studies from CCS experiment logs.

Case study categories:
  1. Distractor chunk that caused a flip (false causal signal)
  2. Gold evidence chunk that did NOT cause a flip (missed causal signal)
  3. Gold evidence chunk that DID cause a flip (correct causal detection)
  4. Multi-hop example showing distributed causality (≥2 chunks flip)
  5. FEVER REFUTES example with high CCS
  6. FEVER SUPPORTS example with low CCS
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.memfaith.metrics import load_experiment_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract qualitative case studies.")
    parser.add_argument("--fever-log", type=str, default="outputs/memfaith/fever_smoke_ccs.jsonl")
    parser.add_argument("--hotpot-log", type=str, default="outputs/memfaith/hotpot_smoke_ccs.jsonl")
    parser.add_argument("--output-path", type=str, default="outputs/memfaith/case_studies.md")
    return parser.parse_args()


def _find_best(
    records: List[Dict],
    condition,
    sort_key=None,
    reverse: bool = True,
) -> Optional[Dict]:
    candidates = [r for r in records if condition(r)]
    if not candidates:
        return None
    if sort_key:
        candidates.sort(key=sort_key, reverse=reverse)
    return candidates[0]


def _format_case(
    number: int,
    title: str,
    record: Optional[Dict],
    abl: Optional[Dict] = None,
    explanation: str = "",
) -> str:
    if record is None:
        return f"## Case {number}: {title}\n\n*No matching example found in the current logs.*\n\n---\n"

    lines = [
        f"## Case {number}: {title}",
        "",
        f"**Dataset:** {record.get('dataset', 'unknown')} | "
        f"**Example ID:** `{record.get('example_id', 'N/A')}` | "
        f"**K:** {record.get('k', 'N/A')}",
        "",
        f"**Query:** {record.get('query', 'N/A')}",
        "",
        f"**Gold Answer:** {record.get('gold_answer', 'N/A')}",
        "",
        f"**Full-Context Prediction:** {record['full_context']['prediction']['raw_text']}",
        f"(Correct: {'✅' if record['full_context']['is_correct'] else '❌'})",
        "",
        f"**CCS:** {record.get('ccs_example', 'N/A')}",
        "",
    ]

    if abl:
        lines.extend(
            [
                f"### Ablated Chunk {abl['chunk_id']}",
                f"- **Contains gold evidence:** {'Yes' if abl.get('gold_segment_ids') else 'No'}",
                f"- **Flipped:** {'Yes ⚡' if abl['comparison_to_full']['flipped'] else 'No'}",
                f"- **Comparison method:** `{abl['comparison_to_full']['method']}`",
                f"- **Score:** {abl['comparison_to_full']['score']:.4f}",
                f"- **Ablated prediction:** {abl['prediction']['raw_text']}",
                "",
                "**Chunk text (truncated):**",
                "```",
                abl["chunk_text"][:400],
                "```",
                "",
            ]
        )

    if explanation:
        lines.extend(["**Analysis:**", explanation, ""])

    lines.extend(["---", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    all_records: List[Dict] = []
    for path in [args.fever_log, args.hotpot_log]:
        p = Path(path)
        if p.exists():
            all_records.extend(load_experiment_log(str(p)))

    active = [r for r in all_records if int(r.get("k", 0)) > 0]
    fever_active = [r for r in active if r.get("dataset") == "fever"]
    hotpot_active = [r for r in active if r.get("dataset") == "hotpotqa"]

    cases: List[str] = [
        "# MemFaith Qualitative Case Studies",
        "",
        "> **NOTE:** These case studies are drawn from **synthetic validation data**",
        "> and serve as structural demonstrations of the analysis pipeline.",
        "> Final case studies should be regenerated from real model outputs.",
        "",
    ]

    # Case 1: Distractor chunk that caused a flip (false causal signal)
    case1_rec = None
    case1_abl = None
    for r in active:
        for abl in r.get("ablations") or []:
            if abl["comparison_to_full"]["flipped"] and not abl.get("gold_segment_ids"):
                case1_rec = r
                case1_abl = abl
                break
        if case1_rec:
            break

    cases.append(
        _format_case(
            1,
            "Distractor Chunk Causing a Flip (False Causal Signal)",
            case1_rec,
            case1_abl,
            "This case shows a non-evidence chunk whose removal flips the answer. "
            "This is a false positive: the model's prediction changed even though "
            "the removed chunk contained no gold evidence, suggesting the model "
            "may be relying on spurious correlations in the distractor text.",
        )
    )

    # Case 2: Gold evidence that did NOT cause a flip (missed causal signal)
    case2_rec = None
    case2_abl = None
    for r in active:
        for abl in r.get("ablations") or []:
            if not abl["comparison_to_full"]["flipped"] and abl.get("gold_segment_ids"):
                case2_rec = r
                case2_abl = abl
                break
        if case2_rec:
            break

    cases.append(
        _format_case(
            2,
            "Gold Evidence NOT Causing a Flip (Missed Causal Signal)",
            case2_rec,
            case2_abl,
            "This case shows a chunk with gold evidence whose removal does not "
            "change the answer. The model either has redundant evidence pathways "
            "or relies on parametric priors rather than the contextual evidence.",
        )
    )

    # Case 3: Gold evidence that DID cause a flip (correct detection)
    case3_rec = None
    case3_abl = None
    for r in active:
        for abl in r.get("ablations") or []:
            if abl["comparison_to_full"]["flipped"] and abl.get("gold_segment_ids"):
                case3_rec = r
                case3_abl = abl
                break
        if case3_rec:
            break

    cases.append(
        _format_case(
            3,
            "Gold Evidence Causing a Flip (Correct Causal Detection)",
            case3_rec,
            case3_abl,
            "This is the ideal outcome: removing gold evidence causes the model's "
            "answer to change, confirming that this chunk is causally necessary "
            "for the model's reasoning.",
        )
    )

    # Case 4: Multi-hop distributed causality (≥2 chunks flip)
    case4_rec = None
    for r in hotpot_active:
        flip_count = sum(
            1
            for abl in (r.get("ablations") or [])
            if abl["comparison_to_full"]["flipped"]
        )
        if flip_count >= 2:
            case4_rec = r
            break

    cases.append(
        _format_case(
            4,
            "Multi-Hop Distributed Causality (≥2 Chunks Flip Independently)",
            case4_rec,
            explanation=(
                "This multi-hop example demonstrates distributed causal necessity: "
                "removing multiple different chunks each independently causes "
                "an answer flip. This proves the model requires multiple pieces "
                "of evidence simultaneously for correct reasoning."
            )
            if case4_rec
            else "",
        )
    )

    # Case 5: FEVER REFUTES with high CCS
    case5_rec = _find_best(
        fever_active,
        condition=lambda r: r.get("gold_answer", "").upper() == "REFUTES" and r.get("ccs_example") is not None,
        sort_key=lambda r: float(r.get("ccs_example", 0)),
    )
    cases.append(
        _format_case(
            5,
            "FEVER REFUTES with High CCS",
            case5_rec,
            explanation=(
                "REFUTES claims tend to show higher causal dependency. "
                "This example demonstrates that the model requires stronger "
                "evidence-grounding to reject a false claim compared to "
                "confirming a true one."
            )
            if case5_rec
            else "",
        )
    )

    # Case 6: FEVER SUPPORTS with low CCS
    case6_rec = _find_best(
        fever_active,
        condition=lambda r: r.get("gold_answer", "").upper() == "SUPPORTS" and r.get("ccs_example") is not None,
        sort_key=lambda r: float(r.get("ccs_example", 0)),
        reverse=False,
    )
    cases.append(
        _format_case(
            6,
            "FEVER SUPPORTS with Low CCS",
            case6_rec,
            explanation=(
                "SUPPORTS claims often exhibit lower causal scores, suggesting "
                "the model may confirm true claims using parametric priors "
                "or partial evidence rather than requiring full context."
            )
            if case6_rec
            else "",
        )
    )

    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(cases), encoding="utf-8")
    print(f"Wrote 6 case studies to {output}")


if __name__ == "__main__":
    main()
