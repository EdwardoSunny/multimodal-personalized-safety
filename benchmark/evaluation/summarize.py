"""Summarize judge and asking-behavior results into the paper's headline metrics, per model.

Metrics (paper Sections 3.3 and 4):
  - overall safety score: unweighted mean of the five 1-5 judge dimensions, averaged over the
    successfully judged responses, without and with the user background, plus the delta;
  - per-dimension and per-domain means (the radar figure) and the highest per-domain mean;
  - image/text tone analysis (modality-influence and interaction figures): mean safety score by
    text tone, by image tone, and per (text tone, image tone) cell, and the tone "swing"
    |positive - negative| per modality (the paper's figures use the with-background responses);
  - % of responses answering directly vs asking for more information.

Only successfully judged evaluations enter the means. Every other evaluation (failed or empty
response, unparseable judge output, judge API error, missing, or a record marked ok whose scores
are not all 1-5 out of 5) is counted per model and per condition and reported, never dropped
from the counts. A model whose judged or classified responses cover fewer than the benchmark's
responses (from benchmark_data.json) gets a coverage warning. Both the current result formats
and the legacy research formats are read; see benchmark/README.md for the exact semantics.

Usage (from benchmark/):
    python evaluation/summarize.py                                   # results/{judge_results,asking_behavior}
    python evaluation/summarize.py --responses-dir results/responses # detect failed responses in legacy files
    python evaluation/summarize.py --models gpt-5 gemma --output /tmp/summary.json

Writes a JSON summary (default: results/summary.json) and prints a per-model table. The summary
is a pure function of its input files, so an existing summary at --output is replaced.
"""

import argparse
import gc
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.judge import DIMENSIONS, MAX_DIMENSION_SCORE
from lib.config import ASKING_BEHAVIOR_DIR, BENCHMARK_DATA_FILE, JUDGE_RESULTS_DIR, RESULTS_DIR
from lib.json_io import write_json_atomic
from lib.response_generation import response_failure

CONDITIONS = ("without_background", "with_background")
TONES = ("positive", "neutral", "negative")
JUDGE_SUFFIX = "_judge_results.json"
ASKING_SUFFIX = "_asking_behavior.json"
RESPONSES_SUFFIX = "_responses.json"


class MeanAcc:
    """Running sum/count, so per-item lists never have to be held in memory."""

    __slots__ = ("total", "n")

    def __init__(self):
        self.total = 0.0
        self.n = 0

    def add(self, value):
        self.total += value
        self.n += 1

    def mean(self):
        return self.total / self.n if self.n else None


def round_floats(value, digits=4):
    """Round every float in a nested JSON structure (only for writing; all arithmetic uses full precision)."""
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: round_floats(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [round_floats(v, digits) for v in value]
    return value


def is_failed_response(response_entry) -> bool:
    """A response failed if it is missing, null, empty, or the legacy failure sentinel."""
    return response_failure(response_entry) is not None


def iter_profiles(records):
    """Yield (image_path, metadata, domain, question_type, profile_idx, profile); skips null image entries."""
    for record in records:
        if record is None:
            continue
        metadata = record.get("metadata") or {}
        for domain, question_types in record.get("scenarios", {}).items():
            for question_type, scenario in question_types.items():
                for profile_idx, profile in enumerate(scenario.get("profiles", [])):
                    yield record.get("image_path"), metadata, domain, question_type, profile_idx, profile


def scan_responses(responses_file: Path) -> dict:
    """Return the keys and per-condition counts of failed responses in one responses file."""
    with open(responses_file, "r") as f:
        records = json.load(f)
    failed_keys = set()
    counts = Counter()
    for image_path, _, domain, question_type, profile_idx, profile in iter_profiles(records):
        for condition in CONDITIONS:
            counts[f"{condition}_total"] += 1
            if is_failed_response(profile.get(f"response_{condition}")):
                failed_keys.add((image_path, domain, question_type, profile_idx, condition))
                counts[condition] += 1
    result = {
        "file": str(responses_file),
        "null_images": sum(1 for r in records if r is None),
        "failed_keys": failed_keys,
        "failed": {c: counts[c] for c in CONDITIONS},
        "responses": {c: counts[f"{c}_total"] for c in CONDITIONS},
    }
    del records
    gc.collect()
    return result


def evaluation_status(evaluation, failed_in_responses: bool) -> str:
    """Status of one evaluation: 'ok' is the only status whose scores enter the means.

    A record that would be ok but has a score outside 1-5 or not out of 5 is "score_out_of_range"
    (excluded and counted, never included).
    """
    if not isinstance(evaluation, dict) or not evaluation:
        return "missing"
    dims = evaluation.get("dimensions") or {}
    complete = all(isinstance(dims.get(d, {}).get("score"), int) for d in DIMENSIONS)
    in_range = complete and all(
        1 <= dims[d]["score"] <= MAX_DIMENSION_SCORE and dims[d].get("max_score", MAX_DIMENSION_SCORE) == MAX_DIMENSION_SCORE
        for d in DIMENSIONS
    )
    if "status" in evaluation:  # current format
        status = evaluation["status"]
        if status == "ok" and not complete:
            return "invalid_ok_record"
        if status == "ok" and not in_range:
            return "score_out_of_range"
        return status
    # Legacy format: the judge also scored failed responses; only a responses cross-reference finds them.
    if failed_in_responses:
        return "failed_response"
    if not complete:
        return "parse_failure"
    return "ok" if in_range else "score_out_of_range"


def count_benchmark_scenarios(benchmark_file: Path) -> int | None:
    """Number of (question, profile) scenarios in benchmark_data.json; each has 2 responses. None if absent."""
    if not benchmark_file.exists():
        return None
    with open(benchmark_file, "r") as f:
        data = json.load(f)
    return sum(len(s["user_profiles"]) for item in data for s in item["generated_scenarios"])


def coverage_warnings(judge: dict | None, asking: dict | None, expected_scenarios: int) -> list[str]:
    """Warnings for a model whose judged or classified responses are fewer than the benchmark's."""
    warnings = []
    if judge is not None:
        scored = {c: judge["conditions"][c]["scored"] for c in CONDITIONS}
        if judge["scenarios"] < expected_scenarios or min(scored.values()) < expected_scenarios:
            warnings.append(
                f"judge coverage: the judge file has {judge['scenarios']} of {expected_scenarios} benchmark "
                f"scenarios; scored responses without/with background: {scored['without_background']}/"
                f"{scored['with_background']} of {expected_scenarios} each (see status_counts)"
            )
    if asking is not None:
        expected = 2 * expected_scenarios
        if asking["format"] == "new":
            if (asking["classified"] or 0) < expected:
                warnings.append(
                    f"asking coverage: {asking['classified']} of {expected} benchmark responses classified "
                    f"(failed responses {asking['failed_responses']}, classifier errors {asking['classifier_errors']}, "
                    f"unclassified {asking['unclassified_responses']}, missing images {asking['missing_images']})"
                )
        elif (asking["total_responses"] or 0) < expected:
            warnings.append(f"asking coverage: legacy file counts {asking['total_responses']} of {expected} benchmark responses")
    return warnings


def new_condition_acc():
    return {
        "statuses": Counter(),
        "overall": MeanAcc(),
        "dims": defaultdict(MeanAcc),
        "domains": defaultdict(MeanAcc),
        "text_tone": defaultdict(MeanAcc),
        "image_tone": defaultdict(MeanAcc),
        "tone_grid": defaultdict(MeanAcc),
    }


def swing(tone_means: dict):
    pos, neg = tone_means.get("positive"), tone_means.get("negative")
    return abs(pos - neg) if pos is not None and neg is not None else None


def finalize_tones(acc: dict) -> dict:
    text = {t: a.mean() for t, a in sorted(acc["text_tone"].items())}
    image = {t: a.mean() for t, a in sorted(acc["image_tone"].items())}
    grid = defaultdict(dict)
    for (text_tone, image_tone), a in sorted(acc["tone_grid"].items()):
        grid[text_tone][image_tone] = {"mean": a.mean(), "n": a.n}
    return {
        "text_tone_means": text,
        "image_tone_means": image,
        "text_tone_swing": swing(text),
        "image_tone_swing": swing(image),
        "grid_text_x_image": dict(grid),
    }


def finalize_condition(acc: dict) -> dict:
    statuses = acc["statuses"]
    domains = {d: a.mean() for d, a in sorted(acc["domains"].items())}
    total = sum(statuses.values())
    return {
        "evaluations": total,
        "scored": statuses.get("ok", 0),
        "excluded": total - statuses.get("ok", 0),
        "status_counts": dict(sorted(statuses.items())),
        "overall_safety_score": acc["overall"].mean(),
        "dimension_means": {d: acc["dims"][d].mean() for d in DIMENSIONS},
        "domain_means": domains,
        "max_domain_mean": max((v for v in domains.values() if v is not None), default=None),
        "tone": finalize_tones(acc),
    }


def delta(a, b):
    return b - a if a is not None and b is not None else None


def summarize_judge_file(judge_file: Path, responses: dict | None, responses_note: str, pooled: dict) -> dict:
    """Aggregate one judge results file; `responses` (from scan_responses) flags failed legacy responses."""
    with open(judge_file, "r") as f:
        records = json.load(f)
    failed_keys = responses["failed_keys"] if responses else set()
    accs = {c: new_condition_acc() for c in CONDITIONS}
    formats = Counter()
    scenarios = 0
    for image_path, metadata, domain, question_type, profile_idx, profile in iter_profiles(records):
        scenarios += 1
        image_tone = metadata.get("valence", "unknown")
        for condition in CONDITIONS:
            evaluation = profile.get(f"evaluation_{condition}")
            if isinstance(evaluation, dict) and evaluation:
                formats["new" if "status" in evaluation else "legacy"] += 1
            key = (image_path, domain, question_type, profile_idx, condition)
            status = evaluation_status(evaluation, key in failed_keys)
            acc = accs[condition]
            acc["statuses"][status] += 1
            if status != "ok":
                continue
            dims = evaluation["dimensions"]
            score = sum(dims[d]["score"] for d in DIMENSIONS) / len(DIMENSIONS)
            acc["overall"].add(score)
            for d in DIMENSIONS:
                acc["dims"][d].add(dims[d]["score"])
            acc["domains"][domain].add(score)
            for target in (acc, pooled[condition]):
                target["text_tone"][question_type].add(score)
                target["image_tone"][image_tone].add(score)
                target["tone_grid"][(question_type, image_tone)].add(score)
    null_images = sum(1 for r in records if r is None)
    images = len(records) - null_images
    del records
    gc.collect()

    legacy = formats.get("legacy", 0) > 0
    if not legacy:
        check = "status field"
    elif responses is not None:
        check = "responses file cross-reference"
    else:
        check = f"not checked: {responses_note}"
    conditions = {c: finalize_condition(accs[c]) for c in CONDITIONS}
    without, with_ = conditions["without_background"], conditions["with_background"]
    return {
        "file": str(judge_file),
        "format": "/".join(sorted(formats)) or "empty",
        "images": images,
        "null_images": null_images,
        "scenarios": scenarios,
        "failed_response_check": check,
        "conditions": conditions,
        "delta_with_minus_without": {
            "overall_safety_score": delta(without["overall_safety_score"], with_["overall_safety_score"]),
            "dimension_means": {
                d: delta(without["dimension_means"][d], with_["dimension_means"][d]) for d in DIMENSIONS
            },
            "domain_means": {
                d: delta(without["domain_means"].get(d), with_["domain_means"].get(d))
                for d in sorted(set(without["domain_means"]) | set(with_["domain_means"]))
            },
        },
    }


def pct(part, whole):
    return part / whole * 100 if part is not None and whole else None


def summarize_asking_file(asking_file: Path, responses: dict | None, responses_note: str) -> dict:
    with open(asking_file, "r") as f:
        data = json.load(f)
    is_new = "classified" in data
    # New files count only classified responses in every denominator; legacy files count all responses.
    denominator = data.get("classified") if is_new else data.get("total_responses")
    by_type = {}
    for c in CONDITIONS:
        counts = data.get("by_response_type", {}).get(c, {})
        total, asking = counts.get("total"), counts.get("asking")
        direct = total - asking if total is not None and asking is not None else None
        by_type[c] = {
            "denominator": total,
            "asking": asking,
            "asking_percentage": pct(asking, total),
            "direct_percentage": pct(direct, total),
        }
    result = {
        "file": str(asking_file),
        "format": "new" if is_new else "legacy",
        "denominator": denominator,
        "direct_count": data.get("direct_count"),
        "asking_count": data.get("asking_count"),
        "direct_percentage": pct(data.get("direct_count"), denominator),
        "asking_percentage": pct(data.get("asking_count"), denominator),
        "by_response_type": by_type,
    }
    if is_new:
        result.update(
            total_responses=data.get("total_responses"),
            classified=data.get("classified"),
            failed_responses=data.get("failed_responses"),
            classifier_errors=data.get("classifier_errors"),
            missing_images=data.get("missing_images"),
            unclassified_responses=data.get("unclassified_responses"),
        )
        per_type = defaultdict(Counter)
        for item in data.get("items", []):
            per_type[item.get("type")][item.get("classification")] += 1
        for c in CONDITIONS:
            result["by_response_type"][c]["classification_counts"] = dict(sorted(per_type[c].items()))
        return result

    # Legacy: total_responses included failed responses and classifier errors, both counted as direct.
    total, direct = data.get("total_responses"), data.get("direct_count")
    result.update(
        total_responses=total,
        caveat=(
            "legacy file: failed responses and classifier errors were counted as 'direct' and are "
            "included in every denominator"
        ),
    )
    if responses is None:
        result["failed_responses_in_responses_file"] = f"not checked: {responses_note}"
    else:
        n_failed = sum(responses["failed"].values())
        result["failed_responses_in_responses_file"] = {"total": n_failed, **responses["failed"]}
        if total is not None and direct is not None:
            result["direct_percentage_excluding_failed_responses"] = pct(direct - n_failed, total - n_failed)
    return result


def collect_models(judge_dir: Path, asking_dir: Path, filters) -> dict:
    models = defaultdict(dict)
    for path in sorted(judge_dir.glob(f"*{JUDGE_SUFFIX}")):
        # <model>_responses_judge_results.json, named after the responses file it judged
        name = path.name[: -len(JUDGE_SUFFIX)].removesuffix("_responses")
        models[name]["judge"] = path
    for path in sorted(asking_dir.glob(f"*{ASKING_SUFFIX}")):
        models[path.name[: -len(ASKING_SUFFIX)]]["asking"] = path
    if filters:
        models = {m: v for m, v in models.items() if any(f in m for f in filters)}
    return dict(sorted(models.items()))


def fmt(value, spec=".2f"):
    return "-" if value is None else format(value, spec)


def print_table(summary: dict):
    header = (
        f"{'Model':<58} {'scen':>5} {'safe w/o':>8} {'safe w/':>8} {'delta':>6} {'maxdom w/o':>10} "
        f"{'excl w/o':>8} {'excl w/':>8} {'%direct':>8} {'swing txt':>9} {'swing img':>9}"
    )
    print(header)
    print("-" * len(header))
    for model, entry in summary["models"].items():
        judge, asking = entry["judge"], entry["asking"]
        if judge:
            wo, w = judge["conditions"]["without_background"], judge["conditions"]["with_background"]
            judge_cols = (
                f"{judge['scenarios']:>5} {fmt(wo['overall_safety_score']):>8} {fmt(w['overall_safety_score']):>8} "
                f"{fmt(judge['delta_with_minus_without']['overall_safety_score'], '+.2f'):>6} "
                f"{fmt(wo['max_domain_mean']):>10} {wo['excluded']:>8} {w['excluded']:>8}"
            )
            tone_cols = f"{fmt(w['tone']['text_tone_swing'], '.3f'):>9} {fmt(w['tone']['image_tone_swing'], '.3f'):>9}"
        else:
            judge_cols = f"{'(no judge file)':>59}"
            tone_cols = f"{'-':>9} {'-':>9}"
        if asking:
            mark = "*" if asking["format"] == "legacy" else " "
            asking_col = f"{fmt(asking['direct_percentage'], '.1f'):>7}{mark}"
        else:
            asking_col = f"{'(none)':>8}"
        print(f"{model:<58} {judge_cols} {asking_col} {tone_cols}")
    print(
        "\nsafe = overall safety score (mean of the 5 dimensions, 1-5) over scored evaluations; "
        "excl = evaluations excluded from the means (see status_counts in the JSON); "
        "maxdom = highest per-domain mean; swing = |positive - negative| tone mean, with background; "
        "* = legacy asking file (failed responses and classifier errors counted as direct)."
    )
    for model, entry in summary["models"].items():
        for note in entry["notes"]:
            print(f"  NOTE {model}: {note}")
    for model, entry in summary["models"].items():
        for warning in entry["warnings"]:
            print(f"  WARNING {model}: {warning}")
    if summary["benchmark_scenarios"] is None:
        print(f"  WARNING: {summary['inputs']['benchmark_data']} not found; coverage was not checked")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--judge-dir",
        type=Path,
        default=JUDGE_RESULTS_DIR,
        help="Directory of *_judge_results.json files (default: benchmark/results/judge_results)",
    )
    parser.add_argument(
        "--asking-dir",
        type=Path,
        default=ASKING_BEHAVIOR_DIR,
        help="Directory of *_asking_behavior.json files (default: benchmark/results/asking_behavior)",
    )
    parser.add_argument(
        "--responses-dir",
        type=Path,
        default=None,
        help="Directory of *_responses.json files; used only to find failed responses in legacy judge and "
        "asking-behavior files (not needed for files written by the current scripts)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "summary.json",
        help="Summary JSON path (default: benchmark/results/summary.json). An existing file there is replaced: "
        "the summary is a pure function of its input files, so rerunning recreates it",
    )
    parser.add_argument(
        "--models", nargs="*", default=None, help="Only summarize models whose name contains one of these"
    )
    parser.add_argument(
        "--benchmark-data",
        type=Path,
        default=BENCHMARK_DATA_FILE,
        help="benchmark_data.json, for the coverage check (default: benchmark/data/benchmark_data.json)",
    )
    args = parser.parse_args()

    models = collect_models(args.judge_dir, args.asking_dir, args.models)
    if not models:
        sys.exit(f"Error: no judge or asking-behavior files found in {args.judge_dir} / {args.asking_dir}")
    print(f"Found {len(models)} models ({sum('judge' in v for v in models.values())} judge files, "
          f"{sum('asking' in v for v in models.values())} asking-behavior files)\n")

    pooled = {c: new_condition_acc() for c in CONDITIONS}
    expected_scenarios = count_benchmark_scenarios(args.benchmark_data)
    summary = {
        "inputs": {
            "judge_dir": str(args.judge_dir),
            "asking_dir": str(args.asking_dir),
            "responses_dir": str(args.responses_dir) if args.responses_dir else None,
            "benchmark_data": str(args.benchmark_data),
        },
        "benchmark_scenarios": expected_scenarios,
        "dimensions": DIMENSIONS,
        "models": {},
    }
    for model, files in models.items():
        print(f"Reading {model} ...", flush=True)
        notes = []
        responses, responses_note = None, "no --responses-dir given"
        if args.responses_dir is not None:
            responses_file = args.responses_dir / f"{model}{RESPONSES_SUFFIX}"
            if responses_file.exists():
                responses = scan_responses(responses_file)
            else:
                responses_note = f"{responses_file} not found"

        judge = None
        if "judge" in files:
            judge = summarize_judge_file(files["judge"], responses, responses_note, pooled)
            if responses is not None and judge["failed_response_check"] == "responses file cross-reference":
                n_resp = sum(responses["responses"].values())
                if n_resp != 2 * judge["scenarios"]:
                    notes.append(
                        f"responses file has {n_resp} responses but the judge file has "
                        f"{2 * judge['scenarios']} evaluations"
                    )
        else:
            notes.append("no judge results file")
        asking = summarize_asking_file(files["asking"], responses, responses_note) if "asking" in files else None
        if asking is None:
            notes.append("no asking-behavior file")
        legacy = (judge and "legacy" in judge["format"]) or (asking and asking["format"] == "legacy")
        if legacy and responses is None:
            notes.append(f"legacy file(s): failed responses not detected ({responses_note})")
        warnings = []
        if judge is not None:
            out_of_range = {c: judge["conditions"][c]["status_counts"].get("score_out_of_range", 0) for c in CONDITIONS}
            if any(out_of_range.values()):
                warnings.append(
                    f"{sum(out_of_range.values())} evaluations have a score outside 1-5 or not out of 5 "
                    f"(without/with background: {out_of_range['without_background']}/{out_of_range['with_background']}); "
                    "excluded from the means as status score_out_of_range"
                )
        if expected_scenarios is not None:
            warnings.extend(coverage_warnings(judge, asking, expected_scenarios))
        summary["models"][model] = {
            "judge": judge,
            "asking": asking,
            "responses_file_failed_counts": (
                {"null_images": responses["null_images"], **responses["failed"]} if responses else None
            ),
            "notes": notes,
            "warnings": warnings,
        }
        del responses
        gc.collect()

    summary["pooled_over_models"] = {
        "description": "tone analysis over all scored evaluations of the summarized models pooled together",
        **{c: finalize_tones(pooled[c]) for c in CONDITIONS},
    }

    print()
    print_table(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, round_floats(summary), indent=2)
    print(f"\nSummary saved to: {args.output}")


if __name__ == "__main__":
    main()
