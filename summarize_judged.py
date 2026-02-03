import json
import argparse
from pathlib import Path
from math import sqrt


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = z * sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return center - margin, center + margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", required=True, help="judged jsonl")
    ap.add_argument(
        "--model_a",
        default="model",
        help=(
            "label to count as 'win'. Used with winner_key or winner_model_vs_chosen "
            "(default: model)"
        ),
    )
    args = ap.parse_args()

    path = Path(args.in_file)

    win = lose = tie = 0
    unknown = 0
    total = 0
    field_counts = {
        "winner_model_vs_chosen": 0,
        "winner_key": 0,
        "winner": 0,
        "missing": 0,
    }

    def _norm(val):
        if val is None:
            return ""
        return str(val).strip()

    def _lower(val):
        return _norm(val).lower()

    def _is_tie(val):
        return _lower(val) in ("tie", "tied", "draw")

    def _pick_winner(ex):
        for key in ("winner_model_vs_chosen", "winner_key", "winner"):
            if key in ex:
                return key, ex[key]
        return None, None

    def _target_for_mvc(ex):
        model_a_lower = _lower(args.model_a)
        if model_a_lower in ("model", "chosen"):
            return model_a_lower

        model_id = ex.get("model_id")
        chosen_id = ex.get("chosen_id")
        if model_id is not None and str(model_id) == args.model_a:
            return "model"
        if chosen_id is not None and str(chosen_id) == args.model_a:
            return "chosen"
        return "model"

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            total += 1

            field, result = _pick_winner(ex)
            if field is None:
                field_counts["missing"] += 1
                unknown += 1
                continue

            field_counts[field] += 1
            if _is_tie(result):
                tie += 1
                continue

            if field == "winner_model_vs_chosen":
                winner = _lower(result)
                target = _target_for_mvc(ex)
                if winner == target:
                    win += 1
                elif winner in ("model", "chosen"):
                    lose += 1
                else:
                    unknown += 1
            elif field == "winner_key":
                winner = _norm(result)
                if winner in ("A", "B", "a", "b") and isinstance(
                    ex.get("labels"), dict
                ):
                    winner = ex["labels"].get(winner.upper(), winner)
                if winner == args.model_a or _lower(winner) == _lower(args.model_a):
                    win += 1
                else:
                    lose += 1
            else:  # field == "winner" (raw A/B)
                winner = _lower(result)
                if winner == "a":
                    win += 1
                elif winner == "b":
                    lose += 1
                else:
                    unknown += 1

    effective = win + lose
    win_rate = win / effective if effective > 0 else 0.0
    ci_low, ci_high = wilson_ci(win, effective)

    print("=" * 60)
    print(f"File: {path.name}")
    print(f"Model label: {args.model_a}")
    print(f"Total judged: {total}")
    print(f"Win / Lose / Tie: {win} / {lose} / {tie}")
    if unknown:
        print(f"Unknown/Unmapped: {unknown}")
    print(f"Win-rate (exclude tie): {win_rate:.3f}")
    print(f"95% CI (Wilson): [{ci_low:.3f}, {ci_high:.3f}]")
    used_fields = [k for k, v in field_counts.items() if v > 0]
    print(f"Winner fields used: {', '.join(used_fields)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
