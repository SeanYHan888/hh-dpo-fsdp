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
    ap.add_argument("--model_a", default="model", help="label of model a")
    args = ap.parse_args()

    path = Path(args.in_file)

    win = lose = tie = 0
    total = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            total += 1

            # expected field from gpt_judge_pairwise.py
            result = ex["winner"]  # "a", "b", or "tie"

            if result == "a":
                win += 1
            elif result == "b":
                lose += 1
            else:
                tie += 1

    effective = win + lose
    win_rate = win / effective if effective > 0 else 0.0
    ci_low, ci_high = wilson_ci(win, effective)

    print("=" * 60)
    print(f"File: {path.name}")
    print(f"Total judged: {total}")
    print(f"Win / Lose / Tie: {win} / {lose} / {tie}")
    print(f"Win-rate (exclude tie): {win_rate:.3f}")
    print(f"95% CI (Wilson): [{ci_low:.3f}, {ci_high:.3f}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
