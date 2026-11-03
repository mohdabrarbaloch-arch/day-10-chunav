"""Generate the three sample datasets shipped with Chunav.

Each one is synthetic but *not* random noise: the features carry a real,
documented relationship to the target, so a model that scores well is
genuinely learning something and a model that scores at baseline is genuinely
failing. That distinction is the whole point of shipping samples.

Run:  python scripts/make_samples.py
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"wrote {path.relative_to(ROOT)}  ({len(rows)} rows)")


def iris_like(n: int = 150, seed: int = 7) -> None:
    """Three flower species separated by petal geometry.

    Deliberately separable but not trivially so — the two versicolor/virginica
    clusters overlap, which is what makes the confusion matrix interesting.
    """
    rng = random.Random(seed)
    rows = []
    centres = {
        "setosa": (1.5, 0.25, 5.0, 3.4),
        "versicolor": (4.3, 1.3, 5.9, 2.8),
        "virginica": (5.6, 2.0, 6.6, 3.0),
    }
    for species, (pl, pw, sl, sw) in centres.items():
        for _ in range(n // 3):
            rows.append(
                [
                    round(rng.gauss(sl, 0.35), 2),
                    round(rng.gauss(sw, 0.30), 2),
                    round(rng.gauss(pl, 0.45), 2),
                    round(max(0.05, rng.gauss(pw, 0.22)), 2),
                    species,
                ]
            )
    rng.shuffle(rows)
    write_csv(
        SAMPLES / "iris.csv",
        ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"],
        rows,
    )


def churn(n: int = 400, seed: int = 11) -> None:
    """Telecom churn with a genuinely weak signal.

    The log-odds are small on purpose. A model here should land a few points
    above the majority baseline and no more — this is the sample that teaches
    you not to trust a 95% score on a 63% base rate.
    """
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        tenure = rng.randint(1, 72)
        monthly = round(rng.uniform(500, 4500), 0)
        support_calls = rng.randint(0, 9)
        contract = rng.choice(["monthly", "one_year", "two_year"])
        has_internet = rng.choice(["yes", "no"])

        logit = (
            -0.9
            - 0.028 * tenure
            + 0.00022 * monthly
            + 0.19 * support_calls
            + (0.55 if contract == "monthly" else -0.35 if contract == "two_year" else 0.0)
            + (0.15 if has_internet == "yes" else -0.1)
        )
        p = 1 / (1 + pow(2.718281828, -logit))
        churned = "yes" if rng.random() < p else "no"

        rows.append([tenure, int(monthly), support_calls, contract, has_internet, churned])
    write_csv(
        SAMPLES / "churn.csv",
        [
            "tenure_months",
            "monthly_bill_pkr",
            "support_calls",
            "contract",
            "has_internet",
            "churned",
        ],
        rows,
    )


def exam_score(n: int = 300, seed: int = 13) -> None:
    """Regression: study hours and attendance predict an exam score.

    The relationship is linear with real noise, so ridge should win and the
    R² should land somewhere in the 0.7–0.9 band. A suspiciously perfect R²
    would mean the generator leaked the target.
    """
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        study_hours = round(rng.uniform(0, 12), 1)
        attendance = round(rng.uniform(40, 100), 1)
        prior_gpa = round(rng.uniform(1.8, 4.0), 2)
        sleep_hours = round(rng.uniform(4, 9), 1)

        score = (
            18.0
            + 3.4 * study_hours
            + 0.42 * attendance
            + 6.1 * prior_gpa
            + 1.1 * sleep_hours
            + rng.gauss(0, 5.5)
        )
        rows.append(
            [study_hours, attendance, prior_gpa, sleep_hours, round(max(0.0, min(100.0, score)), 1)]
        )
    write_csv(
        SAMPLES / "exam_score.csv",
        ["study_hours", "attendance_pct", "prior_gpa", "sleep_hours", "final_score"],
        rows,
    )


if __name__ == "__main__":
    iris_like()
    churn()
    exam_score()
    print("done")
