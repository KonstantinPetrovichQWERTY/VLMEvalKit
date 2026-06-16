from typing import Dict, Any
import math
import json
from pathlib import Path
from .base_detector import BaseDetector
from datetime import datetime


class AnswerOptionsDistributionDetector(BaseDetector):
    NAME = "answer_options_distribution"
    DESCRIPTION = (
        "Detect answer-position distribution bias in multiple-choice datasets."
    )
    DEFAULT_CONFIG = {}

    def analyze(self, context, **kwargs) -> Dict[str, Any]:
        # This method now accepts an AnalysisContext as described in FIXES.md.
        dataset = getattr(context, "dataset", None)
        dataset_name = getattr(context, "dataset_name", None)
        samples = None
        if hasattr(dataset, "data"):
            try:
                samples = dataset.data
            except Exception:
                samples = None

        counts = {}
        total = 0

        if samples is not None:
            try:
                import pandas as pd

                if isinstance(samples, pd.DataFrame):
                    for _, row in samples.iterrows():
                        opts = None
                        if "options" in row and isinstance(
                            row["options"], (list, tuple)
                        ):
                            opts = list(row["options"])
                        else:
                            letter_cols = [
                                c
                                for c in samples.columns
                                if isinstance(c, str) and len(c) == 1 and c.isalpha()
                            ]
                            letter_cols = sorted(letter_cols)
                            if letter_cols:
                                opts = [row[c] for c in letter_cols if c in row]

                        correct = None
                        for key in [
                            "answer",
                            "label",
                            "correct_answer",
                            "correct",
                            "gt",
                            "reference",
                        ]:
                            if key in row and not pd.isna(row[key]):
                                correct = row[key]
                                break

                        if opts is None or correct is None:
                            continue

                        idx = None
                        if isinstance(correct, str):
                            cstr = correct.strip()
                            if len(cstr) == 1 and cstr.isalpha():
                                idx = ord(cstr.upper()) - ord("A")
                        if idx is None:
                            try:
                                idx = int(correct)
                            except Exception:
                                try:
                                    idx = opts.index(correct)
                                except Exception:
                                    idx = None

                        if idx is None or idx < 0 or idx >= len(opts):
                            continue

                        label = chr(ord("A") + idx)
                        counts[label] = counts.get(label, 0) + 1
                        total += 1
            except Exception:
                samples = None

        if samples is None and hasattr(dataset, "__iter__"):
            try:
                for item in dataset:
                    opts = None
                    if isinstance(item, dict):
                        opts = item.get("options") or [
                            item.get(k)
                            for k in sorted(item.keys())
                            if isinstance(k, str) and len(k) == 1 and k.isalpha()
                        ]
                        correct = (
                            item.get("correct_answer")
                            or item.get("answer")
                            or item.get("label")
                        )
                    else:
                        opts = getattr(item, "options", None)
                        correct = (
                            getattr(item, "correct_answer", None)
                            or getattr(item, "answer", None)
                            or getattr(item, "label", None)
                        )

                    if not opts or correct is None:
                        continue

                    idx = None
                    if (
                        isinstance(correct, str)
                        and len(correct) == 1
                        and correct.isalpha()
                    ):
                        idx = ord(correct.upper()) - ord("A")
                    if idx is None:
                        try:
                            idx = int(correct)
                        except Exception:
                            try:
                                idx = opts.index(correct)
                            except Exception:
                                idx = None

                    if idx is None or idx < 0 or idx >= len(opts):
                        continue
                    label = chr(ord("A") + idx)
                    counts[label] = counts.get(label, 0) + 1
                    total += 1
            except Exception:
                pass

        if total == 0:
            return {
                "detector": self.NAME,
                "dataset": dataset_name,
                "num_samples": 0,
                "score": None,
                "distribution": {},
                "recommendation": [
                    "No multiple-choice annotations found or dataset unsupported."
                ],
            }

        max_label_ord = max(ord(k) for k in counts.keys())
        labels = [chr(i) for i in range(ord("A"), max_label_ord + 1)]
        dist = {lbl: counts.get(lbl, 0) / total for lbl in labels}

        n = len(labels)
        expected = 1.0 / n
        deviations = {lbl: abs(prob - expected) for lbl, prob in dist.items()}
        max_dev = max(deviations.values())
        score = max_dev / (1.0 - expected) if (1.0 - expected) > 0 else 0.0
        expected_count = total / n
        chi2 = sum(
            (counts.get(lbl, 0) - expected_count) ** 2 / expected_count
            for lbl in labels
        )
        kl = 0.0
        for lbl, p in dist.items():
            if p <= 0:
                continue
            kl += p * math.log(p / expected)

        rec = []
        if score <= 0.05:
            rec.append(
                "Answer positions are approximately balanced. No action is required."
            )
        elif score <= 0.2:
            rec.append(
                "Correct answers appear slightly imbalanced across positions. Consider rebalancing answer positions or randomizing option order."
            )
        else:
            rec.extend(
                [
                    "Strong answer-position bias detected. Rebalance answer positions.",
                    "Randomize option order in future benchmark revisions.",
                    "Consider CircularEval-style evaluation to mitigate position biases.",
                ]
            )

        findings = []
        if score > 0.2:
            findings.append(
                {
                    "question_id": None,
                    "detector": self.NAME,
                    "severity": "critical",
                    "reason": "strong_option_imbalance",
                    "metadata": {"score": float(score)},
                }
            )
        elif score > 0.05:
            findings.append(
                {
                    "question_id": None,
                    "detector": self.NAME,
                    "severity": "warning",
                    "reason": "moderate_option_imbalance",
                    "metadata": {"score": float(score)},
                }
            )

        result = {
            **{
                "date_time": f"{datetime.now()}",
                "detector": self.NAME,
                "dataset": dataset_name,
                "num_samples": total,
                "score": float(score),
                "distribution": dist,
                "deviations": deviations,
                "chi2": float(chi2),
                "kl_divergence": float(kl),
                "recommendation": rec,
            },
            "findings": findings,
        }

        return result
