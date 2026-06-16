from typing import Dict, Any, List, Tuple
from collections import Counter
from .base_detector import BaseDetector, DetectorInputError, AnalysisContext
from datetime import datetime
from pathlib import Path
import json
import math
from vlmeval.smp.file import get_logger

logger = get_logger(__name__)


class CorrectnessAgreementDetector(BaseDetector):
    NAME = "correctness_agreement"
    DESCRIPTION = "Measure inter-model agreement on correctness (correct/incorrect) using Fleiss' Kappa."
    DEFAULT_CONFIG = {
        "low_agreement_threshold": None,
    }
    REQUIRES_MULTIPLE_MODELS = True
    SUPPORTS_COMPARISON = True

    def _fleiss_kappa(self, matrix: List[List[int]]) -> float:
        if not matrix:
            return float("nan")
        N = len(matrix)
        n = sum(matrix[0])
        if n <= 1:
            return float("nan")
        k = len(matrix[0])
        p = [0.0] * k
        for j in range(k):
            s = sum(matrix[i][j] for i in range(N))
            p[j] = s / (N * n)
        P = []
        for i in range(N):
            row = matrix[i]
            Pi = (sum(x * (x - 1) for x in row)) / (n * (n - 1))
            P.append(Pi)
        P_bar = sum(P) / N
        P_e = sum(x * x for x in p)
        if (1 - P_e) == 0:
            return float("nan")
        kappa = (P_bar - P_e) / (1 - P_e)
        return float(kappa)

    def _binary_entropy(self, p: float) -> float:
        if p <= 0.0 or p >= 1.0:
            return 0.0

        return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)

    def _compute_question_score(
        self,
        correct_full: int,
        correct_blind: int,
        n_models: int,
    ):
        if n_models <= 0:
            return 0.0

        acc_full = correct_full / n_models
        acc_blind = correct_blind / n_models

        # How much the question separates models
        discrimination = self._binary_entropy(acc_full)

        # How much image helps
        vision_gain = max(0.0, acc_full - acc_blind)

        # Require BOTH:
        # 1. model separation
        # 2. image usefulness
        score = math.sqrt(discrimination * vision_gain)

        return {
            "question_score": float(score),
            "discrimination_score": float(discrimination),
            "vision_gain": float(vision_gain),
            "accuracy_full": float(acc_full),
            "accuracy_blind": float(acc_blind),
        }

    def _extract_correctness_by_model(
        self, context: AnalysisContext
    ) -> Tuple[Dict[str, List[int]], List[str]]:
        rp = getattr(context, "result_paths", {})
        loaded = getattr(context, "loaded_results", {})

        model_keys = list(rp.keys())
        correctness_by_model = {}

        for k in model_keys:
            res = loaded.get(k, None)
            labels: List[int] = []
            if res is None:
                correctness_by_model[k] = None
                continue

            try:
                import pandas as pd

                if not isinstance(res, pd.DataFrame):
                    if (
                        isinstance(res, list)
                        and len(res) > 0
                        and isinstance(res[0], dict)
                    ):
                        res = pd.DataFrame(res)
                    elif isinstance(res, dict):
                        res = pd.DataFrame([res])

                # Determine valid options if MCQ
                valid_options = set()
                if "answer" in res.columns:
                    try:
                        valid_options = set(res["answer"].dropna().unique())
                    except Exception:
                        valid_options = set()

                for idx, row in res.iterrows():
                    # Prefer explicit correctness columns
                    hit = None
                    for hcol in ["hit", "correct", "is_correct", "isCorrect"]:
                        if hcol in res.columns:
                            hit = row.get(hcol)
                            break

                    if hit is not None:
                        # Normalize truthy values to 1/0
                        if (
                            str(hit).lower() in ["true", "1", "t", "yes"]
                            or hit is True
                            or hit == 1
                        ):
                            labels.append(1)
                        else:
                            labels.append(0)
                        continue

                    # Fall back to comparing prediction vs answer when possible
                    if "prediction" in res.columns and "answer" in res.columns:
                        pred = row.get("prediction")
                        ans = row.get("answer")
                        # Use MCQ option extraction when possible
                        pred_opt = (
                            self._extract_mcq_option(pred, valid_options)
                            if valid_options
                            else None
                        )
                        ans_norm = self._normalize_answer(ans)
                        if pred_opt and pred_opt != "Z" and ans_norm is not None:
                            # if ans is already a single-letter option, compare
                            if len(str(ans_norm)) == 1:
                                labels.append(
                                    1 if pred_opt == str(ans_norm).upper() else 0
                                )
                                continue
                        # Fallback string compare
                        if pred is None or ans is None:
                            labels.append(None)
                        else:
                            labels.append(
                                1 if self._normalize_answer(pred) == ans_norm else 0
                            )
                        continue

                    # Unknown: mark as None
                    labels.append(None)

                correctness_by_model[k] = labels

            except Exception:
                correctness_by_model[k] = None

        return correctness_by_model, model_keys

    def analyze(self, context: AnalysisContext, **kwargs) -> Dict[str, Any]:
        # support computation on a provided context (with loaded_results)
        def _compute_for_ctx(ctx: AnalysisContext):
            rp_local = getattr(ctx, "result_paths", {})
            if not rp_local or len(rp_local) < 2:
                raise DetectorInputError(
                    "CorrectnessAgreementDetector requires results from at least two models."
                )

            correctness_by_model, model_keys = self._extract_correctness_by_model(ctx)

            ref = None
            for k in model_keys:
                if correctness_by_model.get(k) is not None:
                    ref = correctness_by_model.get(k)
                    break
            if ref is None:
                raise DetectorInputError(
                    "No extractable correctness labels from provided results."
                )

            total_q = len(ref)
            matrix = []
            per_question = []
            all_correct_ids = []
            all_incorrect_ids = []
            split_ids = []

            for i in range(total_q):
                vals = []
                qmap = {}
                skip = False
                for k in model_keys:
                    arr = correctness_by_model.get(k)
                    if arr is None or i >= len(arr):
                        skip = True
                        break
                    v = arr[i]
                    if v is None:
                        skip = True
                        break
                    vals.append(v)
                    qmap[k] = bool(v)
                if skip:
                    continue

                cnt = Counter(vals)
                num_correct = cnt.get(1, 0)
                num_incorrect = cnt.get(0, 0)

                accuracy = num_correct / len(model_keys)
                matrix.append([num_correct, num_incorrect])

                if num_correct == len(model_keys):
                    consensus = "all_correct"
                    all_correct_ids.append(i)
                    difficulty = "easy"
                elif num_incorrect == len(model_keys):
                    consensus = "all_incorrect"
                    all_incorrect_ids.append(i)
                    difficulty = "hard"
                elif num_correct > num_incorrect:
                    consensus = "majority_correct"
                    difficulty = "medium"
                elif num_incorrect > num_correct:
                    consensus = "majority_incorrect"
                    difficulty = "medium"

                per_question.append(
                    {
                        "question_id": i,
                        "correctness": {k: qmap[k] for k in model_keys},
                        "consensus": consensus,
                        "difficulty_signal": difficulty,
                        "num_correct": num_correct,
                        "num_incorrect": num_incorrect,
                        "accuracy": accuracy,
                    }
                )

            if len(matrix) == 0:
                raise DetectorInputError(
                    "No fully-covered questions (all models provided correctness labels)."
                )

            kappa = self._fleiss_kappa(matrix)
            total_included = len(matrix)
            dist = {
                "all_correct": (
                    100.0 * len(all_correct_ids) / total_included
                    if total_included > 0
                    else 0
                ),
                "all_incorrect": (
                    100.0 * len(all_incorrect_ids) / total_included
                    if total_included > 0
                    else 0
                ),
                "split": (
                    100.0 * len(split_ids) / total_included if total_included > 0 else 0
                ),
            }
            solver_consensus = (
                100.0 * (len(all_correct_ids) + len(all_incorrect_ids)) / total_included
                if total_included > 0
                else 0.0
            )

            participants = [v.get("eval", None) for k, v in ctx.result_paths.items()]
            if context.mode == "full_vs_blind":
                participants += [
                    v.get("blind", None) for k, v in ctx.result_paths.items()
                ]

            report = {
                "date_time": f"{datetime.now()}",
                "detector": self.NAME,
                "dataset": getattr(ctx, "dataset_name", None),
                "participants": participants,
                "num_models": len(model_keys),
                "num_questions": total_included,
                "correctness_fleiss_kappa": (
                    float(kappa)
                    if (kappa is not None and not math.isnan(kappa))
                    else None
                ),
                "solver_consensus_percent": solver_consensus,
                "question_outcome_distribution": dist,
                "difficulty_profile": {
                    "easy": len(all_correct_ids),
                    "medium": total_included
                    - len(all_correct_ids)
                    - len(all_incorrect_ids),
                    "hard": len(all_incorrect_ids),
                },
                "warning": "Agreement computed on evaluation correctness labels. Errors in judging or answer extraction may influence results.",
            }

            # exports
            result = report
            result["_question_details"] = per_question

            if result["correctness_fleiss_kappa"] is None:
                result["agreement_score"] = None
                result["instability_score"] = None
            else:
                result["agreement_score"] = result["correctness_fleiss_kappa"]
                result["instability_score"] = (
                    1.0 - result["correctness_fleiss_kappa"]
                ) / 2.0

            return result

        # compute full
        full_ctx = AnalysisContext(
            dataset=context.dataset,
            dataset_name=context.dataset_name,
            result_paths=context.result_paths,
            loaded_results=context.full_results,
        )
        full_report = _compute_for_ctx(full_ctx)

        findings = []
        for q in full_report.get("_question_details", []):
            if q.get("consensus") == "all_incorrect":
                findings.append(
                    {
                        "question_id": q.get("question_id"),
                        "detector": self.NAME,
                        "severity": "critical",
                        "reason": "all_models_incorrect",
                    }
                )
            elif q.get("consensus") == "split":
                findings.append(
                    {
                        "question_id": q.get("question_id"),
                        "detector": self.NAME,
                        "severity": "warning",
                        "reason": "split_correctness",
                    }
                )

        self._full_findings = findings

        # if no blind or not supported -> return full
        if not getattr(context, "mode", None) == "full_vs_blind" or not getattr(
            self, "SUPPORTS_COMPARISON", False
        ):
            # set instance exports
            self._question_details = full_report.get("_question_details", [])
            return full_report

        # compute blind
        blind_ctx = AnalysisContext(
            dataset=context.dataset,
            dataset_name=context.dataset_name,
            result_paths=context.result_paths,
            loaded_results=context.blind_results,
        )
        blind_report = _compute_for_ctx(blind_ctx)

        blind_findings = []
        for q in blind_report.get("_question_details", []):
            if q.get("consensus") == "all_incorrect":
                blind_findings.append(
                    {
                        "question_id": q.get("question_id"),
                        "detector": self.NAME,
                        "severity": "critical",
                        "reason": "all_models_incorrect",
                    }
                )
            elif q.get("consensus") == "split":
                blind_findings.append(
                    {
                        "question_id": q.get("question_id"),
                        "detector": self.NAME,
                        "severity": "warning",
                        "reason": "split_correctness",
                    }
                )

        self._blind_findings = blind_findings

        full_questions = {q["question_id"]: q for q in full_report["_question_details"]}

        blind_questions = {
            q["question_id"]: q for q in blind_report["_question_details"]
        }

        question_scores = []

        all_ids = sorted(set(full_questions.keys()) & set(blind_questions.keys()))

        for qid in all_ids:

            fq = full_questions[qid]
            bq = blind_questions[qid]

            score_info = self._compute_question_score(
                correct_full=fq["num_correct"],
                correct_blind=bq["num_correct"],
                n_models=len(fq["correctness"]),
            )

            fq.update(score_info)
            bq.update(score_info)

            question_scores.append(score_info["question_score"])

        if question_scores:
            detector_score = sum(question_scores) / len(question_scores)
        else:
            detector_score = 0.0

        avg_discrimination = sum(
            q["discrimination_score"]
            for q in full_questions.values()
            if "discrimination_score" in q
        ) / len(question_scores)

        avg_vision_gain = sum(
            q["vision_gain"] for q in full_questions.values() if "vision_gain" in q
        ) / len(question_scores)

        delta = {}
        try:
            kf = full_report.get("correctness_fleiss_kappa")
            kb = blind_report.get("correctness_fleiss_kappa")
            if kf is not None and kb is not None:
                delta["correctness_kappa_delta"] = kf - kb
            cf = full_report.get("solver_consensus_percent")
            cb = blind_report.get("solver_consensus_percent")
            if cf is not None and cb is not None:
                delta["consensus_delta"] = cf - cb
        except Exception:
            pass

        self._full_question_details = full_report.get("_question_details", [])
        self._blind_question_details = blind_report.get("_question_details", [])

        full_section = {
            k: v for k, v in full_report.items() if k != "_question_details"
        }
        blind_section = {
            k: v for k, v in blind_report.items() if k != "_question_details"
        }

        full_section["detector_score"] = detector_score
        blind_section["detector_score"] = detector_score

        full_section["avg_discrimination_score"] = avg_discrimination
        blind_section["avg_discrimination_score"] = avg_discrimination

        full_section["avg_vision_gain"] = avg_vision_gain
        blind_section["avg_vision_gain"] = avg_vision_gain

        top_score = detector_score
        return {
            "full": full_section,
            "blind": blind_section,
            "delta": delta,
            "score": top_score,
        }

    def run(self, context: AnalysisContext, out_dir: str = None, **kwargs):
        res = super().run(context, out_dir=out_dir, **kwargs)
        if out_dir:
            try:
                rpt_dir = Path(out_dir) / "reports" / self.NAME
                rpt_dir.mkdir(parents=True, exist_ok=True)

                p_findings = rpt_dir / f"{self.NAME}_findings.json"
                p_findings.write_text(
                    json.dumps(
                        {
                            "findings": self.merge_findings(
                                self._full_findings, self._blind_findings
                            ),
                            "detector": self.NAME,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                p_all = rpt_dir / "all_full_infer_stat.json"
                p_all.write_text(
                    json.dumps(
                        self._full_question_details, ensure_ascii=False, indent=2
                    ),
                    encoding="utf-8",
                )
                p_all_blind = rpt_dir / "all_blind_infer_stat.json"
                p_all_blind.write_text(
                    json.dumps(
                        self._blind_question_details, ensure_ascii=False, indent=2
                    ),
                    encoding="utf-8",
                )
            except Exception:
                logger.exception("Failed to write correctness agreement reports")
        return res
