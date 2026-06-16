from typing import Dict, Any, List
from collections import Counter
from .base_detector import BaseDetector, DetectorInputError, AnalysisContext
from datetime import datetime
from pathlib import Path
import json
import re
from vlmeval.smp.file import get_logger

logger = get_logger(__name__)


class ConsensusErrorDetector(BaseDetector):
    NAME = "consensus_error"
    DESCRIPTION = (
        "Detect samples where model consensus contradicts benchmark annotation."
    )
    DEFAULT_CONFIG = {
        "majority_threshold": 0.66,  # default threshold
    }
    REQUIRES_MULTIPLE_MODELS = True
    SUPPORTS_COMPARISON = True

    def _compute_question_suspicion(
        self,
        contradiction_full: int,
        contradiction_blind: int,
        support_full: float,
        support_blind: float,
        majority_full,
        majority_blind,
    ):
        weighted_contradiction = (
            contradiction_full * support_full + contradiction_blind * support_blind
        ) / 2.0

        answer_stability = 1.0 if majority_full == majority_blind else 0.5

        suspicion = weighted_contradiction * answer_stability

        return max(0.0, min(1.0, suspicion))

    def analyze(self, context: AnalysisContext, **kwargs) -> Dict[str, Any]:
        # compute on a provided context
        def _compute_for_ctx(ctx: AnalysisContext):
            rp_local = getattr(ctx, "result_paths", {})
            loaded = getattr(ctx, "loaded_results", {})
            if not rp_local or len(rp_local) < 2:
                raise DetectorInputError(
                    "ConsensusErrorDetector requires results from at least two models."
                )

            model_keys = list(rp_local.keys())
            num_models = len(model_keys)

            dataset = getattr(ctx, "dataset", None)
            ground_truths = None
            if dataset is not None and hasattr(dataset, "data"):
                try:
                    df = dataset.data
                    if "answer" in df.columns:
                        ground_truths = [
                            self._normalize_answer(x) for x in list(df["answer"])
                        ]
                    else:
                        ground_truths = [None] * len(df)
                except Exception:
                    ground_truths = None

            answers_by_model, model_keys = self._get_answers_by_model(ctx)

            total_q = (
                len(answers_by_model[model_keys[0]])
                if answers_by_model.get(model_keys[0]) is not None
                else 0
            )
            if total_q == 0:
                raise DetectorInputError(
                    "No extractable answers from provided results."
                )

            threshold = float(self.config.get("majority_threshold", 0.66))
            flagged = []
            all_q = []
            counts = {"very_high": 0, "high": 0, "medium": 0}
            for i in range(total_q):
                model_answers = {}
                for k in model_keys:
                    arr = answers_by_model.get(k)
                    val = None
                    if arr and i < len(arr):
                        val = arr[i]
                    model_answers[k] = val

                gt = None
                if ground_truths is not None and i < len(ground_truths):
                    gt = ground_truths[i]

                vals = [v for v in model_answers.values() if v is not None]
                if not vals:
                    continue
                cnt = Counter(vals)
                majority_answer, majority_count = cnt.most_common(1)[0]
                majority_support = majority_count / float(num_models)

                disagree_with_gt = gt is not None and majority_answer != gt
                all_disagree = gt is not None and all((a != gt) for a in vals)
                all_same_alt = len(set(vals)) == 1 and (
                    gt is None or list(set(vals))[0] != gt
                )

                confidence = None
                if all_same_alt and all_disagree:
                    confidence = "very_high"
                    counts["very_high"] += 1
                elif all_disagree:
                    confidence = "high"
                    counts["high"] += 1
                elif majority_support >= threshold and disagree_with_gt:
                    confidence = "medium"
                    counts["medium"] += 1

                if (
                    confidence is not None
                    and disagree_with_gt
                    and majority_support >= threshold
                ):
                    q = {
                        "question_id": i,
                        "ground_truth": gt,
                        "majority_answer": majority_answer,
                        "majority_support": majority_support,
                        "confidence": confidence,
                        "answers": model_answers,
                    }
                    flagged.append(q)
                    all_q.append(q)
                else:
                    all_q.append(
                        {
                            "question_id": i,
                            "ground_truth": gt,
                            "majority_answer": majority_answer if vals else None,
                            "majority_support": majority_support if vals else None,
                            "answers": model_answers,
                        }
                    )

            flagged_count = len(flagged)
            consensus_error_rate = (
                flagged_count / float(total_q) if total_q > 0 else 0.0
            )
            unanimous_count = counts["very_high"]
            majority_count = counts["medium"] + counts["high"] + counts["very_high"]

            participants = [v.get("eval", None) for k, v in ctx.result_paths.items()]
            if context.mode == "full_vs_blind":
                participants += [
                    v.get("blind", None) for k, v in ctx.result_paths.items()
                ]

            report = {
                "date_time": f"{datetime.now()}",
                "detector": self.NAME,
                "participants": participants,
                "num_models": num_models,
                "num_questions": len(flagged),
                "consensus_error_rate": consensus_error_rate,
                "unanimous_consensus_error_rate": (
                    unanimous_count / float(total_q) if total_q > 0 else 0.0
                ),
                "majority_consensus_error_rate": (
                    majority_count / float(total_q) if total_q > 0 else 0.0
                ),
                "confidence_counts": counts,
                "recommendation": "Manual review of flagged samples is recommended. High/very_high confidence items should be prioritized.",
            }
            result = report
            result["_flagged"] = flagged
            result["_all_q"] = all_q

            return result

        # compute full
        full_ctx = AnalysisContext(
            dataset=context.dataset,
            dataset_name=context.dataset_name,
            result_paths=context.result_paths,
            loaded_results=context.full_results,
        )
        full_report = _compute_for_ctx(full_ctx)
        # attach summary and findings for full report
        findings = []
        for q in full_report.get("_flagged", []):
            sev = (
                "critical"
                if q.get("confidence") == "very_high"
                else "warning" if q.get("confidence") == "high" else "info"
            )
            findings.append(
                {
                    "question_id": q.get("question_id"),
                    "detector": self.NAME,
                    "severity": sev,
                    "reason": "majority_disagrees_with_ground_truth",
                    "score": q.get("majority_support"),
                    "metadata": {
                        "confidence": q.get("confidence"),
                        "ground_truth": q.get("ground_truth"),
                        "majority_answer": q.get("majority_answer"),
                        "confidence": q.get("confidence"),
                        "answers": q.get("answers"),
                    },
                }
            )

        # store for run()/exports
        self._all_questions = full_report.get("_all_q", [])
        self._full_findings = findings

        # if no blind or comparison unsupported -> return full (include normalized score)
        if not getattr(context, "mode", None) == "full_vs_blind" or not getattr(
            self, "SUPPORTS_COMPARISON", False
        ):
            fr = {
                k: v for k, v in full_report.items() if k not in ["_flagged", "_all_q"]
            }
            # normalized severity score: consensus_error_rate (already 0..1, higher == worse)
            fr["score"] = 1.0 - float(fr.get("consensus_error_rate", 0.0))
            return fr

        # compute blind
        blind_ctx = AnalysisContext(
            dataset=context.dataset,
            dataset_name=context.dataset_name,
            result_paths=context.result_paths,
            loaded_results=context.blind_results,
        )
        blind_report = _compute_for_ctx(blind_ctx)

        # attach summary/findings to blind_report
        bfindings = []
        for q in blind_report.get("_flagged", []):
            sev = (
                "critical"
                if q.get("confidence") == "very_high"
                else "warning" if q.get("confidence") == "high" else "info"
            )
            bfindings.append(
                {
                    "question_id": q.get("question_id"),
                    "detector": self.NAME,
                    "severity": sev,
                    "reason": "majority_disagrees_with_ground_truth",
                    "score": q.get("majority_support"),
                    "metadata": {
                        "confidence": q.get("confidence"),
                        "ground_truth": q.get("ground_truth"),
                        "majority_answer": q.get("majority_answer"),
                        "confidence": q.get("confidence"),
                        "answers": q.get("answers"),
                    },
                }
            )
        self._blind_all_questions = blind_report.get("_all_q", [])
        self._blind_findings = bfindings

        full_questions = {q["question_id"]: q for q in self._all_questions}

        blind_questions = {q["question_id"]: q for q in self._blind_all_questions}

        question_scores = []
        question_suspicions = []

        all_question_ids = sorted(
            set(full_questions.keys()) | set(blind_questions.keys())
        )

        for qid in all_question_ids:

            fq = full_questions.get(qid, {})
            bq = blind_questions.get(qid, {})

            gt = fq.get("ground_truth", bq.get("ground_truth"))

            majority_full = fq.get("majority_answer")
            majority_blind = bq.get("majority_answer")

            support_full = (
                fq.get("majority_support")
                if fq.get("majority_support") is not None
                else 0.0
            )

            support_blind = (
                bq.get("majority_support")
                if bq.get("majority_support") is not None
                else 0.0
            )

            contradiction_full = 1 if (gt is not None and majority_full != gt) else 0

            contradiction_blind = 1 if (gt is not None and majority_blind != gt) else 0

            suspicion = self._compute_question_suspicion(
                contradiction_full=contradiction_full,
                contradiction_blind=contradiction_blind,
                support_full=support_full,
                support_blind=support_blind,
                majority_full=majority_full,
                majority_blind=majority_blind,
            )

            score = 1.0 - suspicion

            question_suspicions.append(suspicion)
            question_scores.append(score)

            if qid in full_questions:
                full_questions[qid]["question_suspicion"] = suspicion
                full_questions[qid]["question_score"] = score

            if qid in blind_questions:
                blind_questions[qid]["question_suspicion"] = suspicion
                blind_questions[qid]["question_score"] = score

        if question_scores:
            detector_score = sum(question_scores) / len(question_scores)
        else:
            detector_score = 1.0

        # compute set differences
        full_ids = set([q["question_id"] for q in full_report.get("_flagged", [])])
        blind_ids = set([q["question_id"] for q in blind_report.get("_flagged", [])])
        shared = full_ids.intersection(blind_ids)
        full_only = full_ids - blind_ids
        blind_only = blind_ids - full_ids

        delta = {
            "full_only_count": len(full_only),
            "blind_only_count": len(blind_only),
            "shared_count": len(shared),
        }

        full_section = {
            k: v for k, v in full_report.items() if k not in ["_flagged", "_all_q"]
        }
        blind_section = {
            k: v for k, v in blind_report.items() if k not in ["_flagged", "_all_q"]
        }

        full_section["detector_score"] = detector_score
        blind_section["detector_score"] = detector_score

        full_section["avg_question_suspicion"] = (
            sum(question_suspicions) / len(question_suspicions)
            if question_suspicions
            else 0.0
        )

        blind_section["avg_question_suspicion"] = (
            sum(question_suspicions) / len(question_suspicions)
            if question_suspicions
            else 0.0
        )

        # top-level normalized score uses full-run consensus error rate (0..1, higher == worse)
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

                p_all = rpt_dir / "all_full_infer_stat.json"
                p_all.write_text(
                    json.dumps(self._all_questions, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                p_all_blind = rpt_dir / "all_blind_infer_stat.json"
                p_all_blind.write_text(
                    json.dumps(self._blind_all_questions, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

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
            except Exception:
                logger.exception("Failed to write consensus_errors.json")
        return res
