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
    NAME = 'correctness_agreement'
    DESCRIPTION = 'Measure inter-model agreement on correctness (correct/incorrect) using Fleiss\' Kappa.'
    DEFAULT_CONFIG = {
        'low_agreement_threshold': None,
        'export_max_questions': 1000,
    }

    def _fleiss_kappa(self, matrix: List[List[int]]) -> float:
        if not matrix:
            return float('nan')
        N = len(matrix)
        n = sum(matrix[0])
        if n <= 1:
            return float('nan')
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
            return float('nan')
        kappa = (P_bar - P_e) / (1 - P_e)
        return float(kappa)

    def _extract_correctness_by_model(self, context: AnalysisContext) -> Tuple[Dict[str, List[int]], List[str]]:
        rp = getattr(context, 'result_paths', {})
        loaded = getattr(context, 'loaded_results', {})

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
                    if isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict):
                        res = pd.DataFrame(res)
                    elif isinstance(res, dict):
                        res = pd.DataFrame([res])

                # Determine valid options if MCQ
                valid_options = set()
                if 'answer' in res.columns:
                    try:
                        valid_options = set(res['answer'].dropna().unique())
                    except Exception:
                        valid_options = set()

                for idx, row in res.iterrows():
                    # Prefer explicit correctness columns
                    hit = None
                    for hcol in ['hit', 'correct', 'is_correct', 'isCorrect']:
                        if hcol in res.columns:
                            hit = row.get(hcol)
                            break

                    if hit is not None:
                        # Normalize truthy values to 1/0
                        if str(hit).lower() in ['true', '1', 't', 'yes'] or hit is True or hit == 1:
                            labels.append(1)
                        else:
                            labels.append(0)
                        continue

                    # Fall back to comparing prediction vs answer when possible
                    if 'prediction' in res.columns and 'answer' in res.columns:
                        pred = row.get('prediction')
                        ans = row.get('answer')
                        # Use MCQ option extraction when possible
                        pred_opt = self._extract_mcq_option(pred, valid_options) if valid_options else None
                        ans_norm = self._normalize_answer(ans)
                        if pred_opt and pred_opt != 'Z' and ans_norm is not None:
                            # if ans is already a single-letter option, compare
                            if len(str(ans_norm)) == 1:
                                labels.append(1 if pred_opt == str(ans_norm).upper() else 0)
                                continue
                        # Fallback string compare
                        if pred is None or ans is None:
                            labels.append(None)
                        else:
                            labels.append(1 if self._normalize_answer(pred) == ans_norm else 0)
                        continue

                    # Unknown: mark as None
                    labels.append(None)

                correctness_by_model[k] = labels

            except Exception:
                correctness_by_model[k] = None

        return correctness_by_model, model_keys

    def analyze(self, context: AnalysisContext, **kwargs) -> Dict[str, Any]:
        rp = getattr(context, 'result_paths', {})
        if not rp or len(rp) < 2:
            raise DetectorInputError('CorrectnessAgreementDetector requires results from at least two models.')

        # Extract correctness labels per model (1=correct,0=incorrect)
        correctness_by_model, model_keys = self._extract_correctness_by_model(context)

        # Determine number of samples from first model that has data
        ref = None
        for k in model_keys:
            if correctness_by_model.get(k) is not None:
                ref = correctness_by_model.get(k)
                break
        if ref is None:
            raise DetectorInputError('No extractable correctness labels from provided results.')

        total_q = len(ref)
        # Build matrix rows only for fully-covered questions (all models have non-None label)
        matrix: List[List[int]] = []
        per_question: List[Dict[str, Any]] = []
        all_correct_ids: List[int] = []
        all_incorrect_ids: List[int] = []
        split_ids: List[int] = []

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
            matrix.append([num_correct, num_incorrect])

            # consensus category
            if num_correct == len(model_keys):
                consensus = 'all_correct'
                all_correct_ids.append(i)
                difficulty = 'easy'
            elif num_incorrect == len(model_keys):
                consensus = 'all_incorrect'
                all_incorrect_ids.append(i)
                difficulty = 'hard'
            elif num_correct > num_incorrect:
                consensus = 'majority_correct'
                difficulty = 'medium'
            elif num_incorrect > num_correct:
                consensus = 'majority_incorrect'
                difficulty = 'medium'
            else:
                consensus = 'split'
                split_ids.append(i)
                difficulty = 'medium'

            per_question.append({
                'question_id': i,
                'correctness': {k: qmap[k] for k in model_keys},
                'consensus': consensus,
                'difficulty_signal': difficulty,
            })

        if len(matrix) == 0:
            raise DetectorInputError('No fully-covered questions (all models provided correctness labels).')

        kappa = self._fleiss_kappa(matrix)

        # Aggregate distributions
        total_included = len(matrix)
        dist = {
            'all_correct': 100.0 * len(all_correct_ids) / total_included if total_included > 0 else 0,
            'all_incorrect': 100.0 * len(all_incorrect_ids) / total_included if total_included > 0 else 0,
            'split': 100.0 * len(split_ids) / total_included if total_included > 0 else 0,
        }

        # Solver consensus score: percent where all models agree (either all_correct or all_incorrect)
        solver_consensus = 100.0 * (len(all_correct_ids) + len(all_incorrect_ids)) / total_included if total_included > 0 else 0.0

        report = {
            'date_time': f"{datetime.now()}",
            'detector': self.NAME,
            'dataset': getattr(context, 'dataset_name', None),
            'participants': [v.get('eval', 'unknown') for k, v in context.result_paths.items()],
            'num_models': len(model_keys),
            'num_questions': total_included,
            'correctness_fleiss_kappa': float(kappa) if (kappa is not None and not math.isnan(kappa)) else None,
            'solver_consensus_percent': solver_consensus,
            'question_outcome_distribution': dist,
            'difficulty_profile': {
                'easy': len(all_correct_ids),
                'medium': total_included - len(all_correct_ids) - len(all_incorrect_ids),
                'hard': len(all_incorrect_ids),
            },
            'warning': 'Agreement computed on evaluation correctness labels. Errors in judging or answer extraction may influence results.'
        }

        # Prepare exports
        self._all_correct = all_correct_ids
        self._all_incorrect = all_incorrect_ids
        self._split = split_ids

        # also store detailed lists for run()
        self._question_details = per_question

        # Detector scores
        if report['correctness_fleiss_kappa'] is None:
            report['agreement_score'] = None
            report['instability_score'] = None
        else:
            report['agreement_score'] = report['correctness_fleiss_kappa']
            report['instability_score'] = (1.0 - report['correctness_fleiss_kappa']) / 2.0

        return report

    def run(self, context: AnalysisContext, out_dir: str = None, **kwargs):
        res = super().run(context, out_dir=out_dir, **kwargs)
        if out_dir and hasattr(self, '_question_details'):
            try:
                rpt_dir = Path(out_dir) / 'reports' / self.NAME
                rpt_dir.mkdir(parents=True, exist_ok=True)
                maxq = int(self.config.get('export_max_questions', 1000))

                p_all = rpt_dir / 'all_correct_questions.json'
                all_entries = [q for q in self._question_details if q['consensus'] == 'all_correct'][:maxq]
                p_all.write_text(json.dumps(all_entries, ensure_ascii=False, indent=2), encoding='utf-8')

                p_bad = rpt_dir / 'all_incorrect_questions.json'
                bad_entries = [q for q in self._question_details if q['consensus'] == 'all_incorrect'][:maxq]
                p_bad.write_text(json.dumps(bad_entries, ensure_ascii=False, indent=2), encoding='utf-8')

                p_split = rpt_dir / 'split_questions.json'
                split_entries = [q for q in self._question_details if q['consensus'] == 'split'][:maxq]
                p_split.write_text(json.dumps(split_entries, ensure_ascii=False, indent=2), encoding='utf-8')
                # write all_stat question-level dump
                p_all = rpt_dir / 'all_stat.json'
                p_all.write_text(json.dumps(self._question_details, ensure_ascii=False, indent=2), encoding='utf-8')
            except Exception:
                logger.exception('Failed to write correctness agreement reports')
        return res
