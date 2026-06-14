import string
from typing import Dict, Any, List
import re

from .base_detector import AnalysisContext, BaseDetector, DetectorInputError
from datetime import datetime
import math
from vlmeval.smp.file import get_logger
logger = get_logger(__name__)
from pathlib import Path
import json


class FleissKappaAgreementDetector(BaseDetector):
    NAME = 'fleiss_kappa_agreement'
    DESCRIPTION = 'Measure inter-model agreement using Fleiss\' Kappa.'
    DEFAULT_CONFIG = {
        'full_threshold': 1.0,
        'complete_threshold': 1,  # max_count == 1 => complete disagreement
    }

    def _fleiss_kappa(self, matrix: List[List[int]]) -> float:
        # matrix: rows=subjects, cols=categories, entries=counts; n raters per subject
        if not matrix:
            return float('nan')
        N = len(matrix)
        n = sum(matrix[0])
        if n <= 1:
            return float('nan')
        # p_j
        k = len(matrix[0])
        p = [0.0] * k
        for j in range(k):
            s = sum(matrix[i][j] for i in range(N))
            p[j] = s / (N * n)
        # P_i
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

    def analyze(self, context: AnalysisContext, **kwargs) -> Dict[str, Any]:
        rp = getattr(context, 'result_paths', {})
        if not rp or len(rp) < 2:
            raise DetectorInputError('FleissKappaAgreementDetector requires results from at least two models.')

        dataset = getattr(context, 'dataset')
        if dataset.TYPE != 'MCQ':
            raise DetectorInputError('FleissKappaAgreementDetector only supports MCQ datasets.')
        
        answers_by_model, model_keys = self._get_answers_by_model(context)

        # Determine number of samples
        lengths = [len(v) for v in answers_by_model.values() if v is not None]
        if not lengths:
            raise DetectorInputError('No extractable answers from provided results.')
        num_samples = max(lengths)

        matrix = []
        categories_set = set()
        per_question_details = []
        
        for idx in range(num_samples):
            answers = {}
            skip = False
            
            for k in model_keys:
                arr = answers_by_model.get(k)
                if arr is None or idx >= len(arr):
                    skip = True
                    break
                ans = arr[idx]
                if ans is None:
                    skip = True
                    break
                answers[k] = ans
                categories_set.add(ans)
            
            if skip:
                continue
            # Count per category
            counts_map = {}
            for v in answers.values():
                counts_map[v] = counts_map.get(v, 0) + 1
            
            per_question_details.append({
                'idx': idx, 
                'answers': answers, 
                'counts_map': counts_map,
            })

        # Collect global categories
        all_categories = sorted(categories_set)
        if not all_categories:
            raise DetectorInputError('No answer categories found across models.')

        # Build final matrix rows
        for q in per_question_details:
            row = [q['counts_map'].get(cat, 0) for cat in all_categories]
            matrix.append(row)

        if len(matrix) == 0:
            raise DetectorInputError('No fully-covered questions (all models answered).')

        # Compute Fleiss kappa
        kappa = self._fleiss_kappa(matrix)

        # Agreement distribution
        n_raters = len(model_keys)
        full = 0
        complete = 0
        partial = 0
        q_stats = []
        
        for i, row in enumerate(matrix):
            max_count = max(row)
            if max_count == n_raters:
                full += 1
                level = 'full'
            elif max_count == 1:
                complete += 1
                level = 'complete'
            else:
                partial += 1
                level = 'partial'
            
            q = per_question_details[i]
            q_stats.append({
                'question_idx': q['idx'], 
                'answers': q['answers'], 
                'agreement_level': level,
            })

        total_q = len(matrix)
        dist = {
            'full_agreement': 100.0 * full / total_q if total_q > 0 else 0,
            'partial_agreement': 100.0 * partial / total_q if total_q > 0 else 0,
            'complete_disagreement': 100.0 * complete / total_q if total_q > 0 else 0,
        }

        # Pairwise agreement matrix
        pairwise = {}
        for i, k1 in enumerate(model_keys):
            pairwise[k1] = {}
            for j, k2 in enumerate(model_keys):
                if i == j:
                    pairwise[k1][k2] = 1.0
                    continue
                agree = 0
                total = 0
                arr1 = answers_by_model.get(k1) or []
                arr2 = answers_by_model.get(k2) or []
                mlen = min(len(arr1), len(arr2))
                for idx in range(mlen):
                    a1 = arr1[idx] if idx < len(arr1) else None
                    a2 = arr2[idx] if idx < len(arr2) else None
                    if a1 is None or a2 is None:
                        continue
                    total += 1
                    if a1 == a2:
                        agree += 1
                pairwise[k1][k2] = (agree / total) if total > 0 else None

        # Assemble report
        report = {
            'date_time': f"{datetime.now()}",
            'detector': self.NAME,
            'dataset': getattr(context, 'dataset_name', None),
            'participants': [v.get('eval', 'unknown') for k, v in context.result_paths.items()],
            'num_questions': total_q,
            'num_models': n_raters,
            'fleiss_kappa': float(kappa) if (kappa is not None and not math.isnan(kappa)) else None,
            'agreement_distribution': dist,
            'pairwise_agreement': pairwise,
            'warning': 'Agreement computed on MCQ option letters (A/B/C/D/...). Predictions without clear options are marked as "Z" (unknown).'
        }

        # Detector score and risk
        if report['fleiss_kappa'] is None:
            report['agreement_score'] = None
            report['risk_score'] = None
        else:
            report['agreement_score'] = report['fleiss_kappa']
            # Normalize from [-1,1] to [0,1] risk = 1 - (k+1)/2 = (1 - k)/2
            report['risk_score'] = (1.0 - report['fleiss_kappa']) / 2.0

        # Build low-agreement questions list (complete disagreement always included)
        low_questions = []
        for i, q in enumerate(per_question_details):
            row = matrix[i]
            n_raters = sum(row)
            # vote_distribution
            vote_distribution = {cat: q['counts_map'].get(cat, 0) for cat in all_categories}
            max_count = max(row)
            # majority answer (null if tie)
            majority_candidates = [cat for cat, cnt in vote_distribution.items() if cnt == max_count]
            majority_answer = majority_candidates[0] if len(majority_candidates) == 1 else None
            agreement_ratio = (max_count / n_raters) if n_raters > 0 else 0.0
            # agreement level
            if max_count == n_raters:
                level = 'full'
            elif max_count == 1:
                level = 'complete_disagreement'
            else:
                level = 'partial'

            entry = {
                'question_id': q['idx'],
                'answers': q['answers'],
                'agreement_level': level,
                'vote_distribution': vote_distribution,
                'majority_answer': majority_answer,
                'agreement_ratio': agreement_ratio,
            }

            # include if complete_disagreement or below optional threshold
            include = False
            if level == 'complete_disagreement':
                include = True
            else:
                thresh = self.config.get('low_agreement_threshold', None)
                if thresh is not None and agreement_ratio < float(thresh):
                    include = True

            if include:
                low_questions.append(entry)

        # store low questions on the instance for run() to write separately
        self._low_agreement_questions = low_questions
        self.q_stats = q_stats

        return report

    def run(self, context, out_dir: str = None, **kwargs):
        # Call base run() to write the main detector report (keeps existing behavior)
        result = super().run(context, out_dir=out_dir, **kwargs)
        # Write low agreement questions file into reports/low_agreement_questions.json
        if out_dir is not None and hasattr(self, '_low_agreement_questions'):
            try:
                rpt_dir = Path(out_dir) / 'reports' / self.NAME
                rpt_dir.mkdir(parents=True, exist_ok=True)
                p = rpt_dir / 'low_agreement_questions.json'
                p.write_text(json.dumps(self._low_agreement_questions, ensure_ascii=False, indent=2), encoding='utf-8')
                
                p_all = rpt_dir / 'all_stat.json'
                p_all.write_text(json.dumps(self.q_stats, ensure_ascii=False, indent=2), encoding='utf-8')
            except Exception:
                logger.exception('Failed to write low_agreement_questions.json')
        return result