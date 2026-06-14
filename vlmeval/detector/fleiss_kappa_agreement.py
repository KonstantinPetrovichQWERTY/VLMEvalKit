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
    REQUIRES_MULTIPLE_MODELS = True
    SUPPORTS_COMPARISON = True

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
        # helper that computes report for a given AnalysisContext (allows full/blind)
        def _compute_for_ctx(ctx: AnalysisContext):
            rp_local = getattr(ctx, 'result_paths', {})
            if not rp_local or len(rp_local) < 2:
                raise DetectorInputError('FleissKappaAgreementDetector requires results from at least two models.')

            dataset = getattr(ctx, 'dataset')
            if dataset.TYPE != 'MCQ':
                raise DetectorInputError('FleissKappaAgreementDetector only supports MCQ datasets.')

            answers_by_model, model_keys = self._get_answers_by_model(ctx)

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
                counts_map = {}
                for v in answers.values():
                    counts_map[v] = counts_map.get(v, 0) + 1
                per_question_details.append({'idx': idx, 'answers': answers, 'counts_map': counts_map})

            all_categories = sorted(categories_set)
            if not all_categories:
                raise DetectorInputError('No answer categories found across models.')

            for q in per_question_details:
                row = [q['counts_map'].get(cat, 0) for cat in all_categories]
                matrix.append(row)

            if len(matrix) == 0:
                raise DetectorInputError('No fully-covered questions (all models answered).')

            kappa = self._fleiss_kappa(matrix)

            n_raters = len(model_keys)
            full = complete = partial = 0
            q_stats_local = []
            for i, row in enumerate(matrix):
                max_count = max(row)
                if max_count == n_raters:
                    level = 'full'
                    full += 1
                elif max_count == 1:
                    level = 'complete'
                    complete += 1
                else:
                    level = 'partial'
                    partial += 1
                q = per_question_details[i]
                q_stats_local.append({'question_idx': q['idx'], 'answers': q['answers'], 'agreement_level': level})

            total_q = len(matrix)
            dist = {'full_agreement': 100.0 * full / total_q if total_q > 0 else 0, 'partial_agreement': 100.0 * partial / total_q if total_q > 0 else 0, 'complete_disagreement': 100.0 * complete / total_q if total_q > 0 else 0}

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


            participants = [v.get('eval', None) for k, v in ctx.result_paths.items()]
            if context.mode == 'full_vs_blind':
                participants += [v.get('blind', None) for k, v in ctx.result_paths.items()]

            report = {'date_time': f"{datetime.now()}", 'detector': self.NAME, 'dataset': getattr(ctx, 'dataset_name', None), 'participants': participants, 'num_questions': total_q, 'num_models': n_raters, 'fleiss_kappa': float(kappa) if (kappa is not None and not math.isnan(kappa)) else None, 'agreement_distribution': dist, 'pairwise_agreement': pairwise, 'warning': 'Agreement computed on MCQ option letters (A/B/C/D/...). Predictions without clear options are marked as "Z" (unknown).'}

            if report['fleiss_kappa'] is None:
                report['agreement_score'] = None
                report['risk_score'] = None
            else:
                report['agreement_score'] = report['fleiss_kappa']
                report['risk_score'] = (1.0 - report['fleiss_kappa']) / 2.0

            # build low-agreement list
            low_questions = []
            for i, q in enumerate(per_question_details):
                row = matrix[i]
                n_raters_local = sum(row)
                vote_distribution = {cat: q['counts_map'].get(cat, 0) for cat in all_categories}
                max_count = max(row)
                majority_candidates = [cat for cat, cnt in vote_distribution.items() if cnt == max_count]
                majority_answer = majority_candidates[0] if len(majority_candidates) == 1 else None
                agreement_ratio = (max_count / n_raters_local) if n_raters_local > 0 else 0.0
                if max_count == n_raters_local:
                    level = 'full'
                elif max_count == 1:
                    level = 'complete_disagreement'
                else:
                    level = 'partial'

                entry = {'question_id': q['idx'], 'answers': q['answers'], 'agreement_level': level, 'vote_distribution': vote_distribution, 'majority_answer': majority_answer, 'agreement_ratio': agreement_ratio}
                include = False
                if level == 'complete_disagreement':
                    include = True
                else:
                    thresh = self.config.get('low_agreement_threshold', None)
                    if thresh is not None and agreement_ratio < float(thresh):
                        include = True
                if include:
                    low_questions.append(entry)

            return {'report': report, '_low_questions': low_questions, '_q_stats': q_stats_local}

        # compute full
        full_ctx = AnalysisContext(dataset=context.dataset, dataset_name=context.dataset_name, result_paths=context.result_paths, loaded_results=context.full_results)
        full_out = _compute_for_ctx(full_ctx)
        full_report = full_out['report']

        # preserve full-run exports
        self._low_agreement_questions = full_out.get('_low_questions', [])
        self.q_stats = full_out.get('_q_stats', [])

        # if no blind-support or not in comparison mode, return full report
        if not getattr(context, 'mode', None) == 'full_vs_blind' or not getattr(self, 'SUPPORTS_COMPARISON', False):
            return full_report

        # compute blind
        blind_ctx = AnalysisContext(dataset=context.dataset, dataset_name=context.dataset_name, result_paths=context.result_paths, loaded_results=context.blind_results)
        blind_out = _compute_for_ctx(blind_ctx)
        blind_report = blind_out['report']

        delta = {}
        try:
            kf = full_report.get('fleiss_kappa')
            kb = blind_report.get('fleiss_kappa')
            if kf is not None and kb is not None:
                delta['fleiss_kappa_delta'] = kf - kb
            # distribution deltas
            for key in ['full_agreement', 'partial_agreement', 'complete_disagreement']:
                df = full_report.get('agreement_distribution', {}).get(key)
                db = blind_report.get('agreement_distribution', {}).get(key)
                if df is not None and db is not None:
                    delta[f'dist_delta_{key}'] = df - db
        except Exception:
            pass

        return {'full': full_report, 'blind': blind_report, 'delta': delta}

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