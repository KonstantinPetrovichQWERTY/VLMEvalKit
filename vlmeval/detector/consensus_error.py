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
    NAME = 'consensus_error'
    DESCRIPTION = 'Detect samples where model consensus contradicts benchmark annotation.'
    DEFAULT_CONFIG = {
        'majority_threshold': 0.66,  # default threshold
    }
    REQUIRES_MULTIPLE_MODELS = True
    SUPPORTS_COMPARISON = True

    def analyze(self, context: AnalysisContext, **kwargs) -> Dict[str, Any]:
        # compute on a provided context
        def _compute_for_ctx(ctx: AnalysisContext):
            rp_local = getattr(ctx, 'result_paths', {})
            loaded = getattr(ctx, 'loaded_results', {})
            if not rp_local or len(rp_local) < 2:
                raise DetectorInputError('ConsensusErrorDetector requires results from at least two models.')

            model_keys = list(rp_local.keys())
            num_models = len(model_keys)

            dataset = getattr(ctx, 'dataset', None)
            ground_truths = None
            if dataset is not None and hasattr(dataset, 'data'):
                try:
                    df = dataset.data
                    if 'answer' in df.columns:
                        ground_truths = [self._normalize_answer(x) for x in list(df['answer'])]
                    else:
                        ground_truths = [None] * len(df)
                except Exception:
                    ground_truths = None

            answers_by_model, model_keys = self._get_answers_by_model(ctx)

            total_q = len(answers_by_model[model_keys[0]]) if answers_by_model.get(model_keys[0]) is not None else 0
            if total_q == 0:
                raise DetectorInputError('No extractable answers from provided results.')

            threshold = float(self.config.get('majority_threshold', 0.66))
            flagged = []
            all_q = []
            counts = {'very_high': 0, 'high': 0, 'medium': 0}
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

                disagree_with_gt = (gt is not None and majority_answer != gt)
                all_disagree = (gt is not None and all((a != gt) for a in vals))
                all_same_alt = (len(set(vals)) == 1 and (gt is None or list(set(vals))[0] != gt))

                confidence = None
                if all_same_alt and all_disagree:
                    confidence = 'very_high'
                    counts['very_high'] += 1
                elif all_disagree:
                    confidence = 'high'
                    counts['high'] += 1
                elif majority_support >= threshold and disagree_with_gt:
                    confidence = 'medium'
                    counts['medium'] += 1

                if confidence is not None and disagree_with_gt and majority_support >= threshold:
                    q = {'question_id': i, 'ground_truth': gt, 'majority_answer': majority_answer, 'majority_support': majority_support, 'confidence': confidence, 'answers': model_answers}
                    flagged.append(q)
                    all_q.append(q)
                else:
                    all_q.append({'question_id': i, 'ground_truth': gt, 'majority_answer': majority_answer if vals else None, 'majority_support': majority_support if vals else None, 'answers': model_answers})

            flagged_count = len(flagged)
            consensus_error_rate = flagged_count / float(total_q) if total_q > 0 else 0.0
            unanimous_count = counts['very_high']
            majority_count = counts['medium'] + counts['high'] + counts['very_high']

            participants = [v.get('eval', None) for k, v in ctx.result_paths.items()]
            if context.mode == 'full_vs_blind':
                participants += [v.get('blind', None) for k, v in ctx.result_paths.items()]

            report = {'date_time': f"{datetime.now()}", 'detector': self.NAME, 'participants': participants, 'num_models': num_models, 'num_questions': len(flagged), 'consensus_error_rate': consensus_error_rate, 'unanimous_consensus_error_rate': unanimous_count / float(total_q) if total_q > 0 else 0.0, 'majority_consensus_error_rate': majority_count / float(total_q) if total_q > 0 else 0.0, 'confidence_counts': counts, 'recommendation': 'Manual review of flagged samples is recommended. High/very_high confidence items should be prioritized.'}
            result = report
            result['_flagged'] = flagged
            result['_all_q'] = all_q
            # attach summary and findings
            try:
                summary = {'consensus_error_rate': result.get('consensus_error_rate'), 'unanimous_rate': result.get('unanimous_consensus_error_rate'), 'majority_rate': result.get('majority_consensus_error_rate'), 'flagged_count': len(flagged)}
            except Exception:
                summary = {}
            findings = []
            for q in flagged:
                sev = 'critical' if q.get('confidence') == 'very_high' else 'warning' if q.get('confidence') == 'high' else 'info'
                findings.append({'question_id': q.get('question_id'), 'detector': self.NAME, 'severity': sev, 'reason': 'majority_disagrees_with_ground_truth', 'score': q.get('majority_support'), 'metadata': {'confidence': q.get('confidence')}})
            result['summary'] = summary
            result['findings'] = findings

            return result

        # compute full
        full_ctx = AnalysisContext(dataset=context.dataset, dataset_name=context.dataset_name, result_paths=context.result_paths, loaded_results=context.full_results)
        full_report = _compute_for_ctx(full_ctx)

        # if no blind or comparison unsupported -> store and return full
        if not getattr(context, 'mode', None) == 'full_vs_blind' or not getattr(self, 'SUPPORTS_COMPARISON', False):
            self._flagged_questions = full_report.get('_flagged', [])
            self._all_questions = full_report.get('_all_q', [])
            return full_report

        # compute blind
        blind_ctx = AnalysisContext(dataset=context.dataset, dataset_name=context.dataset_name, result_paths=context.result_paths, loaded_results=context.blind_results)
        blind_report = _compute_for_ctx(blind_ctx)

        # compute set differences
        full_ids = set([q['question_id'] for q in full_report.get('_flagged', [])])
        blind_ids = set([q['question_id'] for q in blind_report.get('_flagged', [])])
        shared = full_ids.intersection(blind_ids)
        full_only = full_ids - blind_ids
        blind_only = blind_ids - full_ids

        delta = {'full_only_count': len(full_only), 'blind_only_count': len(blind_only), 'shared_count': len(shared)}

        # store full/blind flagged for run()
        self._flagged_questions = full_report.get('_flagged', [])
        self._blind_flagged = blind_report.get('_flagged', [])
        self._all_questions = full_report.get('_all_q', [])

        return {'full': full_report, 'blind': blind_report, 'delta': delta}

    def run(self, context: AnalysisContext, out_dir: str = None, **kwargs):
        res = super().run(context, out_dir=out_dir, **kwargs)
        if out_dir and hasattr(self, '_flagged_questions'):
            try:
                rpt_dir = Path(out_dir) / 'reports' / self.NAME
                rpt_dir.mkdir(parents=True, exist_ok=True)
                
                p = rpt_dir / 'consensus_errors.json'
                p.write_text(json.dumps(self._flagged_questions, ensure_ascii=False, indent=2), encoding='utf-8')

                p_all = rpt_dir / 'all_stat.json'
                p_all.write_text(json.dumps(self._all_questions, ensure_ascii=False, indent=2), encoding='utf-8')
                # if blind results were computed, write separate files
                if hasattr(self, '_blind_flagged'):
                    p_full = rpt_dir / 'full_consensus_errors.json'
                    p_full.write_text(json.dumps(self._flagged_questions, ensure_ascii=False, indent=2), encoding='utf-8')
                    p_blind = rpt_dir / 'blind_consensus_errors.json'
                    p_blind.write_text(json.dumps(self._blind_flagged, ensure_ascii=False, indent=2), encoding='utf-8')
                    # compute shared and visual-only
                    full_ids = set([q['question_id'] for q in self._flagged_questions])
                    blind_ids = set([q['question_id'] for q in self._blind_flagged])
                    shared_ids = full_ids.intersection(blind_ids)
                    full_only_ids = full_ids - blind_ids
                    blind_only_ids = blind_ids - full_ids
                    shared = [q for q in self._flagged_questions if q['question_id'] in shared_ids]
                    full_only = [q for q in self._flagged_questions if q['question_id'] in full_only_ids]
                    blind_only = [q for q in self._blind_flagged if q['question_id'] in blind_only_ids]
                    (rpt_dir / 'shared_consensus_errors.json').write_text(json.dumps(shared, ensure_ascii=False, indent=2), encoding='utf-8')
                    (rpt_dir / 'visual_consensus_errors.json').write_text(json.dumps(full_only, ensure_ascii=False, indent=2), encoding='utf-8')
                    (rpt_dir / 'blind_only_consensus_errors.json').write_text(json.dumps(blind_only, ensure_ascii=False, indent=2), encoding='utf-8')
            except Exception:
                logger.exception('Failed to write consensus_errors.json')
        return res
