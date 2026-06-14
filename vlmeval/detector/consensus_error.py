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
        'max_export_questions': 1000,
    }

    def analyze(self, context: AnalysisContext, **kwargs) -> Dict[str, Any]:
        rp = getattr(context, 'result_paths', {})
        loaded = getattr(context, 'loaded_results', {})
        if not rp or len(rp) < 2:
            raise DetectorInputError('ConsensusErrorDetector requires results from at least two models.')

        model_keys = list(rp.keys())
        num_models = len(model_keys)

        # Prefer dataset-provided ground truth if available
        dataset = getattr(context, 'dataset', None)
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

        answers_by_model, model_keys = self._get_answers_by_model(context)

        total_q = len(answers_by_model[model_keys[0]]) if answers_by_model.get(model_keys[0]) is not None else 0
        if total_q == 0:
            raise DetectorInputError('No extractable answers from provided results.')

        # Build flagged list
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

            # ground truth for this sample
            gt = None
            if ground_truths is not None and i < len(ground_truths):
                gt = ground_truths[i]

            # majority
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
                q = {
                    'question_id': i,
                    'ground_truth': gt,
                    'majority_answer': majority_answer,
                    'majority_support': majority_support,
                    'confidence': confidence,
                    'answers': model_answers,
                }
                flagged.append(q)
                all_q.append(q)
            else:
                all_q.append({
                    'question_id': i,
                    'ground_truth': gt,
                    'majority_answer': majority_answer if vals else None,
                    'majority_support': majority_support if vals else None,
                    'answers': model_answers,
                })

        # summary metrics
        flagged_count = len(flagged)
        consensus_error_rate = flagged_count / float(total_q) if total_q > 0 else 0.0
        unanimous_count = counts['very_high']
        majority_count = counts['medium'] + counts['high'] + counts['very_high']

        report = {
            'date_time': f"{datetime.now()}",
            'detector': self.NAME,
            'participants': [rp[k]['model'] + '__' + rp[k]['eval_id'] for k in model_keys],
            'num_models': num_models,
            'num_questions': len(flagged),
            'consensus_error_rate': consensus_error_rate,
            'unanimous_consensus_error_rate': unanimous_count / float(total_q) if total_q > 0 else 0.0,
            'majority_consensus_error_rate': majority_count / float(total_q) if total_q > 0 else 0.0,
            'confidence_counts': counts,
            'recommendation': 'Manual review of flagged samples is recommended. High/very_high confidence items should be prioritized.'
        }

        # store flagged for run() to write
        self._flagged_questions = flagged
        self._all_questions = all_q

        return report

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
            except Exception:
                logger.exception('Failed to write consensus_errors.json')
        return res
