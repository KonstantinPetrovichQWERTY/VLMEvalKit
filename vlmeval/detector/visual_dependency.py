from typing import Dict, Any, List
from collections import Counter
from .base_detector import BaseDetector, DetectorInputError, AnalysisContext
from datetime import datetime
from pathlib import Path
import json
import math
from vlmeval.smp.file import get_logger

logger = get_logger(__name__)


class VisualDependencyDetector(BaseDetector):
    NAME = 'visual_dependency'
    DESCRIPTION = 'Assess how much benchmark questions depend on visual input by comparing full vs blind runs.'
    DEFAULT_CONFIG = {
        'visual_dependent_full_thresh': 0.99,
        'visual_dependent_blind_thresh': 0.01,
        'visual_supplement_gain_thresh': 0.2,
        'conflicting_gain_thresh': 0.2,
    }
    REQUIRES_FULL_RESULTS = True
    REQUIRES_BLIND_RESULTS = True
    REQUIRES_MULTIPLE_MODELS = True

    def _extract_correctness_list(self, res) -> List[int]:
        """Return list of correctness labels per sample: 1,0 or None when unknown."""
        labels = []
        if res is None:
            return None
        try:
            import pandas as pd
            if not isinstance(res, pd.DataFrame):
                if isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict):
                    res = pd.DataFrame(res)
                elif isinstance(res, dict):
                    res = pd.DataFrame([res])
        except Exception:
            # best-effort: try to iterate if it's a list
            if isinstance(res, list):
                for r in res:
                    hit = None
                    for hcol in ['hit', 'correct', 'is_correct', 'isCorrect']:
                        if isinstance(r, dict) and hcol in r:
                            hit = r.get(hcol)
                            break
                    if hit is not None:
                        if str(hit).lower() in ['true', '1', 't', 'yes'] or hit is True or hit == 1:
                            labels.append(1)
                        else:
                            labels.append(0)
                    else:
                        # fallback: compare prediction and answer if present
                        if isinstance(r, dict) and 'prediction' in r and 'answer' in r:
                            pred = self._normalize_answer(r.get('prediction'))
                            ans = self._normalize_answer(r.get('answer'))
                            if pred is None or ans is None:
                                labels.append(None)
                            else:
                                labels.append(1 if pred == ans else 0)
                        else:
                            labels.append(None)
                return labels

        try:
            # now res is DataFrame
            for idx, row in res.iterrows():
                hit = None
                for hcol in ['hit', 'correct', 'is_correct', 'isCorrect']:
                    if hcol in res.columns:
                        hit = row.get(hcol)
                        break

                if hit is not None:
                    if str(hit).lower() in ['true', '1', 't', 'yes'] or hit is True or hit == 1:
                        labels.append(1)
                    else:
                        labels.append(0)
                    continue

                if 'prediction' in res.columns and 'answer' in res.columns:
                    pred = row.get('prediction')
                    ans = row.get('answer')
                    pred_norm = self._normalize_answer(pred)
                    ans_norm = self._normalize_answer(ans)
                    if pred_norm is None or ans_norm is None:
                        labels.append(None)
                    else:
                        labels.append(1 if pred_norm == ans_norm else 0)
                    continue

                labels.append(None)
            return labels
        except Exception:
            return None

    def analyze(self, context: AnalysisContext, **kwargs) -> Dict[str, Any]:
        rp = getattr(context, 'result_paths', {})
        loaded = getattr(context, 'loaded_results', {})
        if not rp or len(rp) < 1:
            raise DetectorInputError('VisualDependencyDetector requires result files for analysis.')

        # Build mapping model -> eval_id -> variant -> key
        # Note: bench_eval also stores a 'blind' path inside the same entry and preloads it
        mapping = {}
        for key, v in rp.items():
            model = v.get('model')
            eval_id = v.get('eval_id')
            variant = v.get('variant', 'full')
            mapping.setdefault(model, {}).setdefault(eval_id, {})[variant] = key
            # If bench_eval stored a blind path in this entry, it also loaded it to loaded_results under key + '__blind'
            if v.get('blind'):
                mapping.setdefault(model, {}).setdefault(eval_id, {})['blind'] = f"{key}__blind"

        models = list(mapping.keys())
        if len(models) < 1:
            raise DetectorInputError('No models found in result paths.')

        # Find eval_ids present with both full and blind for each model
        evals_with_both_per_model = {}
        for m, evals in mapping.items():
            good = set([eid for eid, mp in evals.items() if 'full' in mp and 'blind' in mp])
            evals_with_both_per_model[m] = good

        # intersect eval_ids across models
        common = None
        for s in evals_with_both_per_model.values():
            if common is None:
                common = set(s)
            else:
                common = common.intersection(s)

        if not common:
            raise DetectorInputError('Require matching full and blind runs (same eval_id) across all models.')

        # choose an eval_id (latest sorted)
        chosen_eval = sorted(list(common))[-1]

        # prepare per-model correctness lists for full and blind
        full_by_model = {}
        blind_by_model = {}
        model_keys = []
        for m in models:
            key_map = mapping[m].get(chosen_eval, {})
            full_key = key_map.get('full')
            blind_key = key_map.get('blind')
            # If blind_key is absent but rp entry contains a blind path, try the synthetic loaded key
            if blind_key is None:
                # attempt to derive synthetic blind key
                candidate = f"{full_key}__blind" if full_key is not None else None
                if candidate and candidate in context.loaded_results:
                    blind_key = candidate
            if not full_key or not blind_key:
                raise DetectorInputError(f'Missing full/blind pair for model {m} and eval {chosen_eval}')
            model_keys.append(m)
            full_res = loaded.get(full_key)
            blind_res = loaded.get(blind_key)
            full_by_model[m] = self._extract_correctness_list(full_res)
            blind_by_model[m] = self._extract_correctness_list(blind_res)

        num_models = len(model_keys)

        # Determine total samples from first model
        ref = None
        for m in model_keys:
            if full_by_model.get(m) is not None:
                ref = full_by_model.get(m)
                break
        if ref is None:
            raise DetectorInputError('No extractable correctness labels in full runs.')

        total_q = len(ref)

        per_question = []
        counts = Counter()
        visual_gains = []

        for i in range(total_q):
            skip = False
            full_vals = []
            blind_vals = []
            for m in model_keys:
                f_arr = full_by_model.get(m)
                b_arr = blind_by_model.get(m)
                if f_arr is None or b_arr is None or i >= len(f_arr) or i >= len(b_arr):
                    skip = True
                    break
                fv = f_arr[i]
                bv = b_arr[i]
                if fv is None or bv is None:
                    skip = True
                    break
                full_vals.append(fv)
                blind_vals.append(bv)
            if skip:
                continue

            full_correct = sum(1 for x in full_vals if x == 1)
            blind_correct = sum(1 for x in blind_vals if x == 1)
            full_acc = full_correct / float(num_models)
            blind_acc = blind_correct / float(num_models)
            gain = full_acc - blind_acc
            visual_gains.append(gain)

            # classify
            full_thresh = float(self.config.get('visual_dependent_full_thresh', 0.99))
            blind_thresh = float(self.config.get('visual_dependent_blind_thresh', 0.01))
            supplement_gain = float(self.config.get('visual_supplement_gain_thresh', 0.2))
            conflict_gain = float(self.config.get('conflicting_gain_thresh', 0.2))

            if full_acc >= full_thresh and blind_acc <= blind_thresh:
                category = 'visual_dependent'
            elif blind_acc > full_acc + conflict_gain:
                category = 'conflicting_visual_signal'
            elif abs(full_acc - blind_acc) <= 1e-6:
                category = 'text_only'
            elif (full_acc - blind_acc) >= supplement_gain:
                category = 'visual_supplement'
            else:
                # small gains -> treat as text_only
                category = 'text_only'

            counts[category] += 1
            per_question.append({
                'question_id': i,
                'full_accuracy': full_acc,
                'blind_accuracy': blind_acc,
                'visual_gain': gain,
                'category': category,
            })

        included_q = len(per_question)
        avg_gain = sum(visual_gains) / included_q if included_q > 0 else 0.0

        dist = {
            'visual_dependent': 100.0 * counts.get('visual_dependent', 0) / included_q if included_q > 0 else 0.0,
            'visual_supplement': 100.0 * counts.get('visual_supplement', 0) / included_q if included_q > 0 else 0.0,
            'text_only': 100.0 * counts.get('text_only', 0) / included_q if included_q > 0 else 0.0,
            'conflicting_visual_signal': 100.0 * counts.get('conflicting_visual_signal', 0) / included_q if included_q > 0 else 0.0,
        }

        visual_dependency_score = max(0.0, min(1.0, avg_gain))

        report = {
            'date_time': f"{datetime.now()}",
            'detector': self.NAME,
            'dataset': getattr(context, 'dataset_name', None),
            'participants': [v.get('eval', None) for k, v in context.result_paths.items()] + [v.get('blind', None) for k, v in context.result_paths.items()],
            'num_models': num_models,
            'num_questions': included_q,
            'average_visual_gain': avg_gain,
            'category_distribution': dist,
            'visual_dependency_score': visual_dependency_score,
            'warning': 'Requires matching full and blind runs. Agreement computed on evaluation correctness labels.'
        }

        # cache for run()
        self._per_question = per_question

        # attach summary and findings
        try:
            summary = {'average_visual_gain': report.get('average_visual_gain'), 'visual_dependency_score': report.get('visual_dependency_score'), 'category_distribution': report.get('category_distribution')}
        except Exception:
            summary = {}
        findings = []
        for q in self._per_question:
            if q.get('category') == 'visual_dependent':
                findings.append({'question_id': q.get('question_id'), 'detector': self.NAME, 'severity': 'critical', 'reason': 'visual_dependency', 'score': q.get('visual_gain'), 'metadata': {'full_accuracy': q.get('full_accuracy'), 'blind_accuracy': q.get('blind_accuracy')}})
            elif q.get('category') == 'visual_supplement':
                findings.append({'question_id': q.get('question_id'), 'detector': self.NAME, 'severity': 'warning', 'reason': 'visual_supplement', 'score': q.get('visual_gain')})

        report['summary'] = summary
        report['findings'] = findings

        return report

    def run(self, context: AnalysisContext, out_dir: str = None, **kwargs):
        res = super().run(context, out_dir=out_dir, **kwargs)
        if out_dir and hasattr(self, '_per_question'):
            try:
                rpt_dir = Path(out_dir) / 'reports' / self.NAME
                rpt_dir.mkdir(parents=True, exist_ok=True)
                p = rpt_dir / 'visual_dependency.json'
                p.write_text(json.dumps(self._per_question, ensure_ascii=False, indent=2), encoding='utf-8')

                # export category files
                cats = {'visual_dependent': [], 'visual_supplement': [], 'text_only': [], 'conflicting_visual_signal': []}
                for q in self._per_question:
                    cats[q['category']].append(q)

                (rpt_dir / 'visual_dependent_questions.json').write_text(json.dumps(cats['visual_dependent'], ensure_ascii=False, indent=2), encoding='utf-8')
                (rpt_dir / 'visual_supplement_questions.json').write_text(json.dumps(cats['visual_supplement'], ensure_ascii=False, indent=2), encoding='utf-8')
                (rpt_dir / 'text_only_questions.json').write_text(json.dumps(cats['text_only'], ensure_ascii=False, indent=2), encoding='utf-8')
                (rpt_dir / 'conflicting_visual_signal_questions.json').write_text(json.dumps(cats['conflicting_visual_signal'], ensure_ascii=False, indent=2), encoding='utf-8')
                # write consolidated question-level statistics file
                p_all = rpt_dir / 'all_stat.json'
                p_all.write_text(json.dumps(self._per_question, ensure_ascii=False, indent=2), encoding='utf-8')
            except Exception:
                logger.exception('Failed to write visual dependency reports')
        return res
