from typing import Dict, Any, List, Tuple, Optional

from tqdm import tqdm
from .base_detector import BaseDetector, AnalysisContext, DetectorInputError
from datetime import datetime
from pathlib import Path
import json
import math
import re
from vlmeval.smp.file import get_logger

logger = get_logger(__name__)



class DistractorSimilarityDetector(BaseDetector):
    NAME = 'distractor_similarity'
    DESCRIPTION = 'Detect semantically-similar MCQ distractors within dataset.'
    DEFAULT_CONFIG = {
        'backend': 'auto',  # auto -> try sentence_transformers, fallback to difflib
        'threshold_warning': 0.75,
        'threshold_critical': 0.90,
    }

    REQUIRES_FULL_RESULTS = False
    REQUIRES_BLIND_RESULTS = False
    REQUIRES_MULTIPLE_MODELS = False
    SUPPORTS_COMPARISON = False

    def _find_options_columns(self, df) -> List[List[str]]:
        # Attempt common column names
        if 'options' in df.columns:
            return [list(r) if r is not None else [] for r in df['options']]
        if 'choices' in df.columns:
            return [list(r) if r is not None else [] for r in df['choices']]

        # Look for A/B/C/D style columns
        letter_cols = [c for c in df.columns if re.fullmatch(r'^[A-Z]$', c, re.IGNORECASE)]
        if letter_cols:
            return [[row[c] for c in letter_cols] for _, row in df.iterrows()]

        return []

    def _get_correct_index(self, ans_val, options: List[str]) -> Optional[int]:
        if ans_val is None:
            return None
        # direct match
        try:
            normalized_options = [s.strip().lower() if s is not None else '' for s in options]
            if isinstance(ans_val, str):
                a = ans_val.strip()
                # letter map A/B/C -> index
                if re.fullmatch(r'^[A-Za-z]$', a):
                    idx = ord(a.upper()) - ord('A')
                    if 0 <= idx < len(options):
                        return idx
                # numeric index string
                if re.fullmatch(r'^\d+$', a):
                    ni = int(a)
                    if 0 <= ni < len(options):
                        return ni
                # exact text match
                low = a.lower()
                if low in normalized_options:
                    return normalized_options.index(low)
            else:
                # numeric
                if isinstance(ans_val, (int, float)) and 0 <= int(ans_val) < len(options):
                    return int(ans_val)
        except Exception:
            return None
        return None

    def _embed_backend(self):
        cfg = self.config.get('backend', 'auto')
        if cfg in ('st', 'sentence_transformers', 'auto'):
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                return ('st', model)
            except Exception:
                pass
        # fallback: use difflib
        return ('difflib', None)

    def _pairwise_similarities(self, texts: List[str], backend_info) -> List[Tuple[int, int, float]]:
        kind, model = backend_info
        sims = []
        n = len(texts)
        if kind == 'st' and model is not None:
            import numpy as np
            embs = model.encode(texts, convert_to_numpy=True)
            # normalize
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embs = embs / norms
            for i in range(n):
                for j in range(i + 1, n):
                    sim = float(np.dot(embs[i], embs[j]))
                    sims.append((i, j, sim))
            return sims
        else:
            # difflib fallback
            from difflib import SequenceMatcher
            for i in tqdm(range(n), desc='Computing pairwise similarities'):
                for j in range(i + 1, n):
                    a = texts[i] or ''
                    b = texts[j] or ''
                    sim = SequenceMatcher(None, a, b).ratio()
                    sims.append((i, j, float(sim)))
            return sims

    def analyze(self, context: AnalysisContext, **kwargs) -> Dict[str, Any]:
        dataset = getattr(context, 'dataset', None)
        if dataset is None or not hasattr(dataset, 'data'):
            raise DetectorInputError('Dataset not available in context')

        df = dataset.data
        options_rows = self._find_options_columns(df)
        if not options_rows:
            raise DetectorInputError('No options columns found in dataset')

        backend = self._embed_backend()
        thresh_warn = float(self.config.get('threshold_warning', 0.75))
        thresh_crit = float(self.config.get('threshold_critical', 0.90))

        per_q_reports = []
        max_pairs = []
        duplicate_count = 0

        MISSING_TOKENS = set(['nan', 'none', 'n/a', 'na', ''])

        for idx, opts in enumerate(options_rows):
            # normalize options as strings
            opts = [o if o is not None else '' for o in opts]
            norm_opts = [re.sub(r"\s+", ' ', str(o).strip()) for o in opts]
            # identify non-missing options (ignore placeholder tokens like 'nan')
            lowered = [o.lower() for o in norm_opts]
            idx_map = [i for i, v in enumerate(lowered) if v not in MISSING_TOKENS]
            consider_only = [lowered[i] for i in idx_map]
            has_dup = len(set(consider_only)) < len(consider_only) if consider_only else False
            if has_dup:
                duplicate_count += 1

            # compute similarities over non-missing options only
            if len(idx_map) < 2:
                sims = []
            else:
                texts = [norm_opts[i] for i in idx_map]
                sims_local = self._pairwise_similarities(texts, backend)
                # map local pair indices back to original option indices
                sims = [(idx_map[a], idx_map[b], s) for (a, b, s) in sims_local]
            if not sims:
                max_sim = 0.0
                mean_sim = 0.0
            else:
                vals = [s for (_, _, s) in sims]
                max_sim = max(vals)
                mean_sim = float(sum(vals) / len(vals))

            # correct option similarity
            correct_val = None
            if 'answer' in df.columns:
                correct_val = df['answer'].iloc[idx]
            correct_idx = self._get_correct_index(correct_val, norm_opts) if correct_val is not None else None
            max_correct_sim = None
            most_similar_pair = None
            if sims:
                # find most similar pair indices
                imax = max(sims, key=lambda x: x[2])
                most_similar_pair = (imax[0], imax[1])
            if correct_idx is not None and sims:
                # compute similarities between correct and others
                cs = [s for (i, j, s) in sims if i == correct_idx or j == correct_idx]
                if cs:
                    max_correct_sim = max(cs)

            # severity
            if max_sim >= thresh_crit:
                severity = 'critical'
            elif max_sim >= thresh_warn:
                severity = 'warning'
            else:
                severity = 'healthy'

            per_q_reports.append({'question_id': idx, 'question': df.get('question', df.get('question_text', '')).iloc[idx] if 'question' in df.columns or 'question_text' in df.columns else None, 'options': {chr(ord('A') + i): norm_opts[i] if i < len(norm_opts) else '' for i in range(len(norm_opts))}, 'max_similarity': float(max_sim), 'mean_similarity': float(mean_sim), 'max_correct_similarity': float(max_correct_sim) if max_correct_sim is not None else None, 'severity': severity, 'duplicate_options': bool(has_dup), 'most_similar_pair': [chr(ord('A') + most_similar_pair[0]), chr(ord('A') + most_similar_pair[1])] if most_similar_pair is not None else None})
            max_pairs.append(max_sim)

        total_q = len(per_q_reports)
        avg_max = float(sum(max_pairs) / total_q) if total_q > 0 else 0.0
        duplicate_rate = 100.0 * duplicate_count / total_q if total_q > 0 else 0.0
        warning_count = sum(1 for r in per_q_reports if r['severity'] == 'warning')
        critical_count = sum(1 for r in per_q_reports if r['severity'] == 'critical')
        high_sim_rate = 100.0 * (warning_count + critical_count) / total_q if total_q > 0 else 0.0
        critical_rate = 100.0 * critical_count / total_q if total_q > 0 else 0.0

        dataset_report = {'date_time': f"{datetime.now()}", 'detector': self.NAME, 'dataset': getattr(context, 'dataset_name', None), 'num_questions': total_q, 'avg_max_pair_similarity': avg_max, 'duplicate_rate_percent': duplicate_rate, 'high_similarity_rate_percent': high_sim_rate, 'critical_rate_percent': critical_rate, 'thresholds': {'warning': thresh_warn, 'critical': thresh_crit}}

        self._dataset_report = dataset_report
        self._per_question = [r for r in per_q_reports if r['severity'] in ('warning', 'critical')]
        self._all_stat = per_q_reports

        return dataset_report

    def run(self, context: AnalysisContext, out_dir: str = None, **kwargs):
        res = super().run(context, out_dir=out_dir, **kwargs)
        if out_dir and hasattr(self, '_dataset_report'):
            try:
                rpt_dir = Path(out_dir) / 'reports' / self.NAME
                rpt_dir.mkdir(parents=True, exist_ok=True)
                (rpt_dir / 'high_similarity_questions.json').write_text(json.dumps(self._per_question, ensure_ascii=False, indent=2), encoding='utf-8')
                
                (rpt_dir / 'all_stat.json').write_text(json.dumps(self._all_stat, ensure_ascii=False, indent=2), encoding='utf-8')
            except Exception:
                logger.exception('Failed to write distractor similarity reports')
        return res
