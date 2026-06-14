from typing import Dict, Any, List, Optional

from tqdm import tqdm
from .base_detector import BaseDetector, AnalysisContext, DetectorInputError
from datetime import datetime
from pathlib import Path
import json
import re

from vlmeval.smp.file import get_logger
logger = get_logger(__name__)


class QuestionImageRelevanceDetector(BaseDetector):
    NAME = 'question_image_relevance'
    DESCRIPTION = 'Estimate semantic relevance between question (text) and associated image.'
    DEFAULT_CONFIG = {
        'backend': 'auto',
        'model': 'ViT-B/32',
        'threshold_high': 0.35,
        'threshold_medium': 0.20,
    }

    REQUIRES_FULL_RESULTS = False
    REQUIRES_BLIND_RESULTS = False
    REQUIRES_MULTIPLE_MODELS = False
    SUPPORTS_COMPARISON = False

    def _load_backend(self, device='cpu'):
        # Try OpenAI CLIP, then open_clip
        try:
            import clip
            import torch
            model_name = self.config.get('model', 'ViT-B/32')
            model, preprocess = clip.load(model_name, device=device)
            tokenizer = clip.tokenize
            return ('clip', model, preprocess, tokenizer, torch, device)
        except Exception:
            pass
        try:
            import open_clip
            import torch
            model_name = self.config.get('model', 'ViT-B-32')
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='openai')
            tokenizer = open_clip.get_tokenizer(model_name)
            return ('open_clip', model, preprocess, tokenizer, torch, device)
        except Exception:
            pass
        raise DetectorInputError('No supported CLIP backend found. Install "clip" or "open_clip" to use this detector.')

    def _get_image_path(self, dataset, row_index, row) -> Optional[str]:
        # Reuse dataset.dump_image or image_path column like build_prompt
        try:
            if hasattr(dataset, 'dump_image'):
                p = dataset.dump_image(row)
                # dump_image may return list or single
                if isinstance(p, list):
                    return p[0] if p else None
                return p
        except Exception:
            pass
        # fallback to image_path column
        if 'image_path' in row:
            return row['image_path']
        if 'image' in row:
            return row['image']
        return None

    def analyze(self, context: AnalysisContext, **kwargs) -> Dict[str, Any]:
        dataset = getattr(context, 'dataset', None)
        if dataset is None or not hasattr(dataset, 'data'):
            raise DetectorInputError('Dataset not available in context')

        df = dataset.data
        # prepare backend
        device = 'cpu'
        try:
            backend = self._load_backend(device=device)
        except DetectorInputError:
            # if backend missing, raise so user can install
            raise

        kind, model, preprocess, tokenizer, torch, device = backend

        thresh_high = float(self.config.get('threshold_high', 0.35))
        thresh_med = float(self.config.get('threshold_medium', 0.20))

        per_q = []
        sims = []

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # build text: question + optional hint
            qtext = ''
            if 'question' in row and row['question'] is not None:
                qtext = str(row['question'])
            elif 'question_text' in row and row['question_text'] is not None:
                qtext = str(row['question_text'])
            hint = row['hint'] if 'hint' in row and row['hint'] is not None else None
            if hint:
                qtext_full = f"{qtext} Hint: {hint}"
            else:
                qtext_full = qtext

            img_path = self._get_image_path(dataset, idx, row)
            if img_path is None:
                per_q.append({'question_id': int(row.get('index', idx)), 'question': qtext, 'similarity': None, 'classification': 'no_image'})
                continue

            # load image
            try:
                from PIL import Image
                img = Image.open(img_path).convert('RGB')
            except Exception:
                per_q.append({'question_id': int(row.get('index', idx)), 'question': qtext, 'similarity': None, 'classification': 'image_load_error'})
                continue

            try:
                # tokenize text and preprocess image then encode
                model.eval()

                def _safe_tokenize(tokenizer_fn, text):
                    # Try direct tokenization, then try truncate flag, then progressively shorten text
                    try:
                        return tokenizer_fn([text])
                    except RuntimeError as e:
                        # try truncate kwarg if supported
                        try:
                            return tokenizer_fn([text], truncate=True)
                        except Exception:
                            pass
                        # progressively shorten the text until tokenization succeeds
                        lengths = [512, 256, 128, 77, 64, 48, 32, 16]
                        for L in lengths:
                            try_text = text[:L]
                            try:
                                return tokenizer_fn([try_text])
                            except Exception:
                                continue
                        # re-raise original
                        raise

                with torch.no_grad():
                    image_input = preprocess(img).unsqueeze(0).to(device)

                    # attempt safe tokenization for both clip and open_clip tokenizers
                    try:
                        text_tokens = _safe_tokenize(tokenizer, qtext_full)
                    except Exception:
                        # fall back to truncating to a short prefix
                        text_tokens = tokenizer([qtext_full[:77]]) if hasattr(tokenizer, '__call__') else tokenizer([qtext_full[:77]])

                    # open_clip.tokenizer may return tuple; normalize to tensor
                    if isinstance(text_tokens, tuple):
                        text_tokens = text_tokens[0]
                    text_input = text_tokens.to(device)

                    image_feat = model.encode_image(image_input)
                    text_feat = model.encode_text(text_input)

                    # normalize
                    import numpy as _np
                    im = image_feat.cpu().numpy()
                    tx = text_feat.cpu().numpy()
                    imn = im / (_np.linalg.norm(im, axis=1, keepdims=True) + 1e-12)
                    txn = tx / (_np.linalg.norm(tx, axis=1, keepdims=True) + 1e-12)
                    sim = float(_np.dot(imn, txn.T).squeeze())
            except Exception:
                logger.exception('Encoding failed for sample %s', idx)
                per_q.append({'question_id': int(row.get('index', idx)), 'question': qtext, 'similarity': None, 'classification': 'encode_error'})
                continue

            # classification
            if sim >= thresh_high:
                cls = 'high'
            elif sim >= thresh_med:
                cls = 'medium'
            else:
                cls = 'low'

            per_q.append({'question_id': int(row.get('index', idx)), 'question': qtext, 'similarity': float(sim), 'classification': cls})
            sims.append(sim)

        total = len(per_q)
        mean_sim = float(sum([s for s in sims]) / len(sims)) if sims else 0.0
        low_count = sum(1 for r in per_q if r['classification'] == 'low')
        med_count = sum(1 for r in per_q if r['classification'] == 'medium')
        high_count = sum(1 for r in per_q if r['classification'] == 'high')

        report = {'date_time': f"{datetime.now()}", 'detector': self.NAME, 'dataset': getattr(context, 'dataset_name', None), 'num_questions': total, 'mean_relevance': mean_sim, 'low_rate_percent': 100.0 * low_count / total if total else 0.0, 'medium_rate_percent': 100.0 * med_count / total if total else 0.0, 'high_rate_percent': 100.0 * high_count / total if total else 0.0, 'thresholds': {'high': thresh_high, 'medium': thresh_med}}

        # store for run()
        self._dataset_report = report
        # low relevance questions for audit
        self._low_relevance = [r for r in per_q if r.get('classification') == 'low']
        self._per_question = per_q

        # attach summary and findings
        summary = {'mean_relevance': report.get('mean_relevance'), 'low_rate_percent': report.get('low_rate_percent'), 'thresholds': report.get('thresholds')}
        findings = []
        for q in self._per_question:
            if q.get('classification') == 'low':
                findings.append({'question_id': q.get('question_id'), 'detector': self.NAME, 'severity': 'critical', 'reason': 'low_image_text_relevance', 'score': q.get('similarity')})
            elif q.get('classification') == 'medium':
                findings.append({'question_id': q.get('question_id'), 'detector': self.NAME, 'severity': 'warning', 'reason': 'medium_image_text_relevance', 'score': q.get('similarity')})

        report['summary'] = summary
        report['findings'] = findings

        # update stored dataset report to include summary/findings
        self._dataset_report = report

        return report

    def run(self, context: AnalysisContext, out_dir: str = None, **kwargs):
        res = super().run(context, out_dir=out_dir, **kwargs)
        if out_dir and hasattr(self, '_dataset_report'):
            try:
                rpt_dir = Path(out_dir) / 'reports' / self.NAME
                rpt_dir.mkdir(parents=True, exist_ok=True)
                (rpt_dir / 'question_image_relevance.json').write_text(json.dumps(self._dataset_report, ensure_ascii=False, indent=2), encoding='utf-8')
                (rpt_dir / 'low_relevance_questions.json').write_text(json.dumps(self._low_relevance, ensure_ascii=False, indent=2), encoding='utf-8')
                (rpt_dir / 'per_question.json').write_text(json.dumps(self._per_question, ensure_ascii=False, indent=2), encoding='utf-8')
            except Exception:
                logger.exception('Failed to write question_image_relevance reports')
        return res
