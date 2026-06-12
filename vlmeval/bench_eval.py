"""Detector-oriented benchmark-quality evaluation pipeline.

This module implements the architecture described in FIXES.md:
 - validate exactly one dataset is provided
 - load the dataset once
 - collect all result file paths across models/eval_runs
 - create an AnalysisContext and run detectors with it
 - write detector-level and aggregated reports
"""
from datetime import datetime
from pathlib import Path
import importlib
import json
import logging

from run import get_judge_kwargs
from vlmeval.dataset import build_dataset
from vlmeval.smp.file import get_intermediate_file_path, get_pred_file_path, get_eval_file_path, load, get_logger
from vlmeval.config import detectors as DETECTORS_REGISTRY

from vlmeval.detector.base_detector import AnalysisContext
logger = get_logger(__name__)


def run_pipeline(args):
    # Validate dataset list: exactly one dataset supported for now
    if not args.data or len(args.data) != 1:
        raise ValueError('Benchmark quality analysis supports exactly one dataset per run.')

    dataset_name = args.data[0]
    dataset = build_dataset(dataset_name)
    reports = [{"date_time": f"{datetime.now()}"}]

    # collect result paths across all models and eval runs
    models = args.model if isinstance(args.model, list) else [args.model]
    result_paths = {}
    for model_name in models:
        work_dir = Path(args.work_dir) / model_name
        if not work_dir.exists():
            logger.warning(f'Work dir for model {model_name} does not exist: {work_dir}')
            continue

        eval_dirs = sorted([p for p in work_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
        for eval_dir in eval_dirs:
            # prediction file is stored inside eval_dir named <model>_<dataset>.<ext>
            pred_path = get_pred_file_path(str(eval_dir), model_name, dataset_name, use_env_format=True)
            if not Path(pred_path).exists():
                logger.info(f'No prediction file for {model_name} at {eval_dir} — skipping')
                
            # try to locate corresponding eval file
            try:
                judge_kwargs = get_judge_kwargs(dataset_name, dataset.TYPE, args)
            except Exception:
                judge_kwargs = {}
            judge_model = judge_kwargs.get('model', '')
            eval_path = get_intermediate_file_path(pred_path, f"_{judge_model}_result")
            
            if not Path(eval_path).exists():
                logger.info(f'No eval file for {model_name} at {eval_dir} — skipping')
                eval_path = None

            # determine variant (blind vs full) from filename
            variant = 'blind' if 'blind' in Path(pred_path).name.lower() or 'blind' in eval_dir.name.lower() else 'full'
            key = f"{model_name}__{eval_dir.name}"
            result_paths[key] = {
                'model': model_name,
                'eval_id': eval_dir.name,
                'variant': variant,
                'pred': str(pred_path),
                'eval': str(eval_path) if eval_path is not None else None,
            }

    if not result_paths:
        logger.error('No result files found for the requested models/dataset. Aborting bench-eval.')
        return reports

    # Optionally preload evaluation results into memory for detectors
    loaded_results = {}
    for k, v in result_paths.items():
        try:
            if v.get('eval'):
                loaded_results[k] = load(v['eval'])
            else:
                loaded_results[k] = load(v['pred'])
        except Exception:
            logger.warning(f'Failed to load result for {k}')
            loaded_results[k] = None

    context = AnalysisContext(dataset=dataset, dataset_name=dataset_name, result_paths=result_paths, config=vars(args), loaded_results=loaded_results)

    # Execute detectors
    selected = [d.lower() for d in (args.detectors or ['all'])]
    detector_outputs = {}
    for det_name, det_factory in DETECTORS_REGISTRY.items():
        if 'all' in selected or det_name in selected:
            try:
                det = det_factory()
                det_res = det.run(context, out_dir=str(Path(args.work_dir)))
                detector_outputs[det_name] = det_res
            except Exception as e:
                logger.exception(f'Detector {det_name} failed: {e}')
                detector_outputs[det_name] = {'error': str(e)}

    # write aggregated detector outputs
    try:
        agg_path = Path(args.work_dir) / 'bench_quality_report.json'
        agg_path.parent.mkdir(parents=True, exist_ok=True)
        agg = {'date_time': f'{datetime.now()}', 'dataset': dataset_name, 'detectors': detector_outputs, 'result_paths': result_paths}
        with open(agg_path, 'w', encoding='utf-8') as f:
            json.dump(agg, f, ensure_ascii=False, indent=2)
        logger.info(f'Wrote aggregated bench quality report: {agg_path}')
    except Exception:
        logger.exception('Failed to write aggregated bench quality report')

    return detector_outputs