from abc import ABC, abstractmethod
from pathlib import Path
import json

class AnalysisContext:
    def __init__(self, dataset, dataset_name, result_paths, config=None, loaded_results=None):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.result_paths = result_paths
        self.config = config or {}
        self.loaded_results = loaded_results or {}


class BaseDetector(ABC):

    NAME = None

    DESCRIPTION = None

    DEFAULT_CONFIG = {}

    def __init__(self, **kwargs):

        self.config = {
            **self.DEFAULT_CONFIG,
            **kwargs
        }

    @abstractmethod
    def analyze(self, context, **kwargs):
        """Perform detector analysis using an AnalysisContext.

        Implementations should accept a single `context` object and return a
        serializable result dict. Detectors must not perform filesystem
        traversal — the framework provides `context.result_paths` and
        optionally `context.loaded_results`.
        """
        raise NotImplementedError()

    def run(self, context, out_dir: str = None, **kwargs):
        """Run detector: call `analyze(context)` and optionally write JSON
        report to `out_dir`.

        Returns the detector result dictionary.
        """
        result = self.analyze(context=context, **kwargs)
        if out_dir and result is not None:
            try:
                p = Path(out_dir) / f'{self.NAME}.json'
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            except Exception:
                # best-effort: do not raise from file-writing issues
                pass
        return result