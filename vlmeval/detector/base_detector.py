from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
import json
import string


class DetectorInputError(Exception):
    pass


class AnalysisContext:
    def __init__(
        self,
        dataset,
        dataset_name,
        result_paths,
        config=None,
        loaded_results=None,
        full_results=None,
        blind_results=None,
        mode=None,
    ):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.result_paths = result_paths
        self.config = config or {}
        # legacy single mapping for detectors that expect loaded_results
        self.loaded_results = loaded_results or {}
        # new explicit mappings
        self.full_results = full_results or {}
        self.blind_results = blind_results or {}
        # execution mode: 'full_only' or 'full_vs_blind'
        self.mode = mode or ("full_vs_blind" if self.blind_results else "full_only")


class BaseDetector(ABC):

    NAME = None

    DESCRIPTION = None

    DEFAULT_CONFIG = {}

    # Capability flags (detectors may override)
    REQUIRES_FULL_RESULTS = True
    REQUIRES_BLIND_RESULTS = False
    REQUIRES_MULTIPLE_MODELS = False
    REQUIRES_CORRECTNESS_LABELS = False
    SUPPORTS_COMPARISON = False

    def __init__(self, context=None, **kwargs):
        self.config = {**self.DEFAULT_CONFIG, **kwargs}

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
                p = Path(out_dir) / "reports" / f"{self.NAME}.json"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception:
                # best-effort: do not raise from file-writing issues
                pass
        return result

    def can_run(self, context: AnalysisContext):
        """Return (True, None) if detector can run on the provided context, else (False, reason)."""
        # check full results
        if self.REQUIRES_FULL_RESULTS:
            if not getattr(context, "full_results", None):
                return False, "Full results are required but not provided."
        # check blind results
        if self.REQUIRES_BLIND_RESULTS:
            if not getattr(context, "blind_results", None):
                return False, "Blind results are required but not provided."
        # check multiple models
        if self.REQUIRES_MULTIPLE_MODELS:
            rp = getattr(context, "result_paths", {})
            if not rp or len(rp) < 2:
                return False, "Detector requires results from multiple models."
        return True, None

    def _normalize_answer(self, a):
        if a is None:
            return None
        try:
            import pandas as pd

            if pd.isna(a):
                return None
        except Exception:
            pass
        s = str(a).strip()
        if s == "":
            return None
        return s

    def _extract_mcq_option(self, value, valid_options=None) -> str:
        """
        Extract MCQ option letter (A/B/C/D/etc) from value.
        Returns 'Z' if no valid option found.
        """
        if value is None:
            return "Z"

        try:
            import pandas as pd

            if pd.isna(value):
                return "Z"
        except Exception:
            pass

        s = str(value).strip()
        for punct in string.punctuation:
            s = s.replace(punct, "")

        if s == "":
            return "Z"

        # Try to extract letter from the value itself
        # Single letter
        if len(s) == 1 and s.isalpha() and s.upper() in valid_options:
            return s.upper()

        # No valid option found
        return "Z"

    def _get_answers_by_model(self, context: AnalysisContext):
        rp = getattr(context, "result_paths", {})
        loaded = getattr(context, "loaded_results", {})

        # ordered list of model keys
        model_keys = list(rp.keys())
        # gather answers per model per sample index
        answers_by_model = {}

        for k in model_keys:
            res = loaded.get(k, None)
            answers = []

            if res is None:
                answers_by_model[k] = None
                continue

            try:
                import pandas as pd

                if not isinstance(res, pd.DataFrame):
                    if (
                        isinstance(res, list)
                        and len(res) > 0
                        and isinstance(res[0], dict)
                    ):
                        res = pd.DataFrame(res)
                    elif isinstance(res, dict):
                        res = pd.DataFrame([res])

                # Process each row
                for idx, row in res.iterrows():
                    # Determine if the answer is correct
                    hit = None
                    for hcol in ["hit", "correct", "is_correct"]:
                        if hcol in res.columns:
                            hit = row.get(hcol)
                            break

                    # valid options is a set of all possible options in column 'answer'
                    valid_options = set(res["answer"].dropna().unique())
                    # Select answer or prediction based on hit
                    if hit is not None and hit in [
                        True,
                        1,
                        "1",
                        "True",
                        "true",
                        "TRUE",
                    ]:
                        # Use the correct answer
                        val = row.get("answer") or None
                        option = val if val is not None else "Z"
                    else:
                        # Use the model's prediction
                        val = row.get("prediction") or None
                        option = self._extract_mcq_option(val, valid_options)

                    answers.append(option)

                answers_by_model[k] = answers

            except Exception as e:
                answers_by_model[k] = None

        return answers_by_model, model_keys

    def _get_ground_truth_answers(self, context: AnalysisContext):

        dataset = getattr(context, "dataset", {})
        data = dataset.data

        rp = getattr(context, "result_paths", {})
        loaded = getattr(context, "loaded_results", {})

        # ordered list of model keys
        model_keys = list(rp.keys())
        # gather answers per model per sample index
        answers_by_model = {}

        for k in model_keys:
            res = loaded.get(k, None)
            answers = []

            if res is None:
                answers_by_model[k] = None
                continue

            try:
                import pandas as pd

                if not isinstance(res, pd.DataFrame):
                    if (
                        isinstance(res, list)
                        and len(res) > 0
                        and isinstance(res[0], dict)
                    ):
                        res = pd.DataFrame(res)
                    elif isinstance(res, dict):
                        res = pd.DataFrame([res])

                # Process each row
                for idx, row in res.iterrows():
                    # Determine if the answer is correct
                    hit = None
                    for hcol in ["hit", "correct", "is_correct"]:
                        if hcol in res.columns:
                            hit = row.get(hcol)
                            break

                    # valid options is a set of all possible options in column 'answer'
                    valid_options = set(res["answer"].dropna().unique())
                    # Select answer or prediction based on hit
                    if hit is not None and hit in [
                        True,
                        1,
                        "1",
                        "True",
                        "true",
                        "TRUE",
                    ]:
                        # Use the correct answer
                        val = row.get("answer") or None
                        option = val if val is not None else "Z"
                    else:
                        # Use the model's prediction
                        val = row.get("prediction") or None
                        option = self._extract_mcq_option(val, valid_options)

                    answers.append(option)

                answers_by_model[k] = answers

            except Exception as e:
                answers_by_model[k] = None

        return answers_by_model, model_keys

    def merge_findings(self, full_list, blind_list):
        merged = defaultdict(lambda: {"full": None, "blind": None})

        for item in full_list:
            qid = item["question_id"]
            merged[qid]["full"] = {k: v for k, v in item.items() if k != "question_id"}

        for item in blind_list:
            qid = item["question_id"]
            merged[qid]["blind"] = {k: v for k, v in item.items() if k != "question_id"}

        result = []
        for qid, data in merged.items():
            entry = {"question_id": qid}
            if data["full"]:
                entry["full"] = data["full"]
            if data["blind"]:
                entry["blind"] = data["blind"]
            result.append(entry)

        result.sort(key=lambda x: x["question_id"])
        return result
