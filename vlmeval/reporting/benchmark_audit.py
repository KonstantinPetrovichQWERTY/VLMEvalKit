from pathlib import Path
import json
from collections import defaultdict, Counter
from typing import Any, Dict


class BenchmarkAuditReportGenerator:
    """Aggregate detector findings into aggregated_findings.json and a human-readable markdown report.

    Usage:
        gen = BenchmarkAuditReportGenerator()
        gen.generate(out_dir)
    """

    def generate(self, out_dir: str) -> Dict[str, Any]:
        out = Path(out_dir)
        rpt_dir = out / 'reports'
        if not rpt_dir.exists():
            raise FileNotFoundError(f"Reports directory not found: {rpt_dir}")

        aggregated = {
            'total_findings': 0,
            'by_detector': {},
            'by_severity': {},
            'questions': {},
        }

        # collect findings from any JSON files that include a 'findings' key
        findings_list = []
        for p in rpt_dir.rglob('*_findings.json'):
            try:
                data = json.loads(p.read_text(encoding='utf-8'))
            except Exception:
                continue
            # sometimes detectors write arrays (e.g., per_question lists); skip those
            if isinstance(data, dict):
                # direct findings
                if 'findings' in data and isinstance(data.get('findings'), list):
                    for f in data.get('findings'):
                        f_rec = dict(f)
                        findings_list.append(f_rec)
                # comparison-style reports may have nested 'full'/'blind' entries containing findings
                else:
                    for side in ('full', 'blind'):
                        if side in data and isinstance(data[side], dict) and 'findings' in data[side] and isinstance(data[side].get('findings'), list):
                            for f in data[side].get('findings'):
                                f_rec = dict(f)
                                f_rec.setdefault('comparison_side', side)
                                findings_list.append(f_rec)

        # build aggregates
        by_detector = defaultdict(list)
        by_severity = defaultdict(list)
        questions = defaultdict(list)

        for f in findings_list:
            det = f.get('detector', 'unknown')
            sev = f.get('severity', 'info')
            qid = f.get('question_id', None)
            by_detector[det].append(f)
            by_severity[sev].append(f)
            if qid is not None:
                questions.setdefault(str(qid), []).append(f)

        aggregated['total_findings'] = len(findings_list)
        aggregated['by_detector'] = {k: {'count': len(v), 'examples': v[:10]} for k, v in by_detector.items()}
        aggregated['by_severity'] = {k: {'count': len(v), 'examples': v[:10]} for k, v in by_severity.items()}
        aggregated['questions'] = {k: {'count': len(v), 'findings': v} for k, v in questions.items()}

        # write aggregated JSON
        agg_path = rpt_dir / 'aggregated_findings.json'
        agg_path.write_text(json.dumps(aggregated, ensure_ascii=False, indent=2), encoding='utf-8')

        # create a human-readable markdown report
        md_lines = []
        md_lines.append('# Benchmark Audit Report')
        md_lines.append('')
        md_lines.append(f'*Total findings:* {aggregated["total_findings"]}')
        md_lines.append('')

        md_lines.append('## Findings by Detector')
        for det, info in aggregated['by_detector'].items():
            md_lines.append(f'- **{det}**: {info["count"]} findings')
        md_lines.append('')

        md_lines.append('## Findings by Severity')
        for sev, info in aggregated['by_severity'].items():
            md_lines.append(f'- **{sev}**: {info["count"]} findings')
        md_lines.append('')

        # Most common problems
        md_lines.append('## Most Common Problems')
        problem_counts = Counter([f.get('reason', 'unknown') for f in findings_list])
        for reason, cnt in problem_counts.most_common(20):
            md_lines.append(f'- {reason}: {cnt}')
        md_lines.append('')

        # Top critical questions
        md_lines.append('## Top 20 Critical Questions')
        criticals = [f for f in findings_list if f.get('severity') == 'critical']
        # group by question id
        crit_by_q = defaultdict(list)
        for f in criticals:
            qid = f.get('question_id')
            crit_by_q[qid].append(f)
        top_q = sorted(crit_by_q.items(), key=lambda x: len(x[1]) if x[0] is not None else 0, reverse=True)[:20]
        for qid, flist in top_q:
            md_lines.append(f'- Question `{qid}`: {len(flist)} critical findings')
        md_lines.append('')

        md_lines.append('## Notes')
        md_lines.append('- This report aggregates detector-generated findings. Review findings in `reports/aggregated_findings.json`.')

        md_path = rpt_dir / 'benchmark_audit_report.md'
        md_path.write_text('\n'.join(md_lines), encoding='utf-8')

        return {'aggregated_path': str(agg_path), 'markdown_path': str(md_path)}
