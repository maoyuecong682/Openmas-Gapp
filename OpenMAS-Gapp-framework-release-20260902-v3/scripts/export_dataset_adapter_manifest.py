from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from openmas_bench.dataset_adapters import all_adapters

payload = {"schema_version": "q2-adapter-manifest-v1", "adapters": []}
for adapter in all_adapters():
    payload["adapters"].append({
        "dataset": adapter.dataset_id,
        "source_file": adapter.source_file,
        "split": adapter.split,
        "license": adapter.license,
        "requirement_template": {
            "domain": adapter.template.domain,
            "family": adapter.template.family,
            "text": adapter.template.text,
            "stages": [{"id": x, "description": y} for x, y in adapter.template.stages],
            "constraints": [{"id": x, "kind": y, "target": z, "predicate": q} for x, y, z, q in adapter.template.constraints],
        },
        "execution_adapter": type(adapter.execution).__name__,
        "primary_metric": adapter.execution.metric_name,
        "primary_score_protocol": "answer_score * runtime_valid * trace_contract_rate; code/patch answer_score comes from sandbox execution",
    })
out = ROOT.parents[1] / "q2_dataset_adapter_manifest.json"
out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(out)
