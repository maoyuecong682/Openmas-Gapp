from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .schema import ConstructionCase, ConstructionResult, DomainPackage, MASSpec, construction_result_from_dict


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def read_jsonl(path: str | Path) -> list[Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, values: Iterable[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def load_package(path: str | Path) -> DomainPackage:
    return DomainPackage.from_dict(read_json(path))


def save_spec(path: str | Path, spec: MASSpec) -> None:
    spec.validate()
    write_json(path, spec.to_dict())


def load_construction_case(path: str | Path) -> ConstructionCase:
    return ConstructionCase.from_dict(read_json(path))


def load_construction_result(path: str | Path) -> ConstructionResult:
    return construction_result_from_dict(read_json(path))


def save_construction_result(path: str | Path, result: ConstructionResult) -> None:
    result.validate()
    write_json(path, result.to_dict())
