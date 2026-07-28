"""JSON/CSV interoperability for the teacher stages."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

def json_value(value, default=None):
    if value is None or value == "": return default
    try: return json.loads(value)
    except (TypeError, json.JSONDecodeError): return value

def read_rows(path):
    path = Path(path)
    if path.suffix.lower() in (".json", ".jsonl"):
        data = json.loads(path.read_text()) if path.suffix == ".json" else [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        return data if isinstance(data, list) else data.get("rows", data.get("observations", []))
    with path.open(newline="") as f: return list(csv.DictReader(f))

def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True); Path(path).write_text(json.dumps(value, indent=2))

def write_rows(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json": return write_json(path, rows)
    if path.suffix.lower() == ".jsonl":
        path.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else "")); return
    rows = list(rows); fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def arr(value, shape=None):
    parsed = json_value(value, value)
    if isinstance(parsed, str):
        x = np.fromstring(
            parsed.strip().strip("[]").replace(",", " ").replace(";", " "),
            sep=" ",
            dtype=float,
        )
    else:
        x = np.asarray(parsed, dtype=float)
    if shape and x.size != int(np.prod(shape)): raise ValueError(f"expected {shape}, got {x.shape}")
    return x.reshape(shape) if shape else x
