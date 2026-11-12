"""Named scenario overrides stored as an Excel-friendly CSV, without portfolio values."""

import csv
import io
import math
import os
import tempfile
from pathlib import Path

import polars as pl

FIELDS = {
    "countries": ("iso3", ("onset_pct", "damage_pct")),
    "sites": ("property_name_or_address", ("location_factor", "occupancy_factor", "standalone_probability_pct")),
}

COLUMNS = [
    "scenario",
    "iso3",
    "property_name_or_address",
    "onset_pct",
    "damage_pct",
    "location_factor",
    "occupancy_factor",
    "standalone_probability_pct",
]


def validate(scenarios: dict) -> dict:
    if not isinstance(scenarios, dict):
        raise ValueError("Scenarios must be keyed by scenario name.")
    for name, scenario in scenarios.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(scenario, dict):
            raise ValueError("Each scenario needs a non-empty name and an object of overrides.")
        for table, (key, fields) in FIELDS.items():
            rows = scenario.get(table, [])
            if not isinstance(rows, list):
                raise ValueError(f"{name}: {table} must be a list.")
            seen = set()
            for row in rows:
                if not isinstance(row, dict) or not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"{name}: each {table} row needs {key}.")
                if row[key] in seen:
                    raise ValueError(f"{name}: duplicate {key}: {row[key]}.")
                seen.add(row[key])
                for field in fields:
                    value = row.get(field)
                    if value is not None and (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or not 0 <= value <= 100
                    ):
                        raise ValueError(f"{name}: {field} must be a number from 0 to 100, or null.")
    return scenarios


def load(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}
    if not text.strip():
        return {}
    reader = csv.DictReader(io.StringIO(text), strict=True)
    try:
        if reader.fieldnames != COLUMNS:
            raise ValueError("Scenario CSV headers must be: " + ", ".join(COLUMNS))
        scenarios = {}
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Row {line}: wrong number of columns.")
            row = {key: value.strip() for key, value in row.items()}
            if not any(row.values()):
                continue
            name = row["scenario"]
            if not name:
                raise ValueError(f"Row {line}: enter a scenario name.")
            scenario = scenarios.setdefault(name, {"countries": [], "sites": []})
            if row["iso3"] and row["property_name_or_address"]:
                raise ValueError(f"Row {line}: enter either a country code or a site name, not both.")
            numeric_fields = COLUMNS[3:]
            if not row["iso3"] and not row["property_name_or_address"]:
                if any(row[field] for field in numeric_fields):
                    raise ValueError(f"Row {line}: overrides need a country code or a site name.")
                continue  # A name-only row preserves a saved baseline scenario.
            table = "countries" if row["iso3"] else "sites"
            key, fields = FIELDS[table]
            if any(row[field] for field in numeric_fields if field not in fields):
                raise ValueError(f"Row {line}: country and site overrides must be on separate rows.")
            values = {field: float(row[field]) for field in fields if row[field]}
            identifier = row[key].upper() if table == "countries" else row[key]
            scenario[table].append({key: identifier, **values})
        return validate(scenarios)
    except csv.Error as error:
        raise ValueError(f"Invalid scenarios CSV: {error}") from error


def save(path: Path, name: str, assumptions: dict[str, pl.DataFrame]) -> None:
    name = name.strip()
    if not name:
        raise ValueError("Enter a scenario name.")
    scenario = {}
    for table, (key, fields) in FIELDS.items():
        rows = []
        for row in assumptions[table].select(key, *fields).iter_rows(named=True):
            values = {field: row[field] for field in fields if row[field] is not None and not math.isnan(row[field])}
            if values:
                rows.append({key: row[key], **values})
        scenario[table] = rows
    scenarios = load(path)
    scenarios[name] = scenario
    validate(scenarios)
    # Replace only after the complete file is written, leaving the old file intact on failure.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8-sig", newline="", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            for saved_name, saved in scenarios.items():
                rows = [row for table in FIELDS for row in saved.get(table, [])]
                for row in rows or [{}]:
                    writer.writerow({"scenario": saved_name, **row})
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def apply(scenario: dict, current: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Match current portfolio identifiers; clear overrides absent from the saved scenario."""
    validate({"scenario": scenario})
    result = {}
    for table, (key, fields) in FIELDS.items():
        overrides = {row[key]: row for row in scenario.get(table, [])}
        frame = current[table]
        result[table] = frame.with_columns(
            pl.Series(field, [overrides.get(identifier, {}).get(field) for identifier in frame[key]], dtype=pl.Float64)
            for field in fields
        )
    return result
