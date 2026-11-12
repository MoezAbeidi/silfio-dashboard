"""Thin client for the Silfio API: portfolio CSV in, summary tables out."""

from __future__ import annotations

import io
import math
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import requests

TIMEOUT_S = 600


def load_portfolio(path: str) -> pl.DataFrame:
    df = pl.read_csv(path)
    if "deductible_pct" not in df.columns:
        df = df.with_columns(pl.lit(0.01).alias("deductible_pct"))
    return df


def build_payload(
    portfolio: pl.DataFrame,
    site_overrides: pl.DataFrame | None = None,
    country_overrides: pl.DataFrame | None = None,
) -> dict:
    start = date.today()
    sites = (
        {r["property_name_or_address"]: r for r in site_overrides.iter_rows(named=True)}
        if site_overrides is not None
        else {}
    )
    inventory, policies = [], []
    for i, r in enumerate(portfolio.iter_rows(named=True)):
        policy_id = f"P-{i + 1}"
        item = {
            "property_name_or_address": r["property_name_or_address"],
            "policy_ids": [policy_id],
            "value": float(r["value"]),
            "currency": r["currency"],
            "latitude": float(r["latitude"]),
            "longitude": float(r["longitude"]),
            "iso3": r["iso3"],
            "property_sqm": float(r["property_sqm"]),
            "building_floors": _int_or_none(r.get("building_floors")),
            "occupancy": r["occupancy"],
        }
        if r["property_name_or_address"] in sites:
            o = sites[r["property_name_or_address"]]
            item["location_factor_override"] = _float_or_none(o["location_factor"])
            item["occupancy_factor_override"] = _float_or_none(o["occupancy_factor"])
            probability = _float_or_none(o["standalone_probability_pct"])
            item["subthreshold_probability_override"] = None if probability is None else probability / 100
        inventory.append(item)
        ded = float(r["deductible_pct"])
        policies.append(
            {
                "policy_id": policy_id,
                "inception_date": start.isoformat(),
                "expiry_date": (start + timedelta(days=365)).isoformat(),
                "currency": r["currency"],
                "insurance": {
                    "occurrence_attachment": ded * float(r["value"]),
                    "occurrence_limit": (1 - ded) * float(r["value"]),
                    "deductible_type": "excess",
                    "participation_pct": 1.0,
                },
            }
        )
    payload = {"inventory": inventory, "insurance_policies": policies}
    if country_overrides is not None:
        event, hazard = {}, {}
        for c in country_overrides.iter_rows(named=True):
            if _float_or_none(c["onset_pct"]) is not None:
                event[c["iso3"]] = {"annual_prob": c["onset_pct"] / 100}
            if _float_or_none(c["damage_pct"]) is not None:
                hazard[c["iso3"]] = {"damage": c["damage_pct"] / 100}
        if event:
            payload["event_scenario"] = event
        if hazard:
            payload["hazard_scenario"] = hazard
    return payload


def is_scenario(payload: dict) -> bool:
    return bool(
        payload.get("event_scenario")
        or payload.get("hazard_scenario")
        or any(
            any(
                i.get(key) is not None
                for key in (
                    "location_factor_override",
                    "occupancy_factor_override",
                    "subthreshold_probability_override",
                )
            )
            for i in payload["inventory"]
        )
    )


def run(api_url: str, api_key: str, payload: dict) -> dict[str, pl.DataFrame]:
    endpoint = "/scenario" if is_scenario(payload) else "/run"
    resp = requests.post(
        api_url.rstrip("/") + endpoint,
        params={"format": "json", "rating_factors": "true"},
        json=payload,
        headers={"X-API-Key": api_key} if api_key else {},
        timeout=TIMEOUT_S,
    )
    if not resp.ok:
        raise RuntimeError(f"{endpoint} returned {resp.status_code}: {resp.text[:500]}")
    return {name: pl.DataFrame(rows, infer_schema_length=None) for name, rows in resp.json().items()}


def theatres_version(api_url: str, api_key: str) -> str:
    """Code that changes only when the API's theatre data does."""
    resp = requests.get(
        api_url.rstrip("/") + "/theatres/version",
        headers={"X-API-Key": api_key} if api_key else {},
        timeout=10,
    )
    if not resp.ok:
        raise RuntimeError(f"/theatres/version returned {resp.status_code}: {resp.text[:300]}")
    return resp.json()["version"]


def theatres(api_url: str, api_key: str, countries) -> pa.Table:
    """Theatre outlines and fighting shares for the given ISO3 codes."""
    resp = requests.get(
        api_url.rstrip("/") + "/theatres",
        params={"iso3": ",".join(countries)},
        headers={"X-API-Key": api_key} if api_key else {},
        timeout=60,
    )
    if not resp.ok:
        raise RuntimeError(f"/theatres returned {resp.status_code}: {resp.text[:300]}")
    return pq.read_table(io.BytesIO(resp.content))


def display_theatres(
    theatres: pl.DataFrame,
    scenario_shares: pl.DataFrame | None = None,
    ongoing: bool = False,
    subthreshold: bool = False,
) -> pl.DataFrame:
    out = theatres.with_columns(pl.lit(None, dtype=pl.Float64).alias("scenario_share"))
    if subthreshold:
        return out.with_columns(pl.col("subthreshold_share").alias("shown"))
    if scenario_shares is not None and not scenario_shares.is_empty():
        out = out.drop("scenario_share").join(
            scenario_shares.select("iso3", "theatre_id", pl.col("share").alias("scenario_share")),
            on=["iso3", "theatre_id"],
            how="left",
            validate="m:1",
            maintain_order="left",
        )
    model = pl.coalesce(pl.col("ongoing_share").fill_nan(None), pl.col("share")) if ongoing else pl.col("share")
    return out.with_columns(pl.coalesce(pl.col("scenario_share").fill_nan(None), model).alias("shown"))


def site_results(portfolio: pl.DataFrame, tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    by_site = (
        tables["inventory_summary"]
        .group_by("property_name_or_address")
        .agg(
            pl.col("expected_ground_up_loss").sum().alias("expected_loss"),
            pl.col("hit_frequency").max(),
        )
    )
    return (
        portfolio.join(by_site, on="property_name_or_address", how="left", maintain_order="left")
        .with_columns(pl.col("expected_loss", "hit_frequency").fill_nan(None).fill_null(0.0))
        .join(exposure_values(portfolio, tables), on="property_name_or_address", how="left", validate="1:1")
        .with_columns(
            pl.when(pl.col("value_usd") > 0)
            .then(pl.col("expected_loss") / pl.col("value_usd") * 1e4)
            .otherwise(None)
            .alias("bps")
        )
    )


def headline(tables: dict[str, pl.DataFrame]) -> dict:
    ps = tables["portfolio_summary"]
    gu = ps.filter(pl.col("layer") == "ground_up").row(0, named=True)
    insurer = ps.filter(pl.col("layer") == "insurer")
    net = insurer.row(0, named=True) if not insurer.is_empty() else None
    return {
        "expected_loss": gu["expected_loss"],
        "var_99": gu["var_99"],
        "tvar_99": gu["tvar_99"],
        "net_expected_loss": net["expected_loss"] if net is not None else None,
    }


def _float_or_none(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _int_or_none(v):
    v = _float_or_none(v)
    return None if v is None else int(v)


def pricing_details(portfolio: pl.DataFrame, tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Workbook-style property drivers, preserving unavailable values as null."""
    details = portfolio.select("property_name_or_address", "iso3")
    sources = (
        (
            "rating_factors",
            "property_name_or_address",
            {
                "building_type": pl.String,
                "value_usd": pl.Float64,
                "location_factor": pl.Float64,
                "occupancy_factor": pl.Float64,
            },
        ),
        (
            "inventory_summary",
            "property_name_or_address",
            {
                "property_value": pl.Float64,
                "hit_frequency": pl.Float64,
                "subthreshold_hit_percentage": pl.Float64,
                "expected_ground_up_loss": pl.Float64,
                "ground_up_tvar_99": pl.Float64,
            },
        ),
        (
            "conflict_summary",
            "iso3",
            {
                "conflicts_simulated_ratio": pl.Float64,
                "ongoing_conflicts": pl.Int64,
                "mean_damage_conditional": pl.Float64,
            },
        ),
    )
    for name, key, columns in sources:
        source = tables.get(name, pl.DataFrame())
        available = [column for column in columns if column in source.columns]
        if key in source.columns and available:
            details = details.join(
                source.select(key, *available), on=key, how="left", validate="m:1", maintain_order="left"
            )
        details = details.with_columns(
            pl.lit(None, dtype=dtype).alias(column)
            for column, dtype in columns.items()
            if column not in details.columns
        )
    details = (
        details.drop("value_usd")
        .join(exposure_values(portfolio, tables), on="property_name_or_address", how="left", validate="1:1")
        .with_columns(
            (pl.col("subthreshold_hit_percentage") / 100).alias("subthreshold_share"),
        )
    )
    return details.with_columns(
        pl.when((pl.col("value_usd") > 0) & (pl.col("hit_frequency") > 0))
        .then(pl.col("expected_ground_up_loss") / (pl.col("value_usd") * pl.col("hit_frequency")))
        .otherwise(None)
        .alias("vulnerability_if_hit"),
        pl.when(pl.col("value_usd") > 0)
        .then(pl.col("expected_ground_up_loss") / pl.col("value_usd") * 1e4)
        .otherwise(None)
        .alias("loss_cost_bps"),
    ).drop("property_value", "subthreshold_hit_percentage")


@lru_cache(maxsize=1)
def country_regions() -> dict[str, str]:
    """Bundled snapshot of Silfio's portfolio ISO3-to-region classification."""
    return dict(pl.read_csv(Path(__file__).parent / "data" / "country_regions.csv").iter_rows())


def sort_locations(frame: pl.DataFrame) -> pl.DataFrame:
    """Group display rows geographically without changing model input ordering."""
    return frame.with_columns(
        pl.col("iso3").replace_strict(country_regions(), default="Unknown / other").alias("region")
    ).sort("region", "iso3", "property_name_or_address")


def exposure_values(portfolio: pl.DataFrame, tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """USD exposure from the run; never treat a non-USD input as USD.

    Prefer API-converted property values, then the run's own FX rates. Original
    values are a valid fallback only for USD inputs. No live FX lookup is needed.
    """
    values = portfolio.select("property_name_or_address")
    candidates = []
    for table_name, column in (("rating_factors", "value_usd"), ("inventory_summary", "property_value")):
        table = tables.get(table_name, pl.DataFrame())
        if {"property_name_or_address", column}.issubset(table.columns):
            alias = f"_{table_name}_usd"
            values = values.join(
                table.select("property_name_or_address", pl.col(column).cast(pl.Float64).alias(alias)),
                on="property_name_or_address",
                how="left",
                validate="1:1",
                maintain_order="left",
            )
            candidates.append(pl.col(alias).fill_nan(None))
    if {"currency", "value"}.issubset(portfolio.columns):
        values = values.join(
            portfolio.select("property_name_or_address", "currency", "value"),
            on="property_name_or_address",
            how="left",
            validate="1:1",
            maintain_order="left",
        )
        fx = tables.get("fx_rates", pl.DataFrame())
        if {"currency", "units_per_usd"}.issubset(fx.columns):
            values = values.join(fx.select("currency", "units_per_usd"), on="currency", how="left", validate="m:1")
            candidates.append(pl.when(pl.col("units_per_usd") > 0).then(pl.col("value") / pl.col("units_per_usd")))
        candidates.append(pl.when(pl.col("currency").str.to_uppercase() == "USD").then(pl.col("value")))
    return values.select(
        "property_name_or_address",
        pl.coalesce(*candidates, pl.lit(None, dtype=pl.Float64)).alias("value_usd"),
    )
