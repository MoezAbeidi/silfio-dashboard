"""Thin client for the Silfio API: portfolio CSV in, summary tables out."""

from __future__ import annotations

import io
import math
from datetime import date, timedelta

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
        params={"format": "json"},
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
        .with_columns((pl.col("expected_loss") / pl.col("value") * 1e4).alias("bps"))
    )


def headline(tables: dict[str, pl.DataFrame]) -> dict:
    ps = tables["portfolio_summary"]
    gu = ps.filter(pl.col("layer") == "ground_up").row(0, named=True)
    insurer = ps.filter(pl.col("layer") == "insurer")
    net = insurer.row(0, named=True) if not insurer.is_empty() else gu
    return {
        "expected_loss": gu["expected_loss"],
        "var_99": gu["var_99"],
        "tvar_99": gu["tvar_99"],
        "net_expected_loss": net["expected_loss"],
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
