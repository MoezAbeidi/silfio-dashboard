"""Thin client for the Silfio API: portfolio CSV in, summary tables out."""

from __future__ import annotations

import io
import math
from datetime import date, timedelta

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

TIMEOUT_S = 600


def load_portfolio(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "deductible_pct" not in df:
        df["deductible_pct"] = 0.01
    return df


def build_payload(
    portfolio: pd.DataFrame,
    site_overrides: pd.DataFrame | None = None,
    country_overrides: pd.DataFrame | None = None,
    theatre_shares: dict[str, dict[int, float]] | None = None,
) -> dict:
    start = date.today()
    sites = site_overrides.set_index("property_name_or_address") if site_overrides is not None else None
    inventory, policies = [], []
    for i, r in portfolio.reset_index(drop=True).iterrows():
        policy_id = f"P-{i + 1}"
        item = {
            "property_name_or_address": r.property_name_or_address,
            "policy_ids": [policy_id],
            "value": float(r.value),
            "currency": r.currency,
            "latitude": float(r.latitude),
            "longitude": float(r.longitude),
            "iso3": r.iso3,
            "property_sqm": float(r.property_sqm),
            "building_floors": _int_or_none(r.get("building_floors")),
            "occupancy": r.occupancy,
        }
        if sites is not None and r.property_name_or_address in sites.index:
            o = sites.loc[r.property_name_or_address]
            item["location_factor_override"] = _float_or_none(o.location_factor)
            item["occupancy_factor_override"] = _float_or_none(o.occupancy_factor)
        inventory.append(item)
        ded = float(r.deductible_pct)
        policies.append(
            {
                "policy_id": policy_id,
                "inception_date": start.isoformat(),
                "expiry_date": (start + timedelta(days=365)).isoformat(),
                "currency": r.currency,
                "insurance": {
                    "occurrence_attachment": ded * float(r.value),
                    "occurrence_limit": (1 - ded) * float(r.value),
                    "deductible_type": "excess",
                    "participation_pct": 1.0,
                },
            }
        )
    payload = {"inventory": inventory, "insurance_policies": policies}
    if country_overrides is not None:
        event, hazard = {}, {}
        for _, c in country_overrides.iterrows():
            if _float_or_none(c.onset_pct) is not None:
                event[c.iso3] = {"annual_prob": c.onset_pct / 100, "conflict_type": c.conflict_type}
            if _float_or_none(c.damage_pct) is not None:
                hazard[c.iso3] = {"damage": c.damage_pct / 100}
        if event:
            payload["event_scenario"] = event
        if hazard:
            payload["hazard_scenario"] = hazard
    theatres = {
        iso3: [{"theatre_id": int(t), "share": s} for t, s in shares.items()]
        for iso3, shares in (theatre_shares or {}).items()
        if shares
    }
    if theatres:
        payload["theatre_scenario"] = theatres
    return payload


def is_scenario(payload: dict) -> bool:
    return bool(
        payload.get("event_scenario")
        or payload.get("hazard_scenario")
        or payload.get("theatre_scenario")
        or any(
            i.get("location_factor_override") is not None or i.get("occupancy_factor_override") is not None
            for i in payload["inventory"]
        )
    )


def run(api_url: str, api_key: str, payload: dict) -> dict[str, pd.DataFrame]:
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
    return {name: pd.DataFrame(rows) for name, rows in resp.json().items()}


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


def site_results(portfolio: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    inv = tables["inventory_summary"]
    by_site = inv.groupby("property_name_or_address", as_index=False).agg(
        expected_loss=("expected_ground_up_loss", "sum"), hit_frequency=("hit_frequency", "max")
    )
    out = portfolio.merge(by_site, on="property_name_or_address", how="left").fillna(
        {"expected_loss": 0.0, "hit_frequency": 0.0}
    )
    out["bps"] = out.expected_loss / out.value * 1e4
    return out


def headline(tables: dict[str, pd.DataFrame]) -> dict:
    ps = tables["portfolio_summary"].set_index("layer")
    gu = ps.loc["ground_up"]
    net = ps.loc["insurer"] if "insurer" in ps.index else gu
    return {
        "expected_loss": gu.expected_loss,
        "var_99": gu.var_99,
        "tvar_99": gu.tvar_99,
        "net_expected_loss": net.expected_loss,
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
