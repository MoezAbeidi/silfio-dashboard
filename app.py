"""Silfio dashboard: a portfolio on a world map, baseline and scenario."""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import pydeck as pdk
import streamlit as st

import silfio_client as sc

HERE = Path(__file__).resolve().parent
PORTFOLIO_CSV = HERE / "portfolio.csv"
STATIC = HERE / "static"
LAST_VERSION = HERE / ".streamlit" / "theatres-version"
PALETTE = np.array(
    [[11, 61, 145], [0, 100, 200], [156, 195, 238], [246, 205, 184], [235, 104, 52], [165, 48, 12]], dtype=float
)
BPS_RANGE = (0.1, 200.0)
SHARE_RANGE = (0.001, 0.5)
LABEL_MIN_SHARE = 0.05
EARTH_RADIUS_KM = 6371.0

st.set_page_config(page_title="Silfio", layout="wide")


def secret(name: str, default: str = "") -> str:
    try:
        return st.secrets[name]
    except (KeyError, FileNotFoundError):
        return os.getenv(name, default)


API_URL = secret("SILFIO_API_URL", "https://silfio-production.up.railway.app")
API_KEY = secret("SILFIO_API_KEY") or st.session_state.get("api_key", "")

if not API_KEY:
    st.title("Silfio")
    key = st.text_input("Silfio API key", type="password", help="Or set SILFIO_API_KEY in .streamlit/secrets.toml")
    if key:
        st.session_state.api_key = key
        st.rerun()
    st.stop()


# Results persist on disk, so a restart or a second browser tab does not rerun the
# model. The payload carries today's date, so each day starts a fresh run.
@st.cache_data(show_spinner="Running Silfio…", persist="disk")
def run_cached(payload_json: str, response_version: int = 3) -> dict[str, pl.DataFrame]:
    return sc.run(API_URL, API_KEY, json.loads(payload_json))


@st.cache_data
def load_portfolio(path: str, mtime: float) -> pl.DataFrame:
    return sc.load_portfolio(path)


@st.cache_data(ttl=600, show_spinner=False)
def theatres_version() -> str:
    """The API's theatre version: a few bytes, checked at most every ten minutes and
    again whenever a run is started. Falls back to the last version seen, so saved
    theatres still show if the API cannot be reached."""
    try:
        version = sc.theatres_version(API_URL, API_KEY)
    except Exception:
        if LAST_VERSION.exists():
            return LAST_VERSION.read_text().strip()
        raise
    LAST_VERSION.write_text(version)
    return version


# Saved on disk per version, so a restart or a new day downloads nothing unless the
# API's theatre data has actually changed (a deploy, or the daily ongoing shares).
@st.cache_data(show_spinner="Downloading theatres…", persist="disk", max_entries=8)
def fetch_theatres(countries: tuple[str, ...], version: str) -> tuple[list, pl.DataFrame]:
    """Outlines and shares from the API for one theatre version."""
    table = sc.theatres(API_URL, API_KEY, countries)
    rings = table.column("polygon").to_pylist()
    ongoing = table.column("ongoing").to_pylist()
    t = pl.from_arrow(table.drop(["polygon", "ongoing"]))
    t = t.with_columns(
        pl.Series(
            "ongoing_share",
            [np.mean([o["share"] for o in o_list]) if o_list else None for o_list in ongoing],
            dtype=pl.Float64,
        )
    )
    t = t.with_columns(
        pl.Series(
            "ongoing_text",
            [
                "; ".join(f"ongoing conflict {o['conflict_id']}: {o['share']:.1%}" for o in o_list) if o_list else ""
                for o_list in ongoing
            ],
        )
    )
    return rings, t


@st.cache_data(max_entries=16)
def theatre_layers(
    countries: tuple[str, ...],
    version: str,
    ongoing: bool,
    label_min: float,
    scenario_shares: pl.DataFrame | None = None,
    subthreshold: bool = False,
) -> tuple[str, list[dict], pl.DataFrame]:
    """Polygon file URL, share labels and centres, shaded by a new conflict's split or,
    where a conflict is under way, the split the model uses if it continues.

    The outlines run to megabytes for a global book, so they go to a static file the
    browser fetches once; reruns then send only its URL."""
    rings, t = fetch_theatres(countries, version)
    t = sc.display_theatres(t, scenario_shares, ongoing, subthreshold)
    label = [
        f"{iso3} theatre: {shown:.1%} of sub-threshold incidents"
        if subthreshold
        else (
            f"{iso3} theatre: {s:.1%} of fighting in this scenario"
            if sc._float_or_none(s) is not None
            else f"{iso3} theatre: {p:.1%} of a new conflict's fighting" + (f"; {o}" if o else "")
        )
        for iso3, p, o, s, shown in zip(t["iso3"], t["share"], t["ongoing_text"], t["scenario_share"], t["shown"])
    ]
    colour = ramp(t["shown"].to_numpy(), SHARE_RANGE, PALETTE[3:], alpha=150)
    polygons = [{"polygon": r, "colour": c, "label": name} for r, c, name in zip(rings, colour, label)]
    body = json.dumps(polygons, separators=(",", ":"))
    name = f"theatres-{hashlib.sha1(body.encode()).hexdigest()[:16]}.json"
    STATIC.mkdir(exist_ok=True)
    for old in STATIC.glob("theatres-*.json"):
        if old.name != name and time.time() - old.stat().st_mtime > 86_400:
            old.unlink(missing_ok=True)
    if not (STATIC / name).exists():
        (STATIC / name).write_text(body)
    first = t.select(pl.struct("iso3", "theatre_id").is_first_distinct()).to_series().to_numpy()
    centres = t.filter(first).select("iso3", "theatre_id", "shown", "latitude", "longitude")
    big = first & (t["shown"].to_numpy() >= label_min)
    labels = [
        {"position": [lon, lat], "text": f"{p:.0%}" if p >= 0.1 else f"{p:.1%}", "label": text}
        for p, lat, lon, text in zip(
            t["shown"].filter(big), t["latitude"].filter(big), t["longitude"].filter(big), np.array(label)[big]
        )
    ]
    return f"app/static/{name}", labels, centres


def ramp(values: np.ndarray, bounds: tuple[float, float], palette: np.ndarray, alpha: int) -> list[list[int]]:
    """Log-scale colour ramp over `palette`, clipped to `bounds`."""
    lo, hi = np.log10(bounds)
    t = (np.log10(np.clip(values, *bounds)) - lo) / (hi - lo) * (len(palette) - 1)
    i = np.minimum(t.astype(int), len(palette) - 2)
    rgb = palette[i] + (palette[i + 1] - palette[i]) * (t - i)[:, None]
    return np.column_stack([rgb.round(), np.full(len(values), alpha)]).astype(int).tolist()


def site_theatre_share(sites: pl.DataFrame, centres: pl.DataFrame) -> np.ndarray:
    """Share of the country's fighting in each site's theatre: the nearest centre in
    its own country, as the model assigns it."""
    share = np.full(len(sites), np.nan)
    for iso3 in sites["iso3"].unique():
        idx = np.flatnonzero((sites["iso3"] == iso3).to_numpy())
        c = centres.filter(pl.col("iso3") == iso3)
        if c.is_empty():
            continue
        lat1, lon1 = (
            np.radians(sites["latitude"].to_numpy()[idx, None]),
            np.radians(sites["longitude"].to_numpy()[idx, None]),
        )
        lat2, lon2 = np.radians(c["latitude"].to_numpy()[None, :]), np.radians(c["longitude"].to_numpy()[None, :])
        h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        share[idx] = c["shown"].to_numpy()[h.argmin(axis=1)]
    return share


def fit_view(points: pl.DataFrame) -> pdk.ViewState:
    """Frame every point, from one city up to a global book."""
    view = pdk.data_utils.compute_view(points.select("longitude", "latitude").rows(), view_proportion=1)
    view.zoom = float(min(max(view.zoom - 0.3, 1.0), 6.0))
    return view


def money(x: float) -> str:
    if abs(x) >= 1e9:
        return f"${x / 1e9:,.2f}bn"
    return f"${x / 1e6:,.2f}m" if abs(x) < 1e7 else f"${x / 1e6:,.1f}m"


portfolio = load_portfolio(str(PORTFOLIO_CSV), PORTFOLIO_CSV.stat().st_mtime)
countries = tuple(sorted(portfolio["iso3"].unique()))

by_country = portfolio.group_by("iso3").agg(pl.len().alias("sites"), pl.col("value").sum()).sort("iso3")
# Preserve assumptions and results from browser sessions opened before the migration.
for state_key in ("assumptions", "scenario_results"):
    if state_key in st.session_state:
        st.session_state[state_key] = {
            name: frame if isinstance(frame, pl.DataFrame) else pl.from_dataframe(frame)
            for name, frame in st.session_state[state_key].items()
        }
if "assumptions" not in st.session_state:
    st.session_state.assumptions = {
        "countries": by_country.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("onset_pct"),
            pl.lit(None, dtype=pl.Float64).alias("damage_pct"),
        ),
        "sites": portfolio.select("property_name_or_address", "iso3", "occupancy", "value").with_columns(
            pl.lit(None, dtype=pl.Float64).alias("location_factor"),
            pl.lit(None, dtype=pl.Float64).alias("occupancy_factor"),
            pl.lit(None, dtype=pl.Float64).alias("standalone_probability_pct"),
        ),
    }

# Remove the retired selector from assumptions saved by an older browser session.
if "conflict_type" in st.session_state.assumptions["countries"].columns:
    st.session_state.assumptions["countries"] = st.session_state.assumptions["countries"].drop("conflict_type")
    st.session_state.pop("countries_editor", None)
    st.session_state.pop("scenario_results", None)


@st.dialog("Scenario assumptions", width="large")
def edit_assumptions() -> None:
    st.caption("Cells left as None keep the model's own assumption. Nothing runs until you press Apply.")
    current = st.session_state.assumptions
    tab_countries, tab_sites = st.tabs(["Countries", "Sites"])
    with tab_countries:
        countries_edit = st.data_editor(
            current["countries"].to_arrow(),
            hide_index=True,
            width="stretch",
            height=min(38 + 35 * len(current["countries"]), 520),
            disabled=["iso3", "sites", "value"],
            column_config={
                "iso3": "Country",
                "sites": st.column_config.NumberColumn("Sites"),
                "value": st.column_config.NumberColumn("Insured value", format="$%,.0f"),
                "onset_pct": st.column_config.NumberColumn(
                    "Onset %", min_value=0, max_value=100, help="Annual probability a conflict starts"
                ),
                "damage_pct": st.column_config.NumberColumn(
                    "Damage %", min_value=0, max_value=100, help="Share of buildings damaged in a conflict"
                ),
            },
            key="countries_editor",
        )
    with tab_sites:
        st.caption(
            "Optional annual probability of a damaging targeted strike or other sub-threshold incident. No full conflict is required. This replaces the site's modelled sub-threshold risk; blank keeps the model. Damage severity stays modelled."
        )
        sites_edit = st.data_editor(
            current["sites"].to_arrow(),
            column_order=[
                "property_name_or_address",
                "iso3",
                "occupancy",
                "value",
                "standalone_probability_pct",
                "location_factor",
                "occupancy_factor",
            ],
            hide_index=True,
            width="stretch",
            height=min(38 + 35 * len(current["sites"]), 520),
            disabled=["property_name_or_address", "iso3", "occupancy", "value"],
            column_config={
                "property_name_or_address": "Site",
                "iso3": "Country",
                "occupancy": "Occupancy",
                "value": st.column_config.NumberColumn("Insured value", format="$%,.0f"),
                "standalone_probability_pct": st.column_config.NumberColumn(
                    "Annual targeted-strike probability override (%)",
                    min_value=0,
                    max_value=100,
                    help="Optional expert estimate of the annual probability of a damaging targeted strike. No full conflict is required. Replaces the modelled sub-threshold probability; leave blank to use the model. Damage severity stays modelled. 0 disables sub-threshold damage.",
                ),
                "location_factor": st.column_config.NumberColumn(
                    "Location ×", min_value=0, max_value=100, help="Multiple of the country's average property"
                ),
                "occupancy_factor": st.column_config.NumberColumn(
                    "Occupancy ×", min_value=0, max_value=100, help="Multiple of a residential building"
                ),
            },
            key="sites_editor",
        )
    apply, cancel = st.columns(2)
    if apply.button("Apply and run", type="primary", width="stretch"):
        st.session_state.assumptions = {"countries": pl.from_arrow(countries_edit), "sites": pl.from_arrow(sites_edit)}
        st.session_state.run_requested = True
        theatres_version.clear()  # check for new theatres alongside the run
        st.rerun()
    if cancel.button("Cancel", width="stretch"):
        st.rerun()


def assumption_lines(assumptions: dict[str, pl.DataFrame]) -> list[str]:
    lines = []
    for r in assumptions["countries"].iter_rows(named=True):
        parts = []
        if sc._float_or_none(r["onset_pct"]) is not None:
            parts.append(f"onset {r['onset_pct']:g}%")
        if sc._float_or_none(r["damage_pct"]) is not None:
            parts.append(f"damage {r['damage_pct']:g}%")
        if parts:
            lines.append(f"**{r['iso3']}**: " + ", ".join(parts))
    for r in assumptions["sites"].iter_rows(named=True):
        parts = []
        if sc._float_or_none(r["location_factor"]) is not None:
            parts.append(f"location ×{r['location_factor']:g}")
        if sc._float_or_none(r["occupancy_factor"]) is not None:
            parts.append(f"occupancy ×{r['occupancy_factor']:g}")
        if sc._float_or_none(r["standalone_probability_pct"]) is not None:
            parts.append(f"targeted-strike probability {r['standalone_probability_pct']:g}% annually")
        if parts:
            lines.append(f"**{r['property_name_or_address']}**: " + ", ".join(parts))
    return lines


@st.dialog("Portfolio locations", width="large")
def show_portfolio() -> None:
    st.caption(f"{len(portfolio)} sites · {len(countries)} countries")
    st.dataframe(
        portfolio,
        hide_index=True,
        width="stretch",
        column_config={"value": st.column_config.NumberColumn("Insured value", format="$%,.0f")},
    )
    if st.button("Close", width="stretch"):
        st.rerun()


with st.sidebar:
    st.title("Silfio")
    tab_book, tab_scenario = st.tabs(["Portfolio", "Scenario"])
    with tab_book:
        st.metric("Insured value", money(portfolio["value"].sum()))
        st.caption(f"{len(portfolio)} sites · {len(countries)} countries")
        if st.button("View locations", width="stretch"):
            show_portfolio()
        if st.button("Refresh results", help="Rerun the model instead of using saved results"):
            run_cached.clear()
            theatres_version.clear()
            st.session_state.pop("scenario_results", None)
    with tab_scenario:
        lines = assumption_lines(st.session_state.assumptions)
        if lines:
            st.markdown("\n".join(f"- {line}" for line in lines))
        else:
            st.caption("No assumptions set: the scenario is the baseline.")
        if st.button("Edit assumptions", type="primary", width="stretch"):
            edit_assumptions()
        if lines and st.button("Clear scenario", width="stretch"):
            del st.session_state.assumptions
            st.session_state.pop("scenario_results", None)
            st.rerun()

baseline = run_cached(json.dumps(sc.build_payload(portfolio)))
views = {"Baseline": baseline}
if st.session_state.pop("run_requested", False):
    assumptions = st.session_state.assumptions
    payload = sc.build_payload(portfolio, assumptions["sites"], assumptions["countries"])
    if sc.is_scenario(payload):
        st.session_state.scenario_results = run_cached(json.dumps(payload))
        st.session_state.view = "Scenario"
    else:
        st.session_state.pop("scenario_results", None)
if "scenario_results" in st.session_state:
    views["Scenario"] = st.session_state.scenario_results
if st.session_state.get("view") not in views:
    st.session_state.view = "Baseline"

left, right = st.columns([3, 1])
view = left.radio("View", list(views), horizontal=True, label_visibility="collapsed", key="view")
show_theatres = right.toggle(
    "Theatres", help="Each country's theatres, shaded by the selected view's share of the fighting"
)
tables = views[view]
scenario_shares = tables.get("scenario_theatre_shares") if view == "Scenario" else None
if view == "Scenario" and scenario_shares is None:
    st.info("Missing scenario shares: update the API, then Refresh results and Apply and run.")
base_kpi, kpi = sc.headline(baseline), sc.headline(tables)

cols = st.columns(4)
for col, (label, key) in zip(
    cols,
    [
        ("Expected loss", "expected_loss"),
        ("1-in-100 (VaR 99%)", "var_99"),
        ("TVaR 99%", "tvar_99"),
        ("Net expected loss", "net_expected_loss"),
    ],
):
    delta = None if view == "Baseline" else money(kpi[key] - base_kpi[key])
    col.metric(label, money(kpi[key]), delta, delta_color="inverse")

ongoing_view = subthreshold_view = False
try:
    version = theatres_version()
    has_ongoing = fetch_theatres(countries, version)[1]["ongoing_share"].is_not_null().any()
    if show_theatres:
        choices = ["New conflict", "Sub-threshold"] + (["Ongoing conflicts"] if has_ongoing else [])
        split = right.selectbox("Theatre split", choices)
        ongoing_view, subthreshold_view = split == "Ongoing conflicts", split == "Sub-threshold"
    polygons, share_labels, centres = theatre_layers(
        countries, version, ongoing_view, LABEL_MIN_SHARE, scenario_shares, subthreshold_view
    )
    if not (HERE / polygons.removeprefix("app/")).exists():  # file cleared since it was cached
        theatre_layers.clear()
        polygons, share_labels, centres = theatre_layers(
            countries, version, ongoing_view, LABEL_MIN_SHARE, scenario_shares, subthreshold_view
        )
except Exception as error:  # the map still works without theatres
    polygons, share_labels, centres = (
        None,
        [],
        pl.DataFrame(schema={"iso3": pl.String, "shown": pl.Float64, "latitude": pl.Float64, "longitude": pl.Float64}),
    )
    if show_theatres:
        st.warning(f"Theatres unavailable: {error}")
sites = sc.site_results(portfolio, tables)
sites = sites.with_columns(pl.Series("theatre_share", site_theatre_share(sites, centres)))
theatre_measure = "sub-threshold incidents" if subthreshold_view else "fighting"
radius = 5 + 9 * np.sqrt(sites["value"].to_numpy() / sites["value"].max())
site_data = [
    {
        "position": [lon, lat],
        "radius": r,
        "colour": c,
        "label": f"{name}: {bps:,.1f} bps" + ("" if np.isnan(s) else f" · theatre {s:.1%} of {theatre_measure}"),
    }
    for name, lon, lat, r, c, bps, s in zip(
        sites["property_name_or_address"],
        sites["longitude"],
        sites["latitude"],
        radius,
        ramp(sites["bps"].to_numpy(), BPS_RANGE, PALETTE, alpha=220),
        sites["bps"],
        sites["theatre_share"],
    )
]

layers = []
if show_theatres and polygons:
    layers += [
        pdk.Layer(
            "PolygonLayer",
            polygons,
            get_polygon="polygon",
            get_fill_color="colour",
            get_line_color=[255, 255, 255, 200],
            line_width_min_pixels=0.5,
            stroked=True,
            pickable=True,
        ),
        pdk.Layer(
            "TextLayer",
            share_labels,
            get_position="position",
            get_text="text",
            get_size=11,
            get_color=[11, 22, 48],
            font_weight=600,
            background=True,
            get_background_color=[255, 255, 255, 210],
            background_padding=[3, 1],
            pickable=True,
        ),
    ]
layers.append(
    pdk.Layer(
        "ScatterplotLayer",
        site_data,
        get_position="position",
        get_radius="radius",
        radius_units='"pixels"',
        get_fill_color="colour",
        get_line_color=[11, 22, 48],
        line_width_min_pixels=1,
        stroked=True,
        pickable=True,
    )
)

st.pydeck_chart(
    pdk.Deck(
        map_style="light",
        initial_view_state=fit_view(sites),
        layers=layers,
        tooltip={"text": "{label}"},
    ),
    height=560,
)
caption = "Circle size: insured value. Colour: annual loss cost in bps of value, 0.1 (blue) to 200+ (red)."
if show_theatres:
    caption += f" Shading: theatre's share of its country's {theatre_measure}; labels from {LABEL_MIN_SHARE:.0%}."
    if subthreshold_view:
        caption += " Model incident locations; direct site probability overrides do not redistribute these shares."
    elif ongoing_view:
        caption += " Ongoing conflicts: recent fighting blended with the model split."
    if not subthreshold_view and scenario_shares is not None and not scenario_shares.is_empty():
        caption += " Scenario splits apply to new and ongoing conflicts."
st.caption(caption)

st.dataframe(
    sites.sort("bps", descending=True).select(
        "property_name_or_address", "iso3", "value", "expected_loss", "bps", "hit_frequency", "theatre_share"
    ),
    hide_index=True,
    width="stretch",
    column_config={
        "value": st.column_config.NumberColumn(format="$%,.0f"),
        "expected_loss": st.column_config.NumberColumn("expected loss", format="$%,.0f"),
        "bps": st.column_config.NumberColumn("loss cost (bps)", format="%.1f"),
        "hit_frequency": st.column_config.NumberColumn("hit frequency", format="%.4f"),
        "theatre_share": st.column_config.NumberColumn("theatre share", format="percent"),
    },
)
