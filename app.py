"""Silfio dashboard: a portfolio on a world map, baseline and scenario."""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

import silfio_client as sc

HERE = Path(__file__).resolve().parent
PORTFOLIO_CSV = HERE / "portfolio.csv"
STATIC = HERE / "static"
LAST_VERSION = HERE / ".streamlit" / "theatres-version"
CONFLICT_TYPES = ["interstate", "internationalised_internal", "internal"]
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
def run_cached(payload_json: str) -> dict[str, pd.DataFrame]:
    return sc.run(API_URL, API_KEY, json.loads(payload_json))


@st.cache_data
def load_portfolio(path: str, mtime: float) -> pd.DataFrame:
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
def fetch_theatres(countries: tuple[str, ...], version: str) -> tuple[list, pd.DataFrame]:
    """Outlines and shares from the API for one theatre version."""
    table = sc.theatres(API_URL, API_KEY, countries)
    rings = table.column("polygon").to_pylist()
    ongoing = table.column("ongoing").to_pylist()
    t = table.drop(["polygon", "ongoing"]).to_pandas()
    t["ongoing_share"] = [np.mean([o["share"] for o in o_list]) if o_list else np.nan for o_list in ongoing]
    t["ongoing_text"] = [
        "; ".join(f"ongoing conflict {o['conflict_id']}: {o['share']:.1%}" for o in o_list) if o_list else ""
        for o_list in ongoing
    ]
    return rings, t


@st.cache_resource(max_entries=16)
def theatre_objects(
    countries: tuple[str, ...], version: str, ongoing: bool, scenario: tuple
) -> tuple[list[dict], pd.DataFrame]:
    """One map object per theatre outline, shaded by a new conflict's split or, where a
    conflict is under way, the split the model uses if it continues."""
    rings, t = fetch_theatres(countries, version)
    shown = (t.ongoing_share.fillna(t.share) if ongoing else t.share).to_numpy(copy=True)
    pinned = {(iso3, tid): s for iso3, tid, s in scenario}
    for iso3 in {iso3 for iso3, _, _ in scenario}:
        rows = (t.iso3 == iso3).to_numpy()
        shown[rows] = scenario_shares(t[rows], pinned)
    t = t.assign(shown=shown)
    label = [
        f"{iso3} theatre {tid}: {p:.1%} of a new conflict's fighting"
        + (f"; {o}" if o else "")
        + (f"; scenario {s:.1%}" if iso3 in {i for i, _, _ in scenario} else "")
        for iso3, tid, p, o, s in zip(t.iso3, t.theatre_id, t.share, t.ongoing_text, t.shown)
    ]
    colour = ramp(t.shown.to_numpy(), SHARE_RANGE, PALETTE[3:], alpha=150)
    polygons = [
        {"polygon": r, "colour": c, "label": name, "iso3": iso3, "theatre_id": int(tid), "share": float(p)}
        for r, c, name, iso3, tid, p in zip(rings, colour, label, t.iso3, t.theatre_id, t.share)
    ]
    return polygons, t.assign(label=label)


@st.cache_data(max_entries=16)
def theatre_layers(
    countries: tuple[str, ...], version: str, ongoing: bool, label_min: float, scenario: tuple
) -> tuple[str, list[dict], pd.DataFrame]:
    """Polygon file URL, share labels and centres. The outlines run to megabytes for a
    global book, so they go to a static file the browser fetches once."""
    polygons, t = theatre_objects(countries, version, ongoing, scenario)
    body = json.dumps(polygons, separators=(",", ":"))
    name = f"theatres-{hashlib.sha1(body.encode()).hexdigest()[:16]}.json"
    STATIC.mkdir(exist_ok=True)
    for old in STATIC.glob("theatres-*.json"):
        if old.name != name and time.time() - old.stat().st_mtime > 86_400:
            old.unlink(missing_ok=True)
    if not (STATIC / name).exists():
        (STATIC / name).write_text(body)
    first = ~t.duplicated(["iso3", "theatre_id"]).to_numpy()
    centres = t[first][["iso3", "theatre_id", "shown", "latitude", "longitude"]].reset_index(drop=True)
    big = first & (t.shown.to_numpy() >= label_min)
    labels = [
        {"position": [lon, lat], "text": f"{p:.0%}" if p >= 0.1 else f"{p:.1%}", "label": name}
        for p, lat, lon, name in zip(t.shown[big], t.latitude[big], t.longitude[big], t.label[big])
    ]
    return f"app/static/{name}", labels, centres


def scenario_shares(country: pd.DataFrame, pinned: dict) -> np.ndarray:
    """A country's shares with the pinned theatres set and the rest scaled to fill 100%."""
    keys = list(zip(country.iso3, country.theatre_id))
    fixed = np.array([k in pinned for k in keys])
    set_shares = [pinned[k] for k in keys if k in pinned]
    share = country.share.to_numpy()
    rest = share[~fixed].sum()
    out = share * (1 - sum(set_shares)) / rest if rest else np.zeros_like(share)
    out[fixed] = set_shares
    return out


def ramp(values: np.ndarray, bounds: tuple[float, float], palette: np.ndarray, alpha: int) -> list[list[int]]:
    """Log-scale colour ramp over `palette`, clipped to `bounds`."""
    lo, hi = np.log10(bounds)
    t = (np.log10(np.clip(values, *bounds)) - lo) / (hi - lo) * (len(palette) - 1)
    i = np.minimum(t.astype(int), len(palette) - 2)
    rgb = palette[i] + (palette[i + 1] - palette[i]) * (t - i)[:, None]
    return np.column_stack([rgb.round(), np.full(len(values), alpha)]).astype(int).tolist()


def site_theatre_share(sites: pd.DataFrame, centres: pd.DataFrame) -> np.ndarray:
    """Share of the country's fighting in each site's theatre: the nearest centre in
    its own country, as the model assigns it."""
    share = np.full(len(sites), np.nan)
    for iso3, idx in sites.groupby("iso3").indices.items():
        c = centres[centres.iso3 == iso3]
        if c.empty:
            continue
        lat1, lon1 = np.radians(sites.latitude.to_numpy()[idx, None]), np.radians(sites.longitude.to_numpy()[idx, None])
        lat2, lon2 = np.radians(c.latitude.to_numpy()[None, :]), np.radians(c.longitude.to_numpy()[None, :])
        h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        share[idx] = c.shown.to_numpy()[h.argmin(axis=1)]
    return share


def fit_view(points: pd.DataFrame) -> pdk.ViewState:
    """Frame every point, from one city up to a global book."""
    view = pdk.data_utils.compute_view(points[["longitude", "latitude"]].values.tolist(), view_proportion=1)
    view.zoom = float(min(max(view.zoom - 0.3, 1.0), 6.0))
    return view


def money(x: float) -> str:
    if abs(x) >= 1e9:
        return f"${x / 1e9:,.2f}bn"
    return f"${x / 1e6:,.2f}m" if abs(x) < 1e7 else f"${x / 1e6:,.1f}m"


def edit_theatre_panel(picked: dict | None) -> None:
    shares = st.session_state.assumptions["theatres"]
    if picked:
        iso3, tid = picked["iso3"], int(picked["theatre_id"])
        current = shares.get(iso3, {}).get(tid, picked["share"])
        info, value, set_col, remove_col = st.columns([3, 2, 1, 1], vertical_alignment="bottom")
        info.markdown(f"**{iso3}** theatre {tid} · model share {picked['share']:.1%}")
        pct = value.number_input("Scenario share %", 0.0, 100.0, round(current * 100, 2), step=1.0)
        if set_col.button("Set", width="stretch"):
            shares.setdefault(iso3, {})[tid] = pct / 100
            st.rerun()
        if tid in shares.get(iso3, {}) and remove_col.button("Remove", width="stretch"):
            del shares[iso3][tid]
            if not shares[iso3]:
                del shares[iso3]
            st.rerun()
    else:
        st.caption("Click a theatre on the map to set its share of the country's fighting.")
    over = [iso3 for iso3, s in shares.items() if sum(s.values()) > 1 + 1e-9]
    if over:
        st.error(f"Theatre shares add up to more than 100% in {', '.join(over)}.")
    if shares and st.button("Run scenario", type="primary", disabled=bool(over)):
        st.session_state.run_requested = True
        theatres_version.clear()
        st.rerun()


portfolio = load_portfolio(str(PORTFOLIO_CSV), PORTFOLIO_CSV.stat().st_mtime)
countries = tuple(sorted(portfolio.iso3.unique()))

by_country = portfolio.groupby("iso3").agg(sites=("value", "size"), value=("value", "sum")).reset_index()
if "assumptions" not in st.session_state:
    st.session_state.assumptions = {
        "countries": by_country.assign(onset_pct=np.nan, conflict_type="interstate", damage_pct=np.nan),
        "sites": portfolio[["property_name_or_address", "iso3", "occupancy", "value"]].assign(
            location_factor=np.nan, occupancy_factor=np.nan
        ),
        "theatres": {},
    }


@st.dialog("Scenario assumptions", width="large")
def edit_assumptions() -> None:
    st.caption("Cells left as None keep the model's own assumption. Nothing runs until you press Apply.")
    current = st.session_state.assumptions
    tab_countries, tab_sites = st.tabs(["Countries", "Sites"])
    with tab_countries:
        countries_edit = st.data_editor(
            current["countries"],
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
                "conflict_type": st.column_config.SelectboxColumn(
                    "Conflict type", options=CONFLICT_TYPES, help="Used only when Onset % is set"
                ),
                "damage_pct": st.column_config.NumberColumn(
                    "Damage %", min_value=0, max_value=100, help="Share of buildings damaged in a conflict"
                ),
            },
            key="countries_editor",
        )
    with tab_sites:
        sites_edit = st.data_editor(
            current["sites"],
            hide_index=True,
            width="stretch",
            height=min(38 + 35 * len(current["sites"]), 520),
            disabled=["property_name_or_address", "iso3", "occupancy", "value"],
            column_config={
                "property_name_or_address": "Site",
                "iso3": "Country",
                "occupancy": "Occupancy",
                "value": st.column_config.NumberColumn("Insured value", format="$%,.0f"),
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
        st.session_state.assumptions |= {"countries": countries_edit, "sites": sites_edit}
        st.session_state.run_requested = True
        theatres_version.clear()  # check for new theatres alongside the run
        st.rerun()
    if cancel.button("Cancel", width="stretch"):
        st.rerun()


def assumption_lines(assumptions: dict[str, pd.DataFrame]) -> list[str]:
    lines = []
    for r in assumptions["countries"].itertuples():
        parts = []
        if pd.notna(r.onset_pct):
            parts.append(f"onset {r.onset_pct:g}% ({r.conflict_type.replace('_', ' ')})")
        if pd.notna(r.damage_pct):
            parts.append(f"damage {r.damage_pct:g}%")
        if parts:
            lines.append(f"**{r.iso3}**: " + ", ".join(parts))
    for r in assumptions["sites"].itertuples():
        parts = []
        if pd.notna(r.location_factor):
            parts.append(f"location ×{r.location_factor:g}")
        if pd.notna(r.occupancy_factor):
            parts.append(f"occupancy ×{r.occupancy_factor:g}")
        if parts:
            lines.append(f"**{r.property_name_or_address}**: " + ", ".join(parts))
    for iso3, shares in assumptions["theatres"].items():
        lines += [f"**{iso3}** theatre {t}: {s:.0%} of fighting" for t, s in shares.items()]
    return lines


with st.sidebar:
    st.title("Silfio")
    tab_book, tab_scenario = st.tabs(["Portfolio", "Scenario"])
    with tab_book:
        st.metric("Insured value", money(portfolio.value.sum()))
        st.caption(f"{len(portfolio)} sites · {len(countries)} countries")
        st.dataframe(
            portfolio[["property_name_or_address", "iso3", "occupancy", "value"]],
            hide_index=True,
            column_config={"value": st.column_config.NumberColumn("value", format="$%,.0f")},
        )
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
    payload = sc.build_payload(portfolio, assumptions["sites"], assumptions["countries"], assumptions["theatres"])
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
show_theatres = right.toggle("Theatres", help="Each country's theatres, shaded by the model's share of the fighting")
tables = views[view]
base_kpi, kpi = sc.headline(baseline), sc.headline(tables)

cols = st.columns(4)
for col, (label, key) in zip(
    cols,
    [("Expected loss", "expected_loss"), ("1-in-100 (VaR 99%)", "var_99"), ("TVaR 99%", "tvar_99"), ("Net expected loss", "net_expected_loss")],
):
    delta = None if view == "Baseline" else money(kpi[key] - base_kpi[key])
    col.metric(label, money(kpi[key]), delta, delta_color="inverse")

ongoing_view = False
try:
    version = theatres_version()
    has_ongoing = fetch_theatres(countries, version)[1].ongoing_share.notna().any()
    editing = show_theatres and right.toggle("Edit theatres", help="Pick a country, then click a theatre to set its share")
    if show_theatres and has_ongoing:
        ongoing_view = (
            right.radio(
                "Shares",
                ["New conflict", "Ongoing conflicts"],
                horizontal=True,
                label_visibility="collapsed",
                help="Ongoing: where the model sends fighting if a current conflict continues",
            )
            == "Ongoing conflicts"
        )
    edit_country = st.selectbox("Country to edit", countries) if editing else None
    scenario = tuple(
        (iso3, t, s) for iso3, shares in st.session_state.assumptions["theatres"].items() for t, s in shares.items()
    )
    polygons, share_labels, centres = theatre_layers(countries, version, ongoing_view, LABEL_MIN_SHARE, scenario)
    if not (HERE / polygons.removeprefix("app/")).exists():  # file cleared since it was cached
        theatre_layers.clear()
        polygons, share_labels, centres = theatre_layers(countries, version, ongoing_view, LABEL_MIN_SHARE, scenario)
except Exception as error:  # the map still works without theatres
    editing = False
    polygons, share_labels, centres = None, [], pd.DataFrame(columns=["iso3", "shown", "latitude", "longitude"])
    if show_theatres:
        st.warning(f"Theatres unavailable: {error}")
sites = sc.site_results(portfolio, tables)
sites["theatre_share"] = site_theatre_share(sites, centres)
radius = 5 + 9 * np.sqrt(sites.value / sites.value.max())
site_data = [
    {
        "position": [lon, lat],
        "radius": r,
        "colour": c,
        "label": f"{name}: {bps:,.1f} bps" + ("" if np.isnan(s) else f" · theatre {s:.1%} of fighting"),
    }
    for name, lon, lat, r, c, bps, s in zip(
        sites.property_name_or_address,
        sites.longitude,
        sites.latitude,
        radius,
        ramp(sites.bps.to_numpy(), BPS_RANGE, PALETTE, alpha=220),
        sites.bps,
        sites.theatre_share,
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
            pickable=not editing,
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
if editing:
    # Clicks are only reported for data sent with the page, so send one country
    editable = [o for o in theatre_objects(countries, version, ongoing_view, scenario)[0] if o["iso3"] == edit_country]
    layers.append(
        pdk.Layer(
            "PolygonLayer",
            editable,
            id="theatres",
            get_polygon="polygon",
            get_fill_color=[0, 0, 0, 0],
            get_line_color=[11, 22, 48, 160],
            line_width_min_pixels=1,
            stroked=True,
            pickable=True,
            auto_highlight=True,
        )
    )
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

event = st.pydeck_chart(
    pdk.Deck(
        map_style="light",
        initial_view_state=fit_view(centres[centres.iso3 == edit_country] if editing else sites),
        layers=layers,
        tooltip={"text": "{label}"},
    ),
    height=560,
    on_select="rerun" if editing else "ignore",
    selection_mode="single-object",
    key="map",
)
if editing:
    picked = (event.selection.objects or {}).get("theatres") if event else None
    if picked:
        st.session_state.picked = {k: picked[0][k] for k in ("iso3", "theatre_id", "share")}
    edit_theatre_panel(st.session_state.get("picked"))
caption = "Circle size: insured value. Colour: annual loss cost in bps of value, 0.1 (blue) to 200+ (red)."
if show_theatres:
    caption += (
        " Shading: theatre's share of its country's fighting"
        + (
            " if a current conflict continues (recent fighting blended with the base split; averaged where there are several)"
            if ongoing_view
            else " in a new conflict"
        )
        + f", labelled from {LABEL_MIN_SHARE:.0%}."
    )
st.caption(caption)

st.dataframe(
    sites.sort_values("bps", ascending=False)[
        ["property_name_or_address", "iso3", "value", "expected_loss", "bps", "hit_frequency", "theatre_share"]
    ],
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
