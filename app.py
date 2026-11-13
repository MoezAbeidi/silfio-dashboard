"""Silfio dashboard: a portfolio on a world map, baseline and scenario."""

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl
import pydeck as pdk
import streamlit as st

from silfio_dashboard import scenario_store
from silfio_dashboard import silfio_client as sc

HERE = Path(__file__).resolve().parent
SCENARIOS_FILE = HERE / "silfio_dashboard" / "data" / "scenarios.csv"
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
BOOKS_DIR = Path(secret("SILFIO_BOOKS_DIR", str(HERE / "books"))).expanduser()
BOOK_COLUMNS = (
    "property_name_or_address",
    "iso3",
    "latitude",
    "longitude",
    "occupancy",
    "value",
    "currency",
    "property_sqm",
    "building_floors",
)
CONFIGURED_API_KEY = secret("SILFIO_API_KEY")
API_KEY = CONFIGURED_API_KEY or st.session_state.get("api_key", "")
FREE_MAX_PROPERTIES = 10  # The API runs keyless baselines up to this many properties
# "download" (for hosted use) hands each saved scenario to the visitor as a CSV and keeps nothing on the
# server; uploads load it back. The default "file" saves to SCENARIOS_FILE, as suits a local installation.
DOWNLOAD_SCENARIOS = secret("SILFIO_SCENARIO_STORAGE", "file").strip().lower() in ("download", "session")


def session_dir() -> Path:
    """Private scratch folder for this browser session's uploads."""
    if "session_dir" not in st.session_state:
        st.session_state.session_dir = Path(tempfile.mkdtemp(prefix="silfio-"))
    return st.session_state.session_dir


def scenarios_file() -> Path:
    """Where saved scenarios are read from: the repo file, or a per-session copy of the visitor's upload."""
    return session_dir() / "scenarios.csv" if DOWNLOAD_SCENARIOS else SCENARIOS_FILE


def run_model(payload: dict) -> dict[str, pl.DataFrame]:
    """Request fresh results only when a Run button is pressed."""
    with st.spinner("Running Silfio…"):
        return sc.run(API_URL, API_KEY, payload)


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
        # Write then rename, so a visitor never fetches a file another session is still writing.
        with tempfile.NamedTemporaryFile("w", dir=STATIC, suffix=".tmp", delete=False) as handle:
            handle.write(body)
        os.replace(handle.name, STATIC / name)
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
    missing = ~np.isfinite(values)
    values = np.where(missing, bounds[0], values)
    lo, hi = np.log10(bounds)
    t = (np.log10(np.clip(values, *bounds)) - lo) / (hi - lo) * (len(palette) - 1)
    i = np.minimum(t.astype(int), len(palette) - 2)
    rgb = palette[i] + (palette[i + 1] - palette[i]) * (t - i)[:, None]
    colours = np.column_stack([rgb.round(), np.full(len(values), alpha)]).astype(int)
    colours[missing] = [150, 150, 150, alpha]
    return colours.tolist()


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
    view.min_zoom = 1.0  # Keep interactive zoom within the full-world view.
    view.zoom = float(min(max(view.zoom - 0.3, view.min_zoom), 6.0))
    return view


def money(x: float) -> str:
    if abs(x) >= 1e9:
        return f"${x / 1e9:,.2f}bn"
    return f"${x / 1e6:,.2f}m" if abs(x) < 1e7 else f"${x / 1e6:,.1f}m"


UPLOADED = " (uploaded)"
EDITED = " (edited)"
PROPERTIES_TEMPLATE = (
    ",".join((*BOOK_COLUMNS, "policy_ids")) + "\n"
    "Example office,ARE,25.2,55.27,commercial,100000000,USD,45000,40,POL-001\n"
)


def book_files() -> dict[str, Path]:
    """Every CSV in the books folder, by file name without the extension, then this session's uploads."""
    files = sorted(BOOKS_DIR.glob("*.csv"), key=lambda f: f.stem.lower()) if BOOKS_DIR.is_dir() else []
    session_books = sorted((session_dir() / "books").glob("*.csv"))
    return {f.stem: f for f in files} | {f.stem: f for f in session_books}


def policies_file(book: str, path: Path) -> Path | None:
    """This session's uploaded policies for the book, else the book's bundled policies file."""
    for candidate in (session_dir() / "policies" / f"{book}.csv", path.parent / "policies" / path.name):
        if candidate.exists():
            return candidate
    return None


def take_uploads() -> None:
    """Store newly uploaded properties/policies in this session's folder; never on shared disk."""
    pending = st.session_state.pop("pending_book", None)
    if pending:
        reset_for_new_book()
        st.session_state.book = pending
    upload = st.session_state.get("upload_properties")
    if upload is not None and st.session_state.get("taken_properties") != upload.file_id:
        st.session_state.taken_properties = upload.file_id
        try:
            frame = pl.read_csv(upload.getvalue())
            missing = [c for c in BOOK_COLUMNS if c not in frame.columns]
            if missing:
                raise ValueError(f"missing column(s) {', '.join(missing)}")
        except (ValueError, pl.exceptions.PolarsError) as error:
            st.session_state.upload_error = f"Could not read {upload.name}: {error}"
        else:
            folder = session_dir() / "books"
            folder.mkdir(exist_ok=True)
            stem = "".join(c if c.isalnum() or c in "-_ " else "_" for c in Path(upload.name).stem) or "portfolio"
            (folder / f"{stem}{UPLOADED}.csv").write_bytes(upload.getvalue())
            reset_for_new_book()
            st.session_state.book = stem + UPLOADED
            st.session_state.pop("upload_error", None)
    upload = st.session_state.get("upload_policies")
    if upload is not None and st.session_state.get("taken_policies") != upload.file_id:
        st.session_state.taken_policies = upload.file_id  # applies to the book selected when uploaded
        try:
            sc.load_policies(upload.getvalue())
        except (ValueError, pl.exceptions.PolarsError) as error:
            st.session_state.upload_error = f"Could not read {upload.name}: {error}"
        else:
            folder = session_dir() / "policies"
            folder.mkdir(exist_ok=True)
            (folder / f"{st.session_state.get('book')}.csv").write_bytes(upload.getvalue())
            for key in ("baseline_results", "scenario_results", "run_error"):
                st.session_state.pop(key, None)
            st.session_state.pop("upload_error", None)


def reset_for_new_book() -> None:
    """Assumptions, edits and results belong to one book, so a new book starts clean."""
    for key in (
        "assumptions",
        "baseline_results",
        "scenario_results",
        "countries_editor",
        "sites_editor",
        "scenario_save_name",
        "view",
        "run_requested",
        "baseline_requested",
        "run_error",
    ):
        st.session_state.pop(key, None)


take_uploads()
books = book_files()
if not books:
    st.title("Silfio")
    st.info(f"No books found. Put one portfolio CSV per book in `{BOOKS_DIR}` and reload.")
    st.stop()
if st.session_state.get("book") not in books:
    reset_for_new_book()
    st.session_state.book = next(iter(books))
book_path = books[st.session_state.book]
try:
    portfolio = load_portfolio(str(book_path), book_path.stat().st_mtime)
    missing = [c for c in BOOK_COLUMNS if c not in portfolio.columns]
    if missing:
        raise ValueError(f"missing column(s) {', '.join(missing)}")
except (OSError, ValueError, pl.exceptions.PolarsError) as error:  # a bad file should not hide the other books
    portfolio = None
    book_error = error
programme, programme_error = None, None
policy_path = policies_file(st.session_state.book, book_path)
if policy_path is not None and portfolio is not None:
    try:
        programme = sc.load_policies(policy_path)
        programme = programme if programme and programme[0] else None
        sc.build_payload(portfolio, programme=programme)  # checks every property's policy_ids
    except (OSError, ValueError, pl.exceptions.PolarsError) as error:
        programme, programme_error = None, error
treaties = programme[1] if programme else []

with st.sidebar:
    st.title("Silfio")
    # Render the key box on every run unless a key is configured: a widget skipped for one run
    # loses its value, and a sidebar that changes shape resets the tabs below it.
    if not CONFIGURED_API_KEY:
        API_KEY = st.text_input(
            "Silfio API key",
            type="password",
            key="api_key",
            help=f"Optional for a baseline of up to {FREE_MAX_PROPERTIES} properties. "
            "Scenario testing and larger portfolios need a key.",
        )
    st.selectbox(
        "Book",
        list(books),
        key="book",
        on_change=reset_for_new_book,
    )
    with st.expander("Upload portfolio"):
        st.file_uploader("Properties CSV", type="csv", key="upload_properties")
        st.file_uploader("Policies CSV (optional)", type="csv", key="upload_policies")
        st.caption("Map properties to policies with policy_ids; policies to treaties with reinsurance_ids.")
        st.download_button(
            "Properties template",
            PROPERTIES_TEMPLATE,
            "properties_template.csv",
            "text/csv",
            on_click="ignore",
            width="stretch",
        )
        st.download_button(
            "Policies template",
            sc.POLICY_TEMPLATE,
            "policies_template.csv",
            "text/csv",
            on_click="ignore",
            width="stretch",
        )
        if st.session_state.get("upload_error"):
            st.error(st.session_state.upload_error)
if portfolio is None:
    st.error(f"Could not read {book_path.name}: {book_error}")
    st.stop()
countries = tuple(sorted(portfolio["iso3"].unique()))

by_country = portfolio.group_by("iso3").agg(pl.len().alias("sites")).sort("iso3")
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
        "sites": portfolio.select("property_name_or_address", "iso3", "occupancy", "value", "currency").with_columns(
            pl.lit(None, dtype=pl.Float64).alias("location_factor"),
            pl.lit(None, dtype=pl.Float64).alias("occupancy_factor"),
            pl.lit(None, dtype=pl.Float64).alias("standalone_probability_pct"),
        ),
    }

# Refresh input display metadata for sessions created before currency labels were added.
if "currency" not in st.session_state.assumptions["sites"].columns:
    st.session_state.assumptions["sites"] = st.session_state.assumptions["sites"].join(
        portfolio.select("property_name_or_address", "currency"), on="property_name_or_address", how="left"
    )
    st.session_state.pop("sites_editor", None)
if "value" in st.session_state.assumptions["countries"].columns:
    st.session_state.assumptions["countries"] = st.session_state.assumptions["countries"].drop("value")
    st.session_state.pop("countries_editor", None)

# Remove the retired selector from assumptions saved by an older browser session.
if "conflict_type" in st.session_state.assumptions["countries"].columns:
    st.session_state.assumptions["countries"] = st.session_state.assumptions["countries"].drop("conflict_type")
    st.session_state.pop("countries_editor", None)
    st.session_state.pop("scenario_results", None)


@st.dialog("Scenario assumptions", width="large")
def edit_assumptions() -> None:
    st.caption("Blank cells keep the model assumption.")
    current = st.session_state.assumptions
    tab_countries, tab_sites = st.tabs(["Countries", "Sites"])
    with tab_countries:
        countries_edit = st.data_editor(
            current["countries"].to_arrow(),
            hide_index=True,
            width="stretch",
            height=min(38 + 35 * len(current["countries"]), 520),
            disabled=["iso3", "sites"],
            column_config={
                "iso3": "Country",
                "sites": st.column_config.NumberColumn("Sites"),
                "onset_pct": st.column_config.NumberColumn("Onset %", min_value=0, max_value=100),
                "damage_pct": st.column_config.NumberColumn("Damage %", min_value=0, max_value=100),
            },
            key="countries_editor",
        )
    with tab_sites:
        sites_edit = st.data_editor(
            current["sites"].to_arrow(),
            column_order=[
                "property_name_or_address",
                "iso3",
                "occupancy",
                "value",
                "currency",
                "standalone_probability_pct",
                "location_factor",
                "occupancy_factor",
            ],
            hide_index=True,
            width="stretch",
            height=min(38 + 35 * len(current["sites"]), 520),
            disabled=["property_name_or_address", "iso3", "occupancy", "value", "currency"],
            column_config={
                "property_name_or_address": "Site",
                "iso3": "Country",
                "occupancy": "Occupancy",
                "value": st.column_config.NumberColumn("Insured value (original currency)", format="%,.0f"),
                "standalone_probability_pct": st.column_config.NumberColumn(
                    "Annual targeted-strike probability override (%)",
                    min_value=0,
                    max_value=100,
                    help="Replaces the modelled sub-threshold probability; 0 disables it",
                ),
                "location_factor": st.column_config.NumberColumn(
                    "Location ×", min_value=0, max_value=100, help="Relative to the country average"
                ),
                "occupancy_factor": st.column_config.NumberColumn(
                    "Occupancy ×", min_value=0, max_value=100, help="Relative to residential"
                ),
            },
            key="sites_editor",
        )
    st.divider()
    scenario_name = st.text_input("Scenario name", key="scenario_save_name")
    if DOWNLOAD_SCENARIOS:
        scenario_download(scenario_name, countries_edit, sites_edit)
    else:
        st.caption("Saving under an existing name replaces it.")
    if not DOWNLOAD_SCENARIOS and st.button("Save scenario to file", width="stretch"):
        try:
            scenario_store.save(
                scenarios_file(),
                scenario_name,
                {"countries": pl.from_arrow(countries_edit), "sites": pl.from_arrow(sites_edit)},
            )
        except (OSError, ValueError) as error:
            st.error(f"Could not save scenario: {error}")
        else:
            st.success(f"Saved {scenario_name.strip()} to {SCENARIOS_FILE.relative_to(HERE)}.")
    apply, cancel = st.columns(2)
    if apply.button("Apply changes", type="primary", width="stretch"):
        st.session_state.assumptions = {"countries": pl.from_arrow(countries_edit), "sites": pl.from_arrow(sites_edit)}
        st.session_state.pop("scenario_results", None)
        st.session_state.pop("run_error", None)
        st.rerun()
    if cancel.button("Cancel", width="stretch"):
        st.rerun()


def scenario_download(name: str, countries_edit, sites_edit) -> None:
    """Hosted mode: the save button downloads the scenario as CSV; nothing is stored on the server."""
    data, error = b"", None
    if name.strip():
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scenario.csv"
            try:
                scenario_store.save(
                    path, name, {"countries": pl.from_arrow(countries_edit), "sites": pl.from_arrow(sites_edit)}
                )
                data = path.read_bytes()
            except (OSError, ValueError) as err:
                error = err
    if error:
        st.error(f"Could not save scenario: {error}")
    filename = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip()) or "scenario"
    st.download_button(
        "Save scenario (download CSV)",
        data,
        file_name=f"{filename}.csv",
        mime="text/csv",
        on_click="ignore",
        disabled=not data,
        help=None if data else "Enter a scenario name first.",
        width="stretch",
    )


def scenario_upload() -> None:
    """Hosted mode: load scenarios from a CSV saved earlier; it is kept only for this browser session."""
    upload = st.file_uploader("Load scenarios from CSV", type="csv", key="scenarios_upload")
    if upload is not None and st.session_state.get("scenarios_uploaded") != upload.file_id:
        candidate = scenarios_file().with_name("upload.csv")
        candidate.write_bytes(upload.getvalue())
        try:
            scenario_store.load(candidate)
        except (OSError, ValueError) as error:
            candidate.unlink(missing_ok=True)
            st.error(f"Could not read that file: {error}")
        else:
            os.replace(candidate, scenarios_file())
            st.session_state.scenarios_uploaded = upload.file_id
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


@st.dialog("Edit portfolio", width="large")
def edit_portfolio() -> None:
    book = st.session_state.book
    properties = pl.read_csv(book_path)
    if "policy_ids" not in properties.columns:
        properties = properties.with_columns(pl.lit(None, dtype=pl.String).alias("policy_ids"))
    tab_properties, tab_policies = st.tabs(["Properties", "Policies"])
    with tab_properties:
        properties_edit = st.data_editor(
            properties.to_arrow(),
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "occupancy": st.column_config.SelectboxColumn("occupancy", options=sc.OCCUPANCIES),
                "policy_ids": st.column_config.TextColumn("policy_ids", help="Separate several with ;"),
            },
            key=f"edit_properties_{book}",
        )
    with tab_policies:
        policies_edit = st.data_editor(
            sc.policies_frame(policy_path).to_arrow(),
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "contract_type": st.column_config.SelectboxColumn(options=["insurance", "reinsurance"]),
                "deductible_type": st.column_config.SelectboxColumn(options=["excess", "franchise"]),
                "reinsurance_structure": st.column_config.SelectboxColumn(options=["quota_share", "xol"]),
                "reinsurance_basis": st.column_config.SelectboxColumn(
                    options=["per_risk", "per_occurrence", "aggregate"]
                ),
                "reinsurance_ids": st.column_config.TextColumn(help="Separate several with ;"),
            },
            key=f"edit_policies_{book}",
        )
        st.caption("No policies: ground-up only.")
    properties_csv = pl.from_arrow(properties_edit).filter(~pl.all_horizontal(pl.all().is_null())).write_csv()
    policies_df = pl.from_arrow(policies_edit).filter(pl.col("policy_id").is_not_null())
    policies_csv = policies_df.write_csv() if not policies_df.is_empty() else ""
    down_left, down_right = st.columns(2)
    down_left.download_button(
        "Download properties", properties_csv, "properties.csv", "text/csv", on_click="ignore", width="stretch"
    )
    down_right.download_button(
        "Download policies",
        policies_csv or ",".join(sc.POLICY_COLUMNS),
        "policies.csv",
        "text/csv",
        on_click="ignore",
        width="stretch",
    )
    apply, cancel = st.columns(2)
    if apply.button("Apply", type="primary", width="stretch"):
        try:
            edited = pl.read_csv(properties_csv.encode())
            missing = [c for c in BOOK_COLUMNS if c not in edited.columns]
            if missing:
                raise ValueError(f"missing column(s) {', '.join(missing)}")
            if edited.is_empty():
                raise ValueError("add at least one property")
            blank = [c for c in BOOK_COLUMNS if edited[c].null_count()]
            if blank:
                raise ValueError(f"fill in {', '.join(blank)} for every property")
            if edited["property_name_or_address"].is_duplicated().any():
                raise ValueError("property names must be unique")
            check = sc.load_portfolio(edited.with_columns(pl.col("policy_ids").cast(pl.String)))
            programme = sc.load_policies(policies_csv.encode()) if policies_csv else None
            sc.build_payload(check, programme=programme)
        except (ValueError, pl.exceptions.PolarsError) as error:
            st.error(f"Not applied: {error}")
            return
        target = book if book_path.parent == session_dir() / "books" else book + EDITED
        for folder, data in (("books", properties_csv), ("policies", policies_csv)):
            path = session_dir() / folder / f"{target}.csv"
            path.parent.mkdir(exist_ok=True)
            if data:
                path.write_text(data)
            else:
                path.unlink(missing_ok=True)
        st.session_state.pop(f"edit_properties_{book}", None)
        st.session_state.pop(f"edit_policies_{book}", None)
        st.session_state.pending_book = target
        st.rerun()
    if cancel.button("Cancel", width="stretch"):
        st.session_state.pop(f"edit_properties_{book}", None)
        st.session_state.pop(f"edit_policies_{book}", None)
        st.rerun()


with st.sidebar:
    tab_book, tab_scenario = st.tabs(["Baseline", "Scenario"], key="sidebar_tab", on_change="rerun")
    with tab_book:
        exposure_summary = st.empty()
        st.caption(f"{len(portfolio)} sites · {len(countries)} countries")
        if programme:
            n = len(programme[0])
            st.caption(
                f"{n} {'policy' if n == 1 else 'policies'}"
                + (" · " + "; ".join(sc.treaty_label(t) for t in treaties) if treaties else "")
            )
        else:
            st.caption("No policies: ground-up only")
        if programme_error:
            st.warning(f"Policies not used: {programme_error}")
        if st.button("Edit portfolio", width="stretch"):
            edit_portfolio()
        baseline_requested = st.button("Run baseline", type="primary", width="stretch")
    with tab_scenario:
        try:
            saved_scenarios = scenario_store.load(scenarios_file())
        except (OSError, ValueError) as error:
            st.warning(f"Could not read saved scenarios: {error}")
            saved_scenarios = {}
        if saved_scenarios:
            saved_name = st.selectbox("Saved scenarios", sorted(saved_scenarios))
            if st.button("Load saved scenario", width="stretch"):
                st.session_state.assumptions = scenario_store.apply(
                    saved_scenarios[saved_name], st.session_state.assumptions
                )
                for key in ("countries_editor", "sites_editor", "scenario_results", "run_error"):
                    st.session_state.pop(key, None)
                st.session_state.scenario_save_name = saved_name
                st.session_state.view = "Baseline"
                st.rerun()
        elif not DOWNLOAD_SCENARIOS:
            st.caption("No saved scenarios yet.")
        if DOWNLOAD_SCENARIOS:
            scenario_upload()
        lines = assumption_lines(st.session_state.assumptions)
        if lines:
            st.markdown("\n".join(f"- {line}" for line in lines))
        else:
            st.caption("No overrides set.")
        if st.button("Edit assumptions", width="stretch"):
            edit_assumptions()
        if lines and st.button("Clear scenario", width="stretch"):
            del st.session_state.assumptions
            st.session_state.pop("scenario_results", None)
            st.session_state.pop("run_error", None)
            st.rerun()

        scenario_requested = st.button("Run scenario", type="primary", width="stretch")
        scenario_status = st.empty()
# Reserve the same map position on every rerun so React keeps its viewport alive.
summary_area = st.container()
map_area = st.container()  # Do not emit an empty element that unmounts the map during a run.
details_area = st.container()

if baseline_requested or scenario_requested:
    st.session_state.pop("run_error", None)
    if not API_KEY and scenario_requested:
        st.session_state.run_error = (
            "Scenario testing needs a Silfio API key with Professional access or above. "
            "Enter one in the sidebar, then press Run scenario."
        )
    elif not API_KEY and portfolio.height > FREE_MAX_PROPERTIES:
        st.session_state.run_error = (
            f"This book has {portfolio.height} properties. Without an API key, baselines are limited to "
            f"{FREE_MAX_PROPERTIES} properties. Enter a Silfio API key in the sidebar, or choose a smaller book."
        )
    else:
        theatres_version.clear()
        try:
            if baseline_requested:
                baseline = run_model(sc.build_payload(portfolio, programme=programme))
                st.session_state.baseline_results = baseline
                st.session_state.view = "Baseline"
            if scenario_requested:
                assumptions = st.session_state.assumptions
                payload = sc.build_payload(
                    portfolio, assumptions["sites"], assumptions["countries"], programme=programme
                )
                st.session_state.scenario_results = run_model(payload)
                st.session_state.view = "Scenario"
        except Exception as error:  # noqa: BLE001 -- keep the UI available when a run fails
            st.session_state.run_error = f"Silfio could not run: {error}"

if "scenario_results" not in st.session_state:
    scenario_status.caption("Not run.")
else:
    scenario_status.caption("Results available.")

# Render after processing a run so the sidebar updates immediately with its USD values.
exposure_tables = st.session_state.get(
    "scenario_results" if st.session_state.get("view") == "Scenario" else "baseline_results", {}
)
usd_values = sc.exposure_values(portfolio, exposure_tables)["value_usd"]
with exposure_summary.container():
    if usd_values.null_count() == 0:
        st.metric("Insured value (USD)", money(usd_values.sum()))
    else:
        for currency, amount in portfolio.group_by("currency").agg(pl.col("value").sum()).sort("currency").iter_rows():
            st.metric(f"Insured value ({currency})", f"{amount:,.0f}")

with summary_area:
    if st.session_state.get("run_error"):
        st.error(st.session_state.run_error)
        if "baseline_results" in st.session_state or "scenario_results" in st.session_state:
            st.caption("Showing the last completed results.")

    views = {
        name: st.session_state[key]
        for name, key in (("Baseline", "baseline_results"), ("Scenario", "scenario_results"))
        if key in st.session_state
    }
    if not views:
        st.title("Portfolio")
        locations = [
            {
                "position": [row["longitude"], row["latitude"]],
                "radius": 7,
                "colour": [125, 135, 150, 220],
                "label": (
                    f"{row['property_name_or_address']} · {row['iso3']}\n"
                    f"Insured value: {row['value']:,.0f} {row['currency']}\n"
                    "Loss cost: not calculated"
                ),
            }
            for row in portfolio.iter_rows(named=True)
        ]
        map_area.pydeck_chart(
            pdk.Deck(
                map_style="light",
                initial_view_state=fit_view(portfolio),
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        locations,
                        id="portfolio-locations",
                        transitions={"getFillColor": 600, "getRadius": 600},
                        get_position="position",
                        get_radius="radius",
                        radius_units='"pixels"',
                        get_fill_color="colour",
                        get_line_color=[255, 255, 255],
                        line_width_min_pixels=1,
                        stroked=True,
                        pickable=True,
                    )
                ],
                tooltip={"text": "{label}"},
            ),
            height=560,
        )
        details_area.dataframe(sc.sort_locations(portfolio), hide_index=True, width="stretch")
        st.stop()

    baseline = st.session_state.get("baseline_results")
    if st.session_state.get("view") not in views:
        st.session_state.view = next(iter(views))

    left, right = st.columns([3, 1])
    view = left.radio("View", list(views), horizontal=True, label_visibility="collapsed", key="view")
    show_theatres = right.toggle("Theatres")
    tables = views[view]
    scenario_shares = tables.get("scenario_theatre_shares") if view == "Scenario" else None
    if view == "Scenario" and scenario_shares is None:
        st.info("Missing scenario shares: update the API, then press Run scenario again.")
    base_kpi = sc.headline(baseline) if baseline is not None else {}
    kpi = sc.headline(tables)
    if view == "Scenario" and baseline is None:
        st.info("Run baseline in the Baseline tab to see comparison changes.")

    metrics = [
        ("Ground-up expected loss", "expected_loss"),
        ("Ground-up VaR 99%", "var_99"),
        ("Ground-up TVaR 99%", "tvar_99"),
    ]
    if kpi.get("net_expected_loss") is not None:
        metrics.append(("Insurer net expected loss", "net_expected_loss"))
    if treaties or (kpi.get("reinsurer_expected_loss") or 0) > 0:
        metrics.append(("Reinsurer expected loss", "reinsurer_expected_loss"))
    for col, (label, key) in zip(st.columns(len(metrics)), metrics):
        value, base_value = kpi.get(key), base_kpi.get(key)
        delta = None
        if view != "Baseline" and value is not None and base_value is not None:
            delta = money(value - base_value)
        col.metric(label, money(value) if value is not None else "N/A", delta, delta_color="inverse")
    st.caption(
        f"{view} · annual, USD" + (" · change vs baseline" if view == "Scenario" and baseline is not None else "")
    )

ongoing_view = subthreshold_view = False
try:
    version = theatres_version()
    has_ongoing = fetch_theatres(countries, version)[1]["ongoing_share"].is_not_null().any()
    if show_theatres:
        choices = ["New conflict"] + (["Ongoing conflicts"] if has_ongoing else []) + ["Sub-threshold"]
        split = right.selectbox("Theatre split", choices)
        ongoing_view, subthreshold_view = split == "Ongoing conflicts", split == "Sub-threshold"
    polygons, share_labels, centres = theatre_layers(
        countries, version, ongoing_view, LABEL_MIN_SHARE, scenario_shares, subthreshold_view
    )
    polygons_file = HERE / polygons.removeprefix("app/")
    try:
        os.utime(polygons_file)  # mark as in use, so other sessions' cleanup keeps it
    except FileNotFoundError:
        pass
    if not polygons_file.exists():  # file cleared since it was cached
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
        summary_area.warning(f"Theatres unavailable: {error}")
sites = sc.site_results(portfolio, tables)
sites = sites.with_columns(pl.Series("theatre_share", site_theatre_share(sites, centres)))
theatre_measure = "sub-threshold incidents" if subthreshold_view else "fighting"
usd_sizes = sites["value_usd"].fill_null(0.0).to_numpy()
radius = 5 + 9 * np.sqrt(usd_sizes / usd_sizes.max()) if usd_sizes.max() > 0 else np.full(len(sites), 5.0)
site_data = [
    {
        "position": [lon, lat],
        "radius": r,
        "colour": c,
        "label": (f"{name}: {bps:,.1f} bps" if bps is not None else f"{name}: loss cost unavailable")
        + ("" if np.isnan(s) else f" · theatre {s:.1%} of {theatre_measure}"),
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
        id="portfolio-locations",
        transitions={"getFillColor": 600, "getRadius": 600},
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

map_area.pydeck_chart(
    pdk.Deck(
        map_style="light",
        initial_view_state=fit_view(portfolio),
        layers=layers,
        tooltip={"text": "{label}"},
    ),
    height=560,
)
caption = "Size: insured value. Colour: loss cost, 0.1–200+ bps."
if show_theatres:
    caption += f" Shading: theatre share of {theatre_measure}."
with details_area:
    st.caption(caption)

    st.dataframe(
        sc.sort_locations(sites).select(
            "region", "iso3", "property_name_or_address", "value_usd", "expected_loss", "bps"
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "value_usd": st.column_config.NumberColumn("Insured value (USD)", format="$%,.0f"),
            "expected_loss": st.column_config.NumberColumn("Expected loss (USD)", format="$%,.0f"),
            "bps": st.column_config.NumberColumn("loss cost (bps)", format="%.1f"),
            "region": "Region",
            "iso3": "Country",
            "property_name_or_address": "Property",
        },
    )

    st.subheader("Pricing details")
    if "rating_factors" not in tables:
        st.info("Rating factors unavailable for this run.")
    details = sc.sort_locations(
        sc.pricing_details(portfolio, tables).join(
            sites.select("property_name_or_address", "theatre_share"),
            on="property_name_or_address",
            how="left",
            validate="1:1",
        )
    )
    st.dataframe(
        details,
        hide_index=True,
        width="stretch",
        column_order=[
            "region",
            "iso3",
            "property_name_or_address",
            "building_type",
            "value_usd",
            "conflicts_simulated_ratio",
            "ongoing_conflicts",
            "mean_damage_conditional",
            "location_factor",
            "occupancy_factor",
            "hit_frequency",
            "theatre_share",
            "subthreshold_share",
            "vulnerability_if_hit",
            "expected_ground_up_loss",
            "loss_cost_bps",
            "ground_up_tvar_99",
        ],
        column_config={
            "property_name_or_address": "Property",
            "iso3": "Country",
            "region": "Region",
            "theatre_share": st.column_config.NumberColumn(
                "Theatre share",
                format="percent",
                help=f"Share of the country's {theatre_measure} in this site's theatre, using the current theatre split",
            ),
            "building_type": "Model occupancy class",
            "value_usd": st.column_config.NumberColumn("Insured value (USD)", format="$%,.0f"),
            "conflicts_simulated_ratio": st.column_config.NumberColumn(
                "P(conflict)", format="percent", help="Annual probability of conflict, new or ongoing"
            ),
            "ongoing_conflicts": st.column_config.NumberColumn("Ongoing conflicts", format="%d"),
            "mean_damage_conditional": st.column_config.NumberColumn("Country damage if conflict", format="percent"),
            "location_factor": st.column_config.NumberColumn("Location relativity", format="%.3f ×"),
            "occupancy_factor": st.column_config.NumberColumn("Occupancy relativity", format="%.3f ×"),
            "hit_frequency": st.column_config.NumberColumn("P(damage)", format="percent"),
            "subthreshold_share": st.column_config.NumberColumn(
                "Sub-threshold share",
                format="percent",
                help="Share of damage years attributable to sub-threshold events",
            ),
            "vulnerability_if_hit": st.column_config.NumberColumn("Vulnerability if hit", format="percent"),
            "expected_ground_up_loss": st.column_config.NumberColumn("Expected loss (USD)", format="$%,.0f"),
            "loss_cost_bps": st.column_config.NumberColumn("Loss cost (bps)", format="%.2f"),
            "ground_up_tvar_99": st.column_config.NumberColumn("TVaR 99% (USD)", format="$%,.0f"),
        },
    )
