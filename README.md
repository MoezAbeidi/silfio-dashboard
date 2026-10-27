# Silfio dashboard

A lightweight Streamlit client for the Silfio API: the portfolio in `portfolio.csv` on a world map, with baseline and scenario runs.

Portfolio loading, API result tables, joins, and summaries use Polars. Scenario
editors exchange Arrow tables with Streamlit and convert edits back to Polars.
Streamlit may still install pandas as an internal dependency.

## Run

Install `requirements.txt` into the project interpreter, then run `run.py` like any script (PyCharm's Run button, or `python run.py`). It starts Streamlit and opens the app in your browser; `streamlit run app.py` also works.

The API key is read from `.streamlit/secrets.toml` (copy `secrets.toml.example`) or the `SILFIO_API_KEY` environment variable. If neither is set, the app asks for it on screen.

Scenario runs need a Professional key or above. Portfolio data is sent to the API on each run and is not stored.

## Portfolio

Click **View locations** in the Portfolio tab to open the full locations table in a wide popup.

`portfolio.csv`, one row per site:

The example book contains 58 fictional assets in 42 countries, including Yemen,
Algeria, Turkey, and locations across Asia, Africa, North America, Europe, South
America, and Australia. Coordinates approximate city centres; insured values and
building details are illustrative. All example values are in USD.

| column | |
|---|---|
| `property_name_or_address` | site name, unique |
| `city` | display only |
| `iso3` | ISO 3166-1 alpha-3 country |
| `latitude`, `longitude` | decimal degrees |
| `occupancy` | Silfio occupancy class, e.g. `commercial`, `critical_industrial`, `transport_hub` |
| `value`, `currency` | insured value |
| `property_sqm`, `building_floors` | building size |
| `deductible_pct` | optional, default 0.01 |

Each site gets its own policy with the deductible as a per-occurrence attachment.

## Scenario controls

- **Countries:** onset probability and damaged share of buildings per country. Blank keeps the model's own value. Conflict type is left to the model.
- **Sites:** location factor (multiple of the country-average property) and occupancy factor (multiple of residential). A location factor moves the site's whole theatre: its neighbours rise or fall with it, and the rest of the country adjusts so country damage is unchanged.

Set these under **Edit assumptions** in the Scenario tab; nothing is sent until you press **Apply and run**. Results are saved on disk and reused until the next day or until you press **Refresh results**.

Onset overrides send only `annual_prob`; the API must support an omitted
`conflict_type` so the simulator retains its modelled conflict-type probabilities.

## Theatres

The **Theatres** toggle shades each portfolio country's theatres by the model's share of the fighting there; hover for any theatre's share, and the larger ones are labelled. Where a conflict is under way, a second option shows where the model sends the fighting if it continues. The site table shows the share of the theatre each site falls in.

Theatres come from the API's `/theatres` endpoint (Professional plan and above), so they always match the deployed model. The dashboard checks `/theatres/version` (a few bytes) on opening, every ten minutes and whenever a run starts, and downloads only when the version has changed: after a deploy that rebuilt the theatres, or when the daily job updates the ongoing-conflict shares. Downloads are kept on disk, so restarts reuse them.

Scenario maps and site shares use the API's `scenario_theatre_shares`; Baseline
uses `/theatres`. These are scenario centres, before simulation variation and
feasibility blending. After deployment, refresh results and rerun saved scenarios.

## Development

Install development tools and enable automatic Python formatting on commit:

```sh
python -m pip install -r requirements-dev.txt
pre-commit install
```

Run `pre-commit run --all-files` to format all tracked Python files, or
`ruff format .` to format directly. The formatter uses Ruff with a 120-character line length.

Run regression tests with `python -m unittest discover -s tests -v`.
