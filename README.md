# Silfio dashboard

A lightweight Streamlit client for the Silfio API: the portfolio in `portfolio.csv` on a world map, with baseline and scenario runs.

## Run

Install `requirements.txt` into the project interpreter, then run `run.py` like any script (PyCharm's Run button, or `python run.py`). It starts Streamlit and opens the app in your browser; `streamlit run app.py` also works.

The API key is read from `.streamlit/secrets.toml` (copy `secrets.toml.example`) or the `SILFIO_API_KEY` environment variable. If neither is set, the app asks for it on screen.

Scenario runs need a Professional key or above. Portfolio data is sent to the API on each run and is not stored.

## Portfolio

`portfolio.csv`, one row per site:

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

- **Countries:** onset probability, conflict type and damaged share of buildings per country. Blank keeps the model's own value.
- **Sites:** location factor (multiple of the country-average property) and occupancy factor (multiple of residential).
- **Theatres:** turn on **Theatres**, then **Edit theatres**, pick a country and click a theatre to set its share of the country's fighting. The country's other theatres scale down so the shares still add up to 100%.

Countries and sites are set under **Edit assumptions** in the Scenario tab and sent when you press **Apply and run**; theatres are sent with **Run scenario** under the map. Results are saved on disk and reused until the next day or until you press **Refresh results**.

## Theatres

The **Theatres** toggle shades each portfolio country's theatres by the model's share of the fighting there; hover for any theatre's share, and the larger ones are labelled. Where a conflict is under way, a second option shows where the model sends the fighting if it continues. The site table shows the share of the theatre each site falls in.

Theatres come from the API's `/theatres` endpoint (Professional plan and above), so they always match the deployed model. The dashboard checks `/theatres/version` (a few bytes) on opening, every ten minutes and whenever a run starts, and downloads only when the version has changed: after a deploy that rebuilt the theatres, or when the daily job updates the ongoing-conflict shares. Downloads are kept on disk, so restarts reuse them.
