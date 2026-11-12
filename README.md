# Silfio dashboard

A local dashboard for exploring armed-conflict risk across a property portfolio.
Use it to review expected and tail losses, identify exposed locations, inspect
the assumptions behind each property's loss, and compare scenarios with a model
baseline.

Silfio models conflict occurrence, severity, the geographic distribution of damage
and building vulnerability, including sub-threshold violence. This dashboard
provides a focused interface to those results for portfolio review and underwriting
discussions.

**The dashboard runs on your computer; model calculations run through the Silfio
API.** An internet connection and API access are required to calculate results.
You do not need to install the main Silfio modelling repository.

## Quick setup

Use Python 3.10 or later. Download or clone this project, open a terminal in its
root folder, and create a virtual environment:

```sh
python -m venv .venv
```

Activate it on macOS/Linux:

```sh
source .venv/bin/activate
```

Or in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies and start the dashboard:

```sh
python -m pip install -r requirements.txt
python run.py
```

If your system uses `python3`, use that command to create the environment.
The dashboard opens in your browser. You can also start it with
`streamlit run app.py`. Stop it with **Ctrl+C** in the terminal.

### API access

Enter your own API key in the sidebar when prompted. To retain it between
sessions, create `.streamlit/secrets.toml` with:

```toml
SILFIO_API_KEY = "your-api-key"
```

Alternatively, set the `SILFIO_API_KEY` environment variable. The local secrets
file is excluded from Git; do not include it when sharing the project.

**Scenario testing and larger portfolios require an API key with the appropriate
access level.** The API reference supplied with the model describes:

| Access level | Property portfolio size | Scenario testing |
|---|---|---|
| Free API | Up to 10 properties | Not included |
| Professional | Up to 100 properties | Included |
| Team | Larger portfolios, subject to service-wide limits | Included |
| Enterprise | Agreed limits | Subject to the agreed access configuration |

The API supports keyless free requests, but **this dashboard currently requires a
key to start a run**. The included 58-property example needs access that covers
that portfolio size. Confirm the suitable access level with Silfio when arranging
an evaluation or local implementation; the service enforces your key's limits.

## First run

1. Select a supplied example portfolio from **Book**, or add your own CSV as
   described below. Book names are taken directly from the files in `books/`.
2. Review the location map, available immediately with neutral markers before any
   calculation. Hover for property details, or open **View locations** to check
   the input data.
3. In the **Baseline** tab, click **Run baseline** to calculate the baseline.
   Opening the dashboard or selecting a book does not run the model.
4. Review the headline losses, map, location breakdown and **Pricing details**.
5. Open the **Scenario** tab to test alternative assumptions, then compare
   **Baseline** and **Scenario** using the view selector above the results.

If a run fails, the portfolio and controls remain available. Any previously
completed results remain visible alongside the error, so you can correct the issue
and retry.

## Add and manage portfolios

Each CSV in `books/` is a separate **book**, named after its file. For example,
`books/property_portfolio.csv` appears as **property_portfolio** in the selector.
Add the file and reload the dashboard. Switching books clears the active
assumptions and results; it does not delete scenarios saved to file.

The fictional USD portfolios in `books/` can serve as input templates. The
dashboard discovers CSV files automatically, so you can add or rename books
without changing the code. Use one row per property:

| Column | Required | Description |
|---|---|---|
| `property_name_or_address` | Yes | Unique property name or address; also used to match saved scenario overrides. |
| `iso3` | Yes | Three-letter country code, for example `GBR` or `USA`. |
| `latitude`, `longitude` | Yes | Property coordinates in decimal degrees. |
| `occupancy` | Yes | Silfio occupancy class; examples include `commercial`, `critical_industrial` and `transport_hub`. |
| `value`, `currency` | Yes | Insured property value and its currency code. |
| `property_sqm` | Yes | Building area in square metres. |
| `building_floors` | Yes | Number of building floors. |
| `city` | No | Additional location information. |
| `deductible_pct` | No | Per-occurrence deductible as a fraction of value; defaults to `0.01` (1%). |

Use numeric values without currency symbols or thousands separators, and save as
UTF-8 CSV. Keep property names unique and stable if you intend to reuse scenarios.

Books may contain any API-supported currency, including a mix of currencies.
Input views retain the original values and currency codes. Before a run, the
sidebar shows separate totals by currency; once converted values are available,
it shows a combined USD total. Result tables, map sizing and loss costs use the
API's USD property values consistently.

To use another portfolio folder, add an absolute path to your secrets file:

```toml
SILFIO_BOOKS_DIR = "/path/to/your/books"
```

## Understand the results

### Headline figures

The headline figures describe annual portfolio losses in USD for the selected
view. Scenario changes are shown relative to the baseline.

| Figure | Interpretation |
|---|---|
| **Ground-up expected loss** | Average annual property damage before insurance terms. |
| **Ground-up VaR 99%** | Annual aggregate loss at the 99th percentile, commonly described as a 1-in-100 loss. |
| **Ground-up TVaR 99%** | Average annual loss within the worst 1% of simulated years. |
| **Insurer net expected loss** | Average annual loss retained by the insurer after the financial terms represented in the run. |

The first three figures measure property damage, not the insurer's claim cost.
The fourth reflects insurance terms. Missing insurer results display **N/A**.
Hover over a figure's help icon for its definition.

This dashboard creates **one policy per property**, with 100% participation, an
excess deductible and an occurrence limit equal to the remaining insured value.
It does not submit reinsurance treaties, so insurer net here reflects those policy
terms without a reinsurance programme. Expected loss is a loss estimate, not a
quoted premium or a price including expenses, brokerage and profit margin.

### Map and location breakdown

The map opens with equal-sized, neutral markers before any run. Hover over a
marker for the property name, country and original-currency insured value.

After a run, circle size indicates insured value in USD; colour indicates annual
ground-up loss cost in basis points of value, from blue for lower loss costs to
red for higher loss costs. One basis point is 0.01%. Hover over a location for its
loss cost and, when available, theatre share.

The map updates in place as results arrive, with a short transition in marker
colour and size. Its initial camera settings stay the same for the selected book
so updates can retain the current pan and zoom. Zooming out is limited to the
full-world view.

The main table shows insured value, expected loss and loss cost. Location tables
and Pricing details retain one row per property and sort by **region → country
(ISO3) → property name**. Regions follow Silfio's model classification: Africa,
Americas, Asia, Europe, MENA and Oceania. Unrecognised codes appear as
**Unknown / other**.

### Pricing details

This table explains the property-level drivers behind the selected result:

| Field | Meaning |
|---|---|
| Model occupancy class | Building-use classification returned by the model. |
| Insured value (USD) | Property value in the API's reporting currency. |
| P(conflict) and ongoing conflicts | Country-level annual probability of new or ongoing conflict, and the count already under way at valuation. |
| Country damage if conflict | Average share of the country's buildings damaged, conditional on conflict. |
| Location relativity | Site location factor relative to the country's average building; `1` is average. |
| Occupancy relativity | Occupancy factor relative to residential; `1` is the residential reference. |
| P(damage) | Hit frequency: annual probability that the property is damaged. |
| Theatre share | Share of the country's fighting or sub-threshold incidents assigned to the site's theatre, using the selected split. |
| Sub-threshold share | Share of damage years attributed to sub-threshold events; not a separate annual damage probability. |
| Vulnerability if hit | Expected loss divided by USD insured value and damage probability. Undefined values remain blank. |
| Expected loss, loss cost and TVaR 99% | Property-level ground-up loss measures. |

Blank fields indicate unavailable results, not zero risk. If results lack rating
factors, press **Run baseline** or **Run scenario** to request them again.

### Theatres

Turn on **Theatres** to display geographic areas used to distribute damage within
each country. Shading and labels show their shares of the selected type of activity:

- **New conflict:** the distribution of fighting if a new conflict begins.
- **Ongoing conflicts:** the distribution for conflicts already under way, when
  available for the selected portfolio.
- **Sub-threshold:** the distribution of smaller incidents below full conflict.

Scenario theatre shares are used when supplied by the API. A direct site-level
sub-threshold probability override does not redistribute theatre shares. Theatre
shares describe where activity falls within a country; they are not the property's
probability of damage. If theatre data cannot be loaded, the location map and loss
results remain available.

## Test a scenario

Open **Scenario → Edit assumptions**. Leave a cell blank to retain the model's
assumption, or enter an override:

- **Country onset %:** annual probability that a conflict starts.
- **Country damage %:** share of buildings damaged if conflict occurs.
- **Location ×:** location factor relative to the country-average property. A
  location override can change the distribution across the whole theatre,
  affecting neighbouring properties as well as the selected site.
- **Occupancy ×:** occupancy factor relative to residential.
- **Annual targeted-strike probability override (%):** probability of a damaging
  sub-threshold incident at the site. This replaces the modelled sub-threshold
  probability; damage severity remains modelled. Enter `0` to disable that
  component for the site.

Enter percentages from **0 to 100** and factors as multipliers, such as `2`.
Click **Apply changes** to keep your edits, then **Run scenario** in the Scenario
tab to calculate. Loading, editing, saving and clearing assumptions do not run the
model. Scenario runs are independent of baseline runs; use **Run baseline** in the
Baseline tab when you want a comparison. The view switches to Scenario when results
are ready. Once both views have results, switch between them to compare losses,
maps and pricing details. Applying or loading assumptions clears the previous
scenario results; press **Run scenario** to calculate the updated assumptions.
**Clear scenario** removes the active overrides and scenario results.

### Save and reuse assumptions

In the editor, enter a **Scenario name** and click **Save scenario to file**.
Saving does not run the model. Saving under an existing name replaces it.

To reuse a scenario, select it under **Saved scenarios**, click **Load saved
scenario**, optionally review and **Apply changes** in **Edit assumptions**, then
click **Run scenario** in the sidebar.
Loading replaces the current assumptions but does not launch a calculation.

Scenarios are stored in `silfio_dashboard/data/scenarios.csv` and shared by users
of the same installation. They can be reused across books: country overrides match ISO3 codes
and site overrides match exact property names. Only matching entries are applied.
You can also edit this file in Excel, preserving its header and saving as UTF-8 CSV.
Blank values retain model assumptions; zero is an explicit override.

## Results, local files and connectivity

Each click on **Run baseline** or **Run scenario** requests fresh results from the
API for that view. Completed results remain in the current browser session while
you inspect the dashboard; changing views or editing assumptions does not start
another calculation. Reloading into a new session requires a new run.

Portfolio CSVs and saved scenarios are held locally. Calculating a new result
sends the portfolio, generated policy terms and any overrides to the configured
API. Local installation of this dashboard therefore does not mean local-only
processing. Confirm API data-handling arrangements with Silfio before
using live client data.

The default service address is configured in the application. If Silfio provides
another endpoint for your implementation, set `SILFIO_API_URL` in your secrets file
or environment. Hosting the modelling engine itself is a separate deployment.

## Troubleshooting

| Symptom | Action |
|---|---|
| No books found | Add a CSV to `books/`, or check `SILFIO_BOOKS_DIR`, then reload. |
| A book cannot be read | Check its header, required columns and numeric values. You can still select another book. |
| API key requested or `401` | Enter a valid key, or check the key configured in the secrets file or environment. |
| `403` | Check that the key permits the portfolio size and, for scenarios, scenario access. |
| `429` | Wait before retrying; the service's request rate limit has been reached. |
| Timeout or service error | Check connectivity and retry with **Run baseline** or **Run scenario**. Previously completed results remain visible. |
| Missing rating factors or theatre data | Press the relevant Run button again. Availability also depends on the configured API version and service data. |

## Scope for a local evaluation

This interface supports property portfolio review, geographic concentration
analysis and assumption-based scenario testing. It does not currently expose the
engine's investment modelling, shared multi-property policy structures, custom
insurance layers or reinsurance configuration. Those capabilities require a
separate integration with the wider API.

Results are probabilistic estimates under the model and supplied assumptions, not
predictions of a specific attack or guaranteed loss outcomes. Use the baseline and
scenarios alongside exposure validation and underwriting judgement. For a wider
implementation, agree portfolio size, financial structures, API access and data
handling with Silfio, and consult the main repository's API reference and model
whitepaper for further detail.

## Project layout

- `app.py` and `run.py`: dashboard interface and launcher.
- `silfio_dashboard/`: API client and scenario storage utilities.
- `silfio_dashboard/data/`: bundled reference data, including country regions.
- `books/`: portfolio input CSVs.
- `silfio_dashboard/data/scenarios.csv`: saved user scenarios.
