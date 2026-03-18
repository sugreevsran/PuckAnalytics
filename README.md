# NHL Roster Optimizer

A full data science pipeline for building cap-efficient NHL rosters. You set the constraints — cap limit, roster size, forced includes/excludes — and the optimizer finds the highest-value roster that fits.

Built as a practical exercise in end-to-end ML product development: data ingestion, feature engineering, regression modelling, integer programming, and a deployed interactive app.

**Live app:** [shinyapps.io link]

---

## What it does

1. **Ingests** player performance data (MoneyPuck) and salary data (PuckPedia) via CSV download and manual collection
2. **Cleans and merges** across inconsistent player names, deduplicates, and removes league-minimum contracts
3. **Engineers features** — per-60 production rates, cap efficiency metrics (cost per point, cost per xGoal), net takeaway value, possession impact index, Corsi efficiency
4. **Trains a Ridge regression model** to predict MoneyPuck's `gameScore/60` from engineered stats (Ridge chosen to handle multicollinearity across correlated hockey stats)
5. **Runs a constrained optimizer** using OR-Tools CP-SAT: maximizes total predicted value subject to the $83.5M salary cap, position minimums (12F / 6D), and user-defined include/exclude rules
6. **Surfaces results** in an interactive Shiny for Python app with a roster table and Plotly cap-vs-value scatter chart

---

## Pipeline

```
MoneyPuck CSV + PuckPedia CSV
        │
        ▼
   Name cleaning & merge
        │
        ▼
  Feature engineering
  (per-60 rates, cap efficiency, possession metrics)
        │
        ▼
  Ridge regression → pred_mp_value per player
        │
        ▼
  OR-Tools CP-SAT optimizer
  (maximize Σ pred_mp_value subject to cap + position constraints)
        │
        ▼
  Shiny app (reads predictions CSV from AWS S3 via DuckDB httpfs)
```

---

## Key technical decisions

| Decision | Why |
|---|---|
| Ridge regression | Hockey stats are highly correlated (goals, xGoals, points all move together). Ridge keeps all features while shrinking coefficients to reduce overfitting |
| OR-Tools CP-SAT | Roster construction is an integer programming problem — each player is a binary yes/no. CP-SAT handles this cleanly with hard constraints |
| DuckDB httpfs for S3 reads | Lets the Shiny app query the predictions CSV directly from S3 with a SQL interface, no boto3 setup required in the app |
| Per-60 normalization | Raw counting stats penalize players with less ice time. Per-60 rates make value comparable across different usage contexts |

---

## Scenarios tested

| Scenario | Core | Total Predicted Value |
|---|---|---|
| Leafs — Draisaitl replaces Marner | Matthews, Nylander, Draisaitl | 0.87 |
| Leafs — Original core four | Matthews, Nylander, Marner | 0.79 |
| Leafs — No core four | Matthews, Nylander, Tavares | 0.91 |
| Panthers core | Tkachuk, Barkov, Reinhart, Verhaeghe | 1.00 |

The Panthers scored highest — consistent with them winning back-to-back Cups. Marner's cap hit consistently produced worse roster-wide outcomes than alternatives at similar or lower cost.

---

## Stack

- **Python** — pandas, numpy, scikit-learn, or-tools, shiny, plotly, duckdb, boto3
- **AWS S3** — raw data storage and predictions hosting
- **Deployed** on shinyapps.io

---

## Data sources

- [MoneyPuck](https://moneypuck.com) — player performance and advanced stats (CSV download)
- [PuckPedia](https://puckpedia.com) — salary and contract data

---

## Limitations

- Model assumes past performance predicts future performance — doesn't account for injuries, line changes, or system fit
- Optimizer treats predicted value as perfectly additive; real-world chemistry and special teams roles aren't modelled
- S3-hosted model bundle intended for live API inference isn't yet connected to the app — optimizer currently runs from the saved predictions CSV
