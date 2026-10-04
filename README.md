# Anomaly Transaction detection layer

A ML: XGBoost model treating received transactions from banking for users, analyising  the historical data from that user then see if that transaction is anomalous or not

### project hierarchy 
````
  anomly_bank_detection/
  ├── README.md
  ├── pyproject.toml           # dependencies + project metadata
  ├── .gitignore               # ignores data/, .venv/, models/, .env
  ├── data/
  │   ├── raw/                 # original dataset, never modified (not in git)
  │   └── processed/           # derived data (not in git)
  ├── notebooks/               # exploration only, no logic that matters
  ├── src/
  │   └── fraud_detection/     # the real code, as an installable package
  │       ├── __init__.py
  │       └── config.py        # paths and settings in one place
  └── tests/
````
