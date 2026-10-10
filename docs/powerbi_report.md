# Power BI Report — Fraud Drivers Analysis

Exploratory report showing how variables relate to the fraud label (`is_fraud`).
Data source: `data/processed/users.parquet` and `data/processed/transactions.parquet`
(built by `scripts/build_tables.py`). The `.pbix` file is not committed (it embeds the data).

## Design rules

- Chart the **fraud rate**, not fraud counts (fraud is only 0.58 % of transactions).
- Comparing distributions between labels → use **% within label**, never raw counts.
- One chart = one question; sort comparison charts; the title states the finding.

## Progress

### Setup
- [x] Load `transactions.parquet` and `users.parquet`
- [x] Relationship `users[cc_num]` 1 → * `transactions[cc_num]`
- [x] Power Query columns: `Hour`, `Label`
- [ ] Power Query columns: `Weekday`, `Month`, `Channel`, `Amount Bucket` (+ sort-order column)
- [x] Measures: `Transactions`, `Fraud Transactions`, `Fraud Rate`
- [ ] Measures: `Fraud Amount`, `Median Amount`, `Share Within Label`

### Page 1 — Overview
- [x] Card: Total transactions (1,296,675)
- [x] Card: Fraud transactions (7,506)
- [x] Column chart: fraud by hour of day
- [ ] Card: Fraud rate (0.58 %)
- [ ] Check the hour chart uses `Fraud Rate` on the Y-axis and has a finding as its title
      ("Fraud is ~17× more likely between 22:00 and 03:59")
- [ ] Line + column combo: fraud transactions and fraud rate by `Month`

### Page 2 — Amount
- [ ] Clustered column: `Amount Bucket` × `Share Within Label`, legend `Label`
- [ ] Column: fraud rate by `Amount Bucket`
- [ ] Median amount by label (fraud $396.51 vs  legitimate  $47.28)

### Page 3 — Time
- [ ] Heatmap matrix: `Weekday` × `Hour`, values `Fraud Rate`, colour scale

### Page 4 — Category & channel
- [ ] Bar (sorted): fraud rate by `category`
- [ ] Fraud rate per `Channel` (online vs in store)
- [ ] Heatmap matrix: `category` × `Amount Bucket`, values `Fraud Rate`

### Page 5 — Customer profile (descriptive only, not model features)
- [ ] Filled map: fraud rate by `state` (transactions in tooltip)
- [ ] Fraud rate by `city_pop` bucket
- [ ] Fraud rate by age group and `gender` + fairness note text box

### Page 6 — "What doesn't matter"
- [ ] Home → merchant distance distribution by label (no signal: median 0.797 vs 0.798)

### Later (after feature engineering step 3.2)
- [ ] Load `features.parquet` and add a "Card behaviour" page
      (hours since previous transaction: 1.4 h before fraud vs 4.6 h before legitimate)

### Finish
- [ ] Sync slicers (`Label`, `category`, `Channel`) across pages
- [ ] Export screenshots to `docs/` and link them from the README
