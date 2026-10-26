# Credit card experiment: design and data contract

## Goal
Segment our ~50,000 customers into card-readiness tiers, run randomized card-offer experiments inside each tier, monitor drift, and test everything on a seeded 60-day simulation before real data is used. The US TAM (~50M) is only used to scale measured uplift into a market-size projection.

## Segmentation
Four tiers from k-means (or simple rules) on pre-experiment features only.

| Tier | Meaning | Offer policy |
|---|---|---|
| Prime | Proven repayer, healthy utilisation, real spend | Card offer experiments |
| Watch | Some activity, short track record | Soft offer or re-evaluate later |
| Nurture | Signed up, little or no transacting | Activation incentives first |
| Decline | In arrears or over-extended | Excluded from all credit offers |

- Features: arrears, utilisation, spend, activity, RFM; scaled before clustering. Nothing measured after an offer.
- k chosen by silhouette plus business sense; rule-based tiers are the simpler alternative.
- Frozen model: fit once, save scaler and centroids, assign new customers with predict.
- New customers start in Nurture and are scored after ~30 days of activity.
- Refit on a schedule (e.g. quarterly) or on drift flag, only between experiments.
- Hard credit rules are a separate check; eligibility never depends on the cluster alone.

## Experiment design
- Stratified random assignment: test vs control within each segment, using a seeded hash of customer ID.
- True no-offer holdout in every segment.
- Decline tier removed before randomization.
- Balance check on pre-period spend, arrears, utilisation; re-seed if an arm is off.
- No hand-picked controls.
- Metrics: activation primary; guardrails delinquency, utilisation, complaints (3-6 months to show).
- Tenure clock: outcomes measured from each customer's entry/exposure date; compare cohorts at the same tenure, flag immature ones.
- Run a power calculation first to size arms and set the minimum detectable effect.

## Drift monitoring
Compare each period to a frozen reference (the training data). Alerts trigger review, never automatic refit.

| Check | Method | Thresholds |
|---|---|---|
| Feature drift | PSI per feature on saved reference bin edges | <0.10 OK, 0.10-0.25 watch, >0.25 alert |
| Segment mix | Share per segment vs reference, PSI on mix | Flag any segment moving >=5 points |
| Cluster fit | Mean distance to assigned centroid; share far from every centroid | Rising trend triggers review |
| Structure | Periodic refit on recent data; adjusted Rand index vs frozen labels | Low agreement triggers review |
| Outcomes | Arrears and activation rates per segment | Segment no longer behaves as profiled |

- Reference stays fixed to the training set; a rolling reference hides slow drift.
- Do not alert on one noisy week of PSI for small segments.
- Never refit mid-experiment; apply a new model to the next experiment.

## Simulation
One seeded script simulates 60 days; no scheduler needed.
- Identities from Faker; correlated attributes from SDV (Gaussian copula fit on a seed table); behaviour from a latent profile (risk, spend rate, responsiveness).
- Daily loop: app opens, purchases, repayments, missed payments, with weekday and payday seasonality.
- Arrivals: Poisson with trend and campaign spikes; later cohorts shifted (more paid channel, higher risk) so drift is detectable.
- Scenario config declares campaigns (e.g. day 5), payday spike, macro shock on arrears, fee change.
- True effect sizes planted in advance, including null campaigns.
- ~20 seeds; trust only results that hold across them.
- DST ends Sunday 2026-11-01 (25-hour day); daily counts and RFM windows must handle it.
- Seed table: hand-written latent-profile generator; get approval before training on any real bank data.

### Layers
Every causal effect is planted in exactly one place (the scenario overlay).

1. Attributes: SDV + Faker -> customers, latent profiles (`sim_truth_customer`)
2. State dynamics (optional): Markov/agent rules -> active, dormant, delinquent, churned
3. Events: Poisson/Hawkes counts, lognormal amounts, weekday/payday effects -> `fact_transactions`, `fact_events`
4. Scenario overlay: config multipliers on layers 2-3
5. Realism pass (optional)
6. Scale (optional): dbldatagen on Spark

Build order: layers 1, 3, 4 first. Recovery test: test/control analysis recovers each planted effect within its CI across ~20 seeds.

## MLflow
One parent run per simulation, child run per cohort/seed.
- Params: root seed, Faker locale, synthesizer type, customer count, cohort day, scenario config name.
- Metrics: SDV quality/diagnostic scores, row counts, null rates, PSI vs reference.
- Artifacts: metadata, scenario config, fitted synthesizer, sample output, quality report.
- Tags: code version, SDV and Faker versions, `data_label=synthetic`.

## Gold tables
| Table | Grain | Contents |
|---|---|---|
| dim_customer | customer | Surrogate ID, join date, channel, region. No segment, no latent profile |
| sim_truth_customer | customer | Latent profile and true effects; generator/validation only, never used by models |
| fact_transactions, fact_events | event | Purchases, repayments, app activity; partitioned by business date |
| fact_rfm_snapshot | customer x snapshot date | R/F/M over fixed windows, scores from saved bin edges |
| fact_segment_assignment | customer x snapshot date | Segment, model version |
| fact_experiment_assignment | customer x experiment | Arm, assigned date, stratum |

- RFM comes from `fact_transactions` with event_date_et <= snapshot_date, left-joined from the dimension (no transactions: frequency/monetary 0, recency null or days since join).
- Non-purchase events feed separate engagement features.
- Snapshots, not overwrites.

## Data contract
Time
- Store `event_ts_utc`; derive `event_date_et` once (America/New_York, zone name never fixed offset).
- DST days are 23 or 25 hours: use calendar days, never 24h arithmetic. Convert UTC to local, never the reverse.
- Snapshot cutoff 23:59:59 ET; filter event_date_et <= snapshot_date. Snapshots frozen once written; late events land in the next snapshot and their count is logged.
- Campaigns are zoned timestamps. Simulation uses a fixed calendar, never now().

Randomness
- One root seed per run; per-entity generators from hash(root seed, customer ID, day), so adding customers does not reshuffle others.
- Separate streams for arrivals, behaviour, campaign response.
- Log seed, scenario config, code version.

Loading and experiments
- Idempotent loads (merge on keys), `loaded_at` on every table.
- Segment model and RFM bins frozen per experiment.
- Decline tier excluded before randomization.
- Outcomes by tenure since exposure.
- Synthetic data labeled as such; no real customer fields.

## Open items
- [ ] Power calculation and MDE
- [ ] Extend drift to categorical features and distance-to-centroid
- [ ] Simulator: latent profiles, arrivals, scenario config
- [ ] Decide: RFM as k-means features or descriptive only
- [ ] Confirm America/New_York with data owners
- [ ] Load gold tables, run ~20 seeds
