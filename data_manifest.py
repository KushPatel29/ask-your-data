"""
The catalog of business tables the assistant can query.

One entry per table. Each entry says where the data was vendored from (a sibling
portfolio repo), the domain it belongs to, and a plain-English description. The
description is what the language model reads to decide which tables and columns
answer a question, so it is written for that purpose: what the grain is, what the
money columns mean, and which codes are non-obvious.

Every dataset here is synthetic (Faker, fixed seeds) — no PHI, no real customers,
no real employees. See the source repos for how each was generated.

`table` is the base name; the loaded DuckDB table is `<domain>_<table>` so that
the several `dim_customer` / `fact_orders` tables across domains never collide.
"""

DOMAINS = {
    "healthcare": "Hospital revenue cycle — claims from submission through paid / denied / open AR.",
    "hr": "Workforce & people analytics — employees, attrition, hiring funnel, flight risk.",
    "finance": "GL reconciliation — ERP general ledger vs. subledger, and the exceptions between them.",
    "supplychain": "Cold-chain distribution — orders, inventory lots, warehouses, demand forecast.",
    "retail": "Specialty-meats wholesale — customers, sales lines, RFM analytics, cross-sell recs.",
    "migration": "A legacy-to-Fabric migration program — the moved data plus parallel-run validation.",
    "marketing": "Marketing attribution & incrementality — journeys, channel touchpoints, media spend, and a geo holdout graded against a known ground truth.",
    "clinical": "Clinical trial data management — EDC capture, edit-check queries, medical coding, and the injected-defect manifests that prove the pipeline catches errors.",
    "aml": "Anti-money-laundering transaction monitoring — scored payments, the rules and model that flag them, and the cost-optimal alert threshold.",
    "wholesale": "Northgate Retail Group supercenter chain — store-format sales by merchandising department, vendors, rep quotas, P&L, marketing and labor.",
    "dbt": "The modelled warehouse itself — dbt models, data tests, DAG lineage, and the daily KPI mart.",
}

# domain, table, source path (relative to the portfolio-projects root), description
MANIFEST = [
    # ---- healthcare (revenue cycle)
    ("healthcare", "dim_payer", "healthcare-claims-analytics/data/dim_payer.csv",
     "Insurance payers. payer_type groups them (Medicare, Medicaid, Commercial, Medicare Advantage, Self-Pay)."),
    ("healthcare", "dim_provider", "healthcare-claims-analytics/data/dim_provider.csv",
     "Rendering providers with their specialty and facility."),
    ("healthcare", "dim_service_line", "healthcare-claims-analytics/data/dim_service_line.csv",
     "Clinical service lines (Cardiology, Oncology, Emergency Department, ...)."),
    ("healthcare", "fact_claims", "healthcare-claims-analytics/data/fact_claims.csv",
     "One row per claim. status is Paid / Denied / Pending. submitted_amount is billed charges, "
     "allowed_amount is the contracted amount, paid_amount is cash collected (all blank while Pending). "
     "denial_reason is a CARC code. ar_age_days / ar_bucket age the open (Pending) AR. Joins dim_payer, "
     "dim_provider, dim_service_line by their id columns."),
    ("healthcare", "ar_yield_predictions", "healthcare-claims-analytics/output/ar_yield_predictions.csv",
     "Predictive worklist for open AR: expected_nrv is forecast collectable cash, denial_propensity and "
     "expected_yield_rate are probabilities, priority_score ranks which claims to work first (priority_rank = 1 is highest)."),

    # ---- hr (workforce)
    ("hr", "dim_department", "hr-attrition-analytics/data/dim_department.csv", "Departments."),
    ("hr", "dim_job", "hr-attrition-analytics/data/dim_job.csv", "Job roles and their level."),
    ("hr", "dim_location", "hr-attrition-analytics/data/dim_location.csv", "Office locations / regions."),
    ("hr", "comp_benchmark", "hr-attrition-analytics/data/comp_benchmark.csv",
     "Market pay benchmark by job level — the reference for pay-equity and compa-ratio analysis."),
    ("hr", "fact_employees", "hr-attrition-analytics/data/fact_employees.csv",
     "One row per employee. is_active flags current staff; termination_date / term_type describe leavers "
     "(Voluntary vs Involuntary). compa_ratio is pay vs market (1.0 = at market). engagement_score, "
     "performance_rating, overtime_hours, months_since_promotion are attrition drivers. Joins dim_department, "
     "dim_job, dim_location by id."),
    ("hr", "fact_applications", "hr-attrition-analytics/data/fact_applications.csv",
     "Recruiting funnel — one row per application, with boolean stage flags (reached_screen ... hired) and days_to_fill."),
    ("hr", "fact_interventions", "hr-attrition-analytics/data/fact_hr_interventions.csv",
     "Retention interventions applied to employees and whether they stayed — for measuring intervention effect."),
    ("hr", "flight_risk_scores", "hr-attrition-analytics/output/flight_risk_scores.csv",
     "Model output: per-employee attrition risk_score (0-1), risk_band, and top_reason. Active employees only."),

    # ---- finance (GL reconciliation)
    ("finance", "dim_account", "gl-reconciliation-dashboard/data/dim_account.csv",
     "Chart of accounts. account_type and statement (Balance Sheet / Income Statement) classify each account."),
    ("finance", "dim_cost_center", "gl-reconciliation-dashboard/data/dim_cost_center.csv", "Cost centers."),
    ("finance", "erp_gl", "gl-reconciliation-dashboard/data/source_erp_gl.csv",
     "General-ledger transactions from the ERP (the system of record). amount, period, posted_date, account_id."),
    ("finance", "subledger_gl", "gl-reconciliation-dashboard/data/source_subledger_gl.csv",
     "The same transactions as recorded in the subledger. Reconciliation compares this against erp_gl."),
    ("finance", "reconciliation_exceptions", "gl-reconciliation-dashboard/output/gl_reconciliation_exceptions.csv",
     "Where ERP and subledger disagree. exception_type is Missing / Timing / Amount / Duplicate; variance_amount is the gap."),

    # ---- supplychain (cold-chain distribution)
    ("supplychain", "dim_customer", "supply-chain-control-tower/data/bronze/dim_customer.csv", "Customers being shipped to."),
    ("supplychain", "dim_product", "supply-chain-control-tower/data/bronze/dim_product.csv",
     "Products with category, shelf_life_days, unit_cost and unit_price."),
    ("supplychain", "dim_supplier", "supply-chain-control-tower/data/bronze/dim_supplier.csv", "Suppliers."),
    ("supplychain", "dim_warehouse", "supply-chain-control-tower/data/bronze/dim_warehouse.csv", "Distribution warehouses."),
    ("supplychain", "dim_lot", "supply-chain-control-tower/data/bronze/dim_lot.csv",
     "Inventory lots with production and expiry dates — drives FEFO (first-expiry-first-out) and spoilage risk."),
    ("supplychain", "fact_orders", "supply-chain-control-tower/data/bronze/fact_orders.csv",
     "One row per order line. qty_ordered vs qty_shipped measures fill rate; promised_date vs shipped_date "
     "measures on-time delivery (OTIF). Joins dim_customer, dim_product, dim_lot, dim_warehouse by id."),
    ("supplychain", "fact_inventory", "supply-chain-control-tower/data/bronze/fact_inventory_snapshot.csv",
     "Inventory on hand by product / lot / warehouse over time."),
    ("supplychain", "demand_forecast", "supply-chain-control-tower/analytics/output/forecast_next_28d.csv",
     "Model output: forecast_units by product category for the next 28 days."),

    # ---- retail (specialty-meats wholesale)
    ("retail", "dim_product", "Customer-Recommendation-Engine/data/catalog.csv",
     "Product catalog (sku, protein, description, unit_cost, unit_price)."),
    ("retail", "dim_customer", "Customer-Recommendation-Engine/data/customers.csv",
     "Wholesale customers with persona, region and assigned sales rep."),
    ("retail", "fact_sales", "Customer-Recommendation-Engine/data/sales_lines.csv",
     "One row per sales line. revenue and cost are dollars; quantity_lb is pounds sold. Denormalized with "
     "customer_name, region, rep, protein already joined in."),
    ("retail", "customer_analytics", "Customer-Recommendation-Engine/output/customer_analytics.csv",
     "Per-customer analytics: total_revenue, total_margin, rfm_segment, churn_risk, clv_12m_runrate, recency_days, "
     "expected_next_order — the customer-health table."),
    ("retail", "cross_sell_recommendations", "Customer-Recommendation-Engine/output/cross_sell_recommendations.csv",
     "Model output: recommended next SKUs per customer with a score and dollar opportunity."),

    # ---- migration (legacy -> Fabric program)
    ("migration", "dim_customer", "legacy-to-fabric-migration/data/customers.csv", "Customers in the migrated dataset."),
    ("migration", "dim_product", "legacy-to-fabric-migration/data/products.csv", "Products in the migrated dataset."),
    ("migration", "fact_orders", "legacy-to-fabric-migration/data/orders.csv", "Orders in the migrated dataset."),
    ("migration", "migration_plan", "legacy-to-fabric-migration/data/migration/migration_plan.csv",
     "The migration program plan — artifacts to move, their wave, complexity and status."),
    ("migration", "parallel_run_results", "legacy-to-fabric-migration/data/migration/parallel_run_results.csv",
     "Parallel-run validation: for each artifact, whether row counts / control totals / checksums matched between "
     "legacy and Fabric, and the GO/NO-GO verdict."),

    # ---- marketing
    ("marketing", "dim_channel", "marketing-attribution-analytics/data/dim_channel.csv",
     "The seven marketing channels. channel_group buckets them Paid / Owned / "
     "Earned. is_paid = 1 for the four buyable channels (paid_search, paid_social, "
     "display, affiliate); email, organic_search and direct are is_paid = 0 with "
     "cost_per_click 0 — they cannot be bought, which is why crediting them is the "
     "whole problem in this domain. Joins the fact tables on channel_id, or on the "
     "channel name string (the name is denormalised into fact_sessions and "
     "fact_spend)."),
    ("marketing", "dim_user", "marketing-attribution-analytics/data/dim_user.csv",
     "One row per user (14,000 users, 1,521 of them converted). converted = 1 if "
     "the user ever placed an order. first_touch_channel and last_touch_channel "
     "are the first and last channel that user was exposed to, so COUNT(*) WHERE "
     "converted = 1 grouped by last_touch_channel *is* last-touch attribution and "
     "grouped by first_touch_channel *is* first-touch attribution — this is the "
     "table for 'which channel gets the credit' questions. touches is how many "
     "sessions the journey contained (converters average 4.22, non-converters "
     "3.52). region is a Canadian region code (AB, BC, ON, QC, Prairies, "
     "Atlantic). Joins fact_sessions, fact_journeys and fact_orders on user_id."),
    ("marketing", "fact_sessions", "marketing-attribution-analytics/data/fact_sessions.csv",
     "One row per session, and a session is one marketing touchpoint — the atomic "
     "grain of the whole domain (50,282 rows, Aug 2025 through Jul 2026). "
     "touch_position is this session's 1-based place in that user's journey and "
     "touches_in_journey is that journey's total length, so touch_position = 1 "
     "means the session OPENED the journey and touch_position = touches_in_journey "
     "means it CLOSED it; the ratio of closes to total touches is how you separate "
     "assist channels (display closes only 4.7% of its touches) from closer "
     "channels (email 70.5%). is_converting_session = 1 if the order happened in "
     "that session. is_click distinguishes a click from a view-through. device is "
     "desktop / mobile / tablet. Joins dim_user on user_id, dim_channel on "
     "channel_id, fact_orders on session_id."),
    ("marketing", "fact_journeys", "marketing-attribution-analytics/data/fact_journeys.csv",
     "One row per USER, not per touch — the collapsed view of a journey. `path` is "
     "that user's entire ordered channel sequence joined with '>' (e.g. "
     "'direct>display>paid_social>organic_search'), and converted says whether the "
     "journey ended in an order. Average path length is 3.59 channels; no path is "
     "empty. Use this for path-SHAPE questions — journey length, which channel "
     "opens or closes, repeated touches, converting vs non-converting path "
     "patterns — with DuckDB string functions such as str_split(path, '>'), "
     "LEN(str_split(...)) and list indexing. Same 14,000 users as dim_user; "
     "fact_sessions is the exploded per-touch version of these same journeys, so "
     "never join the two and then sum — you will multiply rows."),
    ("marketing", "fact_orders", "marketing-attribution-analytics/data/fact_orders.csv",
     "One row per order (2,169 orders). revenue and gross_margin are dollars; "
     "gross_margin is already net of cost, roughly 42% of revenue. is_first_order "
     "= 1 marks the acquisition order and 0 a repeat purchase. session_id is NULL "
     "on 648 rows — repeat orders that arrived with no marketing session attached "
     "— so joining orders to fact_sessions to attribute revenue silently drops "
     "~30% of it; join dim_user on user_id instead and credit the revenue to "
     "first_touch_channel or last_touch_channel. Joins dim_user on user_id."),
    ("marketing", "fact_spend", "marketing-attribution-analytics/data/fact_spend.csv",
     "Daily media spend, one row per date_key x channel (2,392 rows, $37,202 "
     "total). spend is dollars, alongside impressions and clicks. Only the four "
     "paid channels carry money — paid_search $17,982, paid_social $9,335, "
     "affiliate $7,766, display $2,119 — while email, organic_search and direct "
     "appear with impressions and clicks but spend = 0, so ROAS and CPA are only "
     "meaningful when filtered to dim_channel.is_paid = 1 (otherwise you divide by "
     "zero). Combine with conversion counts from dim_user (converted = 1, grouped "
     "by last_touch_channel) to get cost per acquisition, or with fact_orders "
     "revenue to get ROAS. Note display buys 346,765 impressions for only 3,831 "
     "clicks — cheap, wide, and almost never the closer."),
    ("marketing", "experiment_geo_weekly", "marketing-attribution-analytics/data/experiment_geo_weekly.csv",
     "The geo holdout experiment — the only table here that measures CAUSALITY "
     "rather than correlation. Grain is one row per geo per week: 20 DMA-style "
     "geos x 16 weeks (1.77M sessions, 65,885 conversions). in_holdout = 1 marks "
     "the 10 treated geos; period is 'pre' (weeks 0-7, Aug 1 - Sep 19 2025) or "
     "'treatment' (weeks 8-15, Sep 26 - Nov 14 2025); channel_suppressed = 1 only "
     "on holdout geos during the treatment period, when paid search was switched "
     "off. Read it with difference-in-differences: compare the holdout arm's "
     "conversion-rate change to the control arm's change, because both arms fell "
     "together on a seasonal downswing — a naive pre/post on the holdout alone "
     "reads 11.0% lift, roughly double the planted truth of 5.5%, while DiD lands "
     "near 4.3%. This is a SEPARATE simulated population from fact_sessions and "
     "dim_user; it shares no user_id or session_id and must never be joined to "
     "them."),
    ("marketing", "attribution_comparison", "marketing-attribution-analytics/output/attribution_comparison.csv",
     "The scoreboard: one row per channel, with six attribution models' credit "
     "shares side by side against the known truth. Every model column is that "
     "channel's SHARE of all conversions under that model, and each column sums to "
     "1.0 across the seven rows. first_touch gives 100% of the credit to the "
     "channel that opened the journey; last_touch gives it all to the channel that "
     "closed it; linear splits it evenly across every touch; position_based is the "
     "40/20/40 heuristic (40% opener, 40% closer, 20% spread across the middle); "
     "markov is a removal-effect Markov chain; shapley is an exact Shapley value "
     "over channel coalitions. true_lift is the channel's real planted causal lift "
     "on conversion probability and true_incremental_share is that lift normalised "
     "into a share — these two columns are the ground truth the models are graded "
     "against, and they exist only because the data was generated with the "
     "counterfactual known. The headline result lives here: direct's last_touch "
     "share is 0.245 against a true share of 0.019, a 13.1x overcredit of a "
     "channel nobody can buy, while paid_search is undercredited (0.253 last-touch "
     "vs 0.321 true). Ranking models by AVG(ABS(share - true_incremental_share)) "
     "puts position_based first and last_touch last — the sophisticated models "
     "(markov, shapley) both lose to the 40/20/40 heuristic. UNPIVOT the six model "
     "columns to compare them in one query."),

    # ---- clinical
    ("clinical", "subjects", "clinical-data-management/data/subjects.csv",
     "Enrolled subjects in synthetic study SYN-2026-01 — one row per subject, 120 "
     "subjects across 5 sites. subject_id is formatted <site_id>-<seq>. arm is the "
     "randomised treatment arm (A or B). consent_date is informed consent, "
     "baseline_date is the baseline visit. completed = 1 if the subject finished "
     "the study, 0 if they discontinued — so AVG(completed) is the completion "
     "rate. Joins edc_item_data, query_log and coding_results by subject_id, and "
     "everything by site_id. Fully synthetic, no PHI."),
    ("clinical", "edc_item_data", "clinical-data-management/data/edc_item_data.csv",
     "The raw electronic data capture (EDC) extract — one row per collected CRF "
     "item, i.e. the long/EAV grain of subject x visit x form x record x item. "
     "visit_id is the scheduled visit (SCR screening, BASE baseline, WK4, WK12, "
     "EOS end of study); form_id is the CRF form (DM demographics, IE "
     "inclusion/exclusion, VS vital signs, EX exposure, AE adverse events, CM "
     "concomitant meds, DS disposition, LB labs); item_oid is the CDASH item name "
     "(WEIGHT, SYSBP, AESTDAT, ...). record_num distinguishes repeating records on "
     "the same form (e.g. a subject's 2nd adverse event). value is stored as text "
     "for every item regardless of the item's real type — cast it before doing "
     "arithmetic or date maths. Joins subjects by subject_id and injected_defects "
     "/ query_log by (subject_id, visit_id, form_id, record_num, item_oid). "
     "Synthetic, no PHI."),
    ("clinical", "injected_defects", "clinical-data-management/data/injected_defects.csv",
     "The ground-truth manifest of errors DELIBERATELY planted into edc_item_data "
     "by the data generator, so the validation engine can be scored against a "
     "known answer rather than against an absence of complaints. One row per "
     "planted defect (49 of them). defect_class names the fault (missing_required, "
     "out_of_range, sbp_not_above_dbp, ae_before_consent, end_before_start, "
     "recovered_no_end_date, serious_ae_no_action, discont_no_reason, "
     "visit_out_of_window) and expected_check is the edit-check ID that OUGHT to "
     "fire for it. form_id / record_num / item_oid are NULL for visit-level "
     "defects (EC-VISIT-01), which are keyed only to a subject and visit — so join "
     "with COALESCE on those three columns. Detection rate = fraction of these "
     "rows that match a row in query_log on (subject_id, visit_id, form_id, "
     "record_num, item_oid, expected_check = check_id); the reverse anti-join "
     "gives the false-positive count. Synthetic, no PHI."),
    ("clinical", "query_log", "clinical-data-management/output/query_log.csv",
     "Data queries (discrepancies) the validation engine actually raised against "
     "edc_item_data, with their lifecycle — one row per query. check_id is the "
     "edit check that fired (EC-REQ required-field, EC-RANGE out-of-range, EC- "
     "VS-03 systolic must exceed diastolic, EC-AE-01/04/05/06 adverse-event "
     "consistency, EC-DS-01 discontinuation reason, EC-VISIT-01 visit outside its "
     "protocol window); severity is 'query' (site must respond) or 'warning'. "
     "protocol_ref cites the protocol section that justifies the query. status is "
     "open / closed. days_to_close is populated for closed queries; age_days and "
     "age_band ('0-7 days' ... '60+ days') are populated for open ones — the two "
     "are mutually exclusive, so filter on status before aggregating either. "
     "form_id / record_num / item_oid are NULL for visit-level queries. This is "
     "the DETECTED side of the reconciliation against injected_defects. Joins "
     "subjects by subject_id. Synthetic, no PHI."),
    ("clinical", "query_site_performance", "clinical-data-management/output/query_site_performance.csv",
     "Pre-aggregated site scorecard for query management — one row per site (5 "
     "rows). queries_raised / queries_open / queries_closed are counts, close_rate "
     "is queries_closed / queries_raised (0-1, not a percentage), "
     "median_days_to_close is over closed queries only. A convenience rollup of "
     "query_log; it is NOT normalised by enrolment, so divide by the site's "
     "subject count in subjects for a per-subject query rate. Synthetic, no PHI."),
    ("clinical", "coding_results", "clinical-data-management/output/coding_results.csv",
     "Medical coding outcomes — one row per verbatim term reported on an AE or CM "
     "form. dictionary is MedDRA (adverse events) or WHODrug (concomitant "
     "medications). verbatim is the site's free text; coded_term is the dictionary "
     "term it mapped to; soc is the MedDRA System Organ Class for MedDRA rows and "
     "an ATC code for WHODrug rows. status is the coding outcome: 'auto' (exact "
     "dictionary hit), 'synonym' (matched via a synonym list), 'ambiguous' "
     "(multiple candidates, needs a coder decision) or 'uncoded' (no match — needs "
     "a query back to the site). coded_term and soc are NULL for uncoded rows, so "
     "the uncoded/ambiguous share is the coding backlog metric. Joins subjects and "
     "edc_item_data by (subject_id, visit_id, form_id, record_num). Synthetic, no "
     "PHI."),
    ("clinical", "injected_fhir_defects", "clinical-data-management/data/injected_fhir_defects.csv",
     "The second ground-truth manifest — invalid HL7 FHIR R4 resources "
     "deliberately corrupted at known positions in the bulk-export NDJSON before "
     "ingestion, to prove the quarantine catches them and records the RIGHT "
     "reason. 24 rows across 8 defect_class values, three each: malformed_json, "
     "malformed_id, coding_absent, code_outside_value_set, "
     "missing_required_status, medication_choice_both_branches, "
     "unmodelled_resource_type, organization_without_name. source_ref is the "
     "NDJSON file and line_number the exact line seeded; expected_reason_fragment "
     "is the text the quarantine reason must contain. This is the injection "
     "manifest only — the per-resource quarantine outcome lives in the source "
     "repo's SQL Server tables, not in this CSV, so this table answers 'what was "
     "planted, of what kind, where' rather than 'what was caught'. resource_id "
     "values are Synthea-generated UUIDs. Synthetic, no PHI."),
    ("clinical", "patient_change_manifest", "clinical-data-management/data/patient_change_manifest.csv",
     "The Type 2 slowly-changing-dimension test manifest for the FHIR warehouse — "
     "one row per patient whose demographics were changed in the change-feed run "
     "(1,429 patients), used to assert the warehouse closed the old row and opened "
     "a new version rather than overwriting. new_version_id is the version created "
     "(all 2 — this is the first amendment), change_date is the single feed date. "
     "change_kind is 'address', 'marital' or 'both', and the *_before / *_after "
     "pairs give the values on either side of the change; for change_kind = "
     "'address' the marital before/after are identical and vice versa. Cities are "
     "Synthea-generated Massachusetts place names against UUID patient ids — "
     "synthetic, no PHI, no names or identifiers."),
    ("clinical", "warehouse_summary", "clinical-data-management/output/warehouse_summary.csv",
     "Measured benchmark facts from the FHIR-to-SQL-Server pipeline run, in a "
     "narrow metric/label/value shape — filter on metric, never SUM the whole "
     "value column. metric = 'kpi' gives run headlines (resources ingested, "
     "quarantined, fact rows, patients with history); 'norm' and 'dw' give row "
     "counts per resource type in the normalised (3NF) model and in the star "
     "schema respectively, so joining the two on label shows where the dimensional "
     "load fans rows out; 'timing' gives elapsed SECONDS per pipeline stage "
     "(schema, ingest, shred, load, change feed, SCD2 re-load); 'setting' gives "
     "encounter counts by care setting; 'feasibility' gives the trial-feasibility "
     "funnel (active panel -> aged 18-64 -> qualifying diagnosis -> has a numeric "
     "lab -> meeting all criteria). Synthetic benchmark over Synthea-generated "
     "data, no PHI."),

    # ---- aml
    ("aml", "dim_entity", "aml-transaction-monitoring/data/generated/entities.csv.gz",
     "The monitored population — 1,500 customers of the institution. entity_type "
     "is individual vs business; segment is the behavioural profile (salaried, "
     "self_employed, professional_services, wholesale, gig, ecommerce, retired, "
     "retail_cash_intensive) and is the strongest non-model risk signal — "
     "wholesale and retail_cash_intensive entities alert several times more often "
     "than salaried or gig ones. home_region is genericised to Region A..Region F; "
     "there is no real geography and no account numbers anywhere. "
     "payroll_base_cad, rent_cad and typical_amount_cad are the entity's expected "
     "recurring amounts — the baseline that anomalous activity departs from. Joins "
     "fact_transactions and cases by entity_id."),
    ("aml", "dim_counterparty", "aml-transaction-monitoring/data/generated/counterparties.csv.gz",
     "The 6,000 counterparties entities pay or are paid by. counterparty_type is "
     "employer / merchant / supplier / landlord / individual / utility / "
     "money_service. home_region is genericised (Region A..Region F); pairing it "
     "against the entity's home_region gives the payment corridor. Joins "
     "fact_transactions by counterparty_id — note that cash deposits have no "
     "counterparty, so 5,980 transactions carry a NULL counterparty_id and an "
     "inner join silently drops them."),
    ("aml", "cases", "aml-transaction-monitoring/data/generated/cases.csv.gz",
     "The 60 planted suspicious-activity cases — the ground truth the detection "
     "model is scored against. One row per case. typology is the laundering "
     "pattern (structuring, layering, smurfing, round_dollar, dormant_reactivation "
     "— exactly 12 cases each) and typology_variant refines it (e.g. structuring "
     "'slow' vs 'fast', layering 'dispersed'). n_transactions and total_amount_cad "
     "size the case; span_days is how long it ran; layering cases move by far the "
     "most money. retention, gap_days, baseline_mean_cad, n_senders, median_cad, "
     "variant and used_mule_pool are typology-specific generator parameters and "
     "are NULL for the typologies they do not apply to. Joins dim_entity by "
     "entity_id and fact_transactions by case_id."),
    ("aml", "fact_transactions", "aml-transaction-monitoring/results/scored_transactions.csv.gz",
     "One row per monitored payment, already scored by the detection model — "
     "100,299 payments by 1,500 entities over the twelve months from 2025-08-01 to "
     "2026-07-31 ($390.1M total). Scope is monitored payment activity only; retail "
     "card spend is deliberately excluded. amount_cad is the payment in Canadian "
     "dollars, direction is inbound/outbound from the entity's point of view, "
     "channel is eft / etransfer / wire / cheque / cash_deposit / bill_payment, "
     "and is_cash flags physical cash. LABELS (ground truth, not model output): "
     "is_suspicious = 1 marks the 375 payments belonging to a planted case — 0.37% "
     "prevalence; typology, typology_variant and case_id are populated only on "
     "those rows and NULL everywhere else. SCORES: R1_STRUCTURING..R5_DORMANT are "
     "the five expert rules (booleans), n_rules_fired counts how many fired, and "
     "reason_codes is the analyst-readable narrative for them (NULL when no rule "
     "fired). risk_score is the unified 0-100 score: a 60/40 blend of "
     "rule_component (rule hits, saturating at 0.5) and model_component (the out- "
     "of-fold isolation-forest anomaly percentile). THE SHIPPED OPERATING "
     "THRESHOLD IS risk_score >= 39.6 — that is what 'alerted' means; it yields "
     "1,371 alerted transactions across 366 distinct entities at 25.2% "
     "transaction-level precision and 92% recall. The trailing columns are the "
     "point-in-time features the model saw (txn_count_7d, amount_zscore_entity, "
     "days_since_prev_txn, distinct_counterparties_7d, cash_ratio_7d, hour_of_day, "
     "is_weekend); days_since_prev_txn is -1 on an entity's first payment. Joins "
     "dim_entity by entity_id, dim_counterparty by counterparty_id (NULL on cash "
     "deposits), cases by case_id. VENDORED SUBSET: all 100,299 rows are kept, but "
     "14 of the source file's 43 columns (raw epoch/cents duplicates of amount and "
     "datetime, the CV fold id, the raw isolation-forest score, and the less "
     "interesting rolling-window features) were dropped and six float columns "
     "rounded, to bring the committed CSV to 16.5 MB. The alert set at 39.6 is "
     "unchanged by the rounding."),
    ("aml", "threshold_sweep", "aml-transaction-monitoring/results/threshold_sweep.csv",
     "The cost curve behind the operating threshold: one row per candidate "
     "risk_score cutoff, 0.0 to 100.0 in 0.1 steps. Each row reports the confusion "
     "matrix, precision, recall, F1 and expected cost twice — once counting every "
     "flagged transaction (txn_*) and once counting each flagged entity as a "
     "single investigation (entity_*), which is the honest unit because one entity "
     "with six flagged payments is one case to work. expected_cost_cad = "
     "false_positives x $212.50 (one wasted investigation: 2.5 analyst hours) + "
     "false_negatives x $25,000 (proxy for a missed case); true positives are not "
     "charged. Both constants are illustrative teaching numbers, not industry "
     "benchmarks — read the shape of the curve, not the dollar. "
     "entity_alerts_per_week and entity_within_capacity test the alert volume "
     "against an assumed 48 investigations/week of analyst capacity. "
     "entity_expected_cost_cad bottoms out at threshold 39.6 ($65,025, 366 entity "
     "alerts, 100% case recall) — that is the operating point the project ships."),

    # ---- wholesale
    ("wholesale", "fact_department_month", "wholesale-analytics-platform/cache/ask_your_data/fact_department_month.csv",
     "The Northgate Retail Group sales fact (a US supercenter chain — this is NOT "
     "the specialty-meats book in the `retail` domain). Pre-aggregated from "
     "538,859 order lines: one row per month x merchandising department x region, "
     "34 months (Oct 2023 - Jul 2026, the last month covering only Jul 1-4), 10 "
     "departments, 7 regions. Because it is already summed, SUM these columns and "
     "never COUNT rows. department is the merchandising department (Grocery, Fresh "
     "& Produce, Meat & Seafood, Dairy & Frozen, Health & Wellness, Household "
     "Essentials, Home & Kitchen, Apparel, Toys & Seasonal, Electronics) — the "
     "source warehouse still calls this column ProteinType. revenue is net "
     "invoiced sales = gross_sales - discount_amount, billed on pounds for scale / "
     "catch-weight SKUs and on cases for everything else; cogs is landed cost on "
     "the same basis, and gross_margin = revenue - cogs (margin % = "
     "SUM(gross_margin)/SUM(revenue)). cases_ordered / cases_shipped are selling "
     "units, pounds_shipped is catch-weight pounds. order_lines = on_time_lines + "
     "late_lines; short_ship_lines and stockout_lines are availability failures. "
     "fiscal_year runs Oct-Sep (FY2025 = Oct 2024 - Sep 2025) and fiscal_period 1 "
     "is October. There is deliberately no order count here — one order spans "
     "several departments, so counting orders at this grain would double-count; "
     "use wholesale_fact_segment_month for orders."),
    ("wholesale", "fact_segment_month", "wholesale-analytics-platform/cache/ask_your_data/fact_segment_month.csv",
     "The same Northgate sales fact aggregated one row per month x store format "
     "(34 months x 7 formats). store_format is the chain's banner: Supercenter, "
     "Neighborhood Market, Discount Store, Club Warehouse, Express, Online "
     "Fulfillment Center, Pickup & Delivery Hub. is_physical_store is false for "
     "the two e-commerce nodes (Online Fulfillment Center, Pickup & Delivery Hub) "
     "and true for the bricks-and-mortar banners. Every order belongs to exactly "
     "one store and one month, so `orders` is safe to SUM across formats and "
     "months here (it is not safe in wholesale_fact_department_month). revenue / "
     "cogs / gross_margin carry the same meaning as in "
     "wholesale_fact_department_month; margin differs sharply by format because "
     "Club Warehouse prices ~12% under list. backorder_cases is unfilled demand, "
     "active_stores is how many stores in that format ordered that month. Use this "
     "table for order counts, order size (revenue / orders) and fill rate "
     "(cases_shipped / cases_ordered)."),
    ("wholesale", "fact_rep_month", "wholesale-analytics-platform/cache/ask_your_data/fact_rep_month.csv",
     "Northgate account-manager performance: one row per month x sales rep (34 "
     "months x 8 reps). monthly_quota is that rep's revenue target for the month — "
     "it is a plan figure, so SUM it across months to get a period target rather "
     "than averaging attainment. quota_attainment_pct is the pre-computed 100 * "
     "revenue / monthly_quota for that single month; for a multi-month attainment "
     "figure recompute it as 100 * SUM(revenue) / SUM(monthly_quota). "
     "active_stores is how many of the rep's stores ordered that month; each store "
     "is assigned to exactly one rep (see wholesale_dim_store.sales_rep_id). Rep "
     "names are synthetic."),
    ("wholesale", "dim_store", "wholesale-analytics-platform/cache/ask_your_data/dim_store.csv",
     "The 620 Northgate locations that place replenishment orders — the chain's "
     "own stores and fulfillment nodes, not third-party customers. One row per "
     "store, with its format, city, region and assigned sales rep, plus a lifetime "
     "rollup of the whole Oct 2023 - Jul 2026 window: orders, order_lines, "
     "revenue, cogs, gross_margin, on_time_line_pct (share of that store's lines "
     "delivered on the expected date) and fill_rate_pct (cases_shipped / "
     "cases_ordered). last_order_date supports recency / dormancy questions — the "
     "data cuts off 2026-07-04, so measure recency from that date rather than from "
     "today. In the source warehouse this dimension is still called CustomerId / "
     "CustomerName / CustomerSegment. The source column `Province` was dropped on "
     "export because it holds one representative state per region rather than each "
     "city's actual state; use region for geography."),
    ("wholesale", "dim_product", "wholesale-analytics-platform/cache/ask_your_data/dim_product.csv",
     "The 880-SKU Northgate assortment, one row per product, joined to "
     "wholesale_dim_supplier by supplier_id and to the department in "
     "wholesale_fact_department_month by department. billing_basis is 'weight' "
     "(unit_of_billing_id = 3, a scale / catch-weight item billed per pound — "
     "mostly Fresh & Produce and Meat & Seafood) or 'each' (billed per case at a "
     "fixed price); avg_list_price, avg_net_price and avg_unit_cost are therefore "
     "per pound for weight SKUs and per case for each SKUs, and they are realized "
     "averages across the period, not a price list. sell_through_pct is the share "
     "of units bought that sell at full value — the source warehouse still calls "
     "this column YieldPct; low values (Toys & Seasonal ~81%, Apparel ~82%) mean "
     "markdown and shrink losses. revenue / cogs / gross_margin are lifetime "
     "totals for the SKU. avg_days_of_supply, avg_on_hand_cases, "
     "avg_reorder_point_cases and stockout_lines describe the inventory position "
     "lines were picked against."),
    ("wholesale", "dim_supplier", "wholesale-analytics-platform/cache/ask_your_data/dim_supplier.csv",
     "The 46 vendors behind the Northgate assortment, one row per supplier, with a "
     "lifetime rollup over Oct 2023 - Jul 2026. primary_department is the "
     "department the vendor ships most lines into and departments_supplied / "
     "skus_supplied show how broad the vendor is. This is the vendor scorecard "
     "table: gross_margin_pct is margin earned on that vendor's goods, "
     "on_time_line_pct is the share of its lines delivered on the expected date, "
     "fill_rate_pct is cases_shipped / cases_ordered, stockout_lines counts lines "
     "picked against an empty position, and avg_transit_days is lead time. Joins "
     "wholesale_dim_product by supplier_id."),
    ("wholesale", "finance_monthly", "wholesale-analytics-platform/cache/ask_your_data/finance_monthly.csv",
     "Northgate's monthly management accounts — one row per month, 33 months Oct "
     "2023 - Jun 2026. Note this stops one month earlier than the sales facts: the "
     "partial July 2026 is excluded here, so FY2026 revenue is lower in this table "
     "than in wholesale_fact_department_month. revenue / cogs / gross_profit tie "
     "to the sales fact for the months both cover. The P&L flows revenue -> "
     "gross_profit -> operating_expenses (broken out as opex_selling / warehouse / "
     "occupancy / admin / technology / marketing) -> pre_tax_income -> net_income. "
     "Balance-sheet columns (cash through total_liabilities) are period-end "
     "balances, so they must be averaged or point-picked, never summed. ar_current "
     "/ ar_d1_30 / ar_d31_60 / ar_d61_90 / ar_d90_plus are the receivables ageing "
     "buckets and add to accounts_receivable; ap_overdue is the past-due slice of "
     "accounts_payable. capex is cash spent on fixed assets in the month."),
    ("wholesale", "marketing_campaigns", "wholesale-analytics-platform/cache/ask_your_data/marketing_campaigns.csv",
     "Northgate's B2B / trade marketing spend, one row per month x campaign (21 "
     "months x 6 channels: Field sales prospecting, Trade shows & industry events, "
     "Referral & partner program, Email & CRM nurture, Search & trade directories, "
     "Trade press & print). spend, leads, qualified_leads and new_customers are "
     "the campaign's own figures and are safe to SUM; cost per acquisition is "
     "SUM(spend) / SUM(new_customers) and the lead-qualification rate is "
     "qualified_leads / leads. WARNING: selling_spend_month, "
     "acquisition_selling_spend_month and revenue_month are company-level monthly "
     "totals repeated on every campaign row of that month — SUMming them "
     "multiplies by the number of campaigns; take MAX or a DISTINCT month before "
     "using them."),
    ("wholesale", "labor_department_month", "wholesale-analytics-platform/cache/ask_your_data/labor_department_month.csv",
     "Northgate timeclock data rolled up to one row per month x operating "
     "function, Dec 2025 - Jul 2026 only (8 months x 8 functions, 40 employees) — "
     "a much shorter window than the sales facts, so never compare it period-for- "
     "period against them without filtering. function_name is an operating "
     "function (Distribution Center, Store Operations, Fresh Operations, "
     "Transportation, Merchandising, Customer Service, Technology & Data, Finance "
     "& Administration) and is NOT a merchandising department — it does not join "
     "to the `department` column in wholesale_fact_department_month. paid_hours = "
     "regular_hours + overtime_hours + absence_hours; scheduled_hours is what the "
     "roster planned, so paid_hours - scheduled_hours is schedule overrun. "
     "labor_cost is fully-loaded pay at the effective (premium-inclusive) rate and "
     "overtime_cost is the overtime slice of it. overtime_hours_pct and "
     "cost_per_paid_hour are pre-computed for the row — recompute them from the "
     "sums when aggregating across rows. separations counts employees whose exit "
     "fell in that month. Employee-level detail (names, individual punches) is "
     "deliberately not vendored."),

    # ---- dbt
    ("dbt", "models", "supply-chain-analytics-dbt/exports/models.csv",
     "One row per node in the dbt project: 15 models plus 7 seeds, a snapshot, two "
     "analyses, one MetricFlow semantic model, its 5 metrics and the Power BI "
     "exposure. layer is the folder the node lives in (staging / marts / semantic "
     "/ seed / snapshot / analysis / exposure). materialization is view / table / "
     "incremental for models, but doubles as the metric type (simple / ratio) on "
     "metric rows and the exposure type (dashboard) on the exposure row. "
     "parent_count and child_count are DAG edge counts (child_count includes data "
     "tests); test_count is data tests attached to that node. row_count / "
     "column_count are the built relation's real shape and are blank for nodes "
     "that materialize nothing (metrics, exposure, analyses). build_status and "
     "execution_seconds come from the last dbt build (all succeeded). description "
     "is blank where the node is genuinely undocumented — that is real "
     "documentation coverage, not missing data. Joins dbt_tests on model_name = "
     "model_tested and dbt_lineage on model_name = parent or child."),
    ("dbt", "tests", "supply-chain-analytics-dbt/exports/tests.csv",
     "One row per dbt data test — the project's data-quality contract. test_type "
     "is the generic test (not_null, unique, relationships = referential "
     "integrity, accepted_values, expression_is_true) or 'singular' for a hand- "
     "written SQL assertion in tests/. test_style separates generic from singular; "
     "package says whether the test ships with dbt, with dbt_utils, or is the "
     "project's own. model_tested / column_tested name what the test guards "
     "(column_tested is blank for table-level and singular tests). "
     "models_referenced is 2 for relationships tests because they span both sides "
     "of the key. failures is the number of rows that broke the assertion, so 0 "
     "with status 'pass' is a clean test. Joins dbt_models on model_tested = "
     "model_name."),
    ("dbt", "lineage", "supply-chain-analytics-dbt/exports/lineage.csv",
     "The dbt DAG as edges — one row per parent -> child dependency. parent and "
     "child are node names; parent_type / child_type are seed, model, test, "
     "snapshot, analysis, metric, semantic_model or exposure, and parent_layer / "
     "child_layer are the folder (seed / staging / marts / semantic / analysis / "
     "snapshot / test / exposure). 37 of the 69 edges point at data tests, so "
     "filter child_type <> 'test' for the build DAG alone. The chain runs raw_* "
     "seed -> stg_* -> fct_/dim_ marts -> kpi_daily, and out to the MetricFlow "
     "metrics and the supply_chain_control_tower_report Power BI exposure. Use it "
     "for impact analysis ('what breaks if X changes') and upstream questions. "
     "Joins dbt_models on parent or child = model_name."),
    ("dbt", "kpi_daily", "supply-chain-analytics-dbt/exports/kpi_daily.csv",
     "The kpi_daily mart — one row per order date, 2025-01-01 to 2025-06-30. "
     "orders is order lines that day. revenue and gross_margin are dollars, summed "
     "from the modelled fct_orders measures (qty_shipped * unit_price and "
     "qty_shipped * (unit_price - unit_cost)). otif_rate is the share of that "
     "day's orders shipped on or before the promised date AND filled to at least "
     "95%; avg_fill_rate is the mean qty_shipped/qty_ordered. This is a modelled "
     "daily rollup and the one grain the raw supplychain_* tables do not carry — "
     "use it for KPI trend, month-over-month and variance questions instead of re- "
     "aggregating supplychain_fact_orders."),
]


def table_name(domain, table):
    return f"{domain}_{table}"
