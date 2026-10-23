# Strategy Experiments Log

One experiment at a time; every result recorded here before moving on.
Baseline model: `catboost_model_v2.cbm` (V2 epoch-aligned, AUC 0.5750, DOWN-only).
Evaluation set: chronological test holdout = last 20% of `dataset_labeled_v2.csv`
(1,683 epochs, UP rate 44.0%), exactly matching `train_model_v2.py`'s split.

## Summary

| # | Question | Outcome |
|---|---|---|
| 1 | Tighter conviction threshold on the V2 model | Adopted (gate 0.450 → 0.445) |
| 2 | Regime filters (volatility, spread, streaks) | Rejected — effect vanished under restatement |
| 3 | Isotonic vs Platt calibration | Adopted isotonic |
| 4 | First-90s features + 2.6× data | Adopted (test AUC 0.52 → 0.75); found the old trainer selected configs on the test set |
| 5 | Strategy at real Polymarket prices (June) | +28.4% per $ staked, before spread |
| 6 | Blend model with the market price | Rejected — better calibrated, traded worse |
| 7 | 33-hour forward dry run | −6.2% per $ (optimistic bound) — live canary on hold |
| 8 | Rebuild training data from Binance archives | 364,676 epochs over 3.5 years; AUC 0.75 on the live feed |
| 9 | 3.9-day forward test of the multi-year model | −1.9% per $; win rate ≈ entry price in every bucket |
| 10 | Maker (resting-order) variant | +23.6% per $, but the 95% CI touches zero |
| 11 | Polymarket's switch to TWAP settlement | Rule derived from data (94.4% agreement); AUC 0.77 |
| 12 | Maker forward test under TWAP rules | −5.2% per $ — stop the directional strategy |
| 13 | Order flow from the public data API | Feed incomplete; flow adds nothing beyond price |
| 14 | Wallet skill in 60 GB of full on-chain fills | Skill is real but not followable; caught a look-ahead leak |
| 15 | Late-epoch favorite–longshot bias | Real, but about one tick wide |
| 16 | Cross-market efficiency survey | No consistent bias in sibling markets |
| 17 | Weather markets | The market beat our forecast |

**Bottom line:** no tradeable edge found net of costs. Total trading losses: $0.

---

## Exp 1 — Threshold sweep on deployed V2 model (2026-07-06)

**Question:** does tightening the DOWN-entry threshold (currently `prob_up <= 0.450`
via `min_model_confidence=0.05`) improve trade quality?

**Method:** loaded the production model + calibrator, scored the untouched test
holdout, swept thresholds. Win rate = DOWN rate among trades; Wilson 95% CIs.

| threshold | trades | % of epochs | win rate | 95% CI | breakeven price |
|-----------|--------|-------------|----------|--------|-----------------|
| 0.450 (current) | 928 | 55.1% | 60.5% | [57.3, 63.6] | 0.60 |
| 0.445 | 325 | 19.3% | 66.2% | [60.9, 71.1] | 0.66 |
| 0.440 | 137 | 8.1% | 75.2% | [67.3, 81.7] | 0.75 |
| 0.435 | 80 | 4.8% | 80.0% | [70.0, 87.3] | 0.80 |

Decile analysis: the edge is concentrated — the lowest prob decile (prob ≤ 0.442)
resolves DOWN 73.2%; deciles 1–8 hover at 51–61% (mostly noise); the *top* decile
(prob ≥ 0.456) resolves UP 56.8% vs 44.0% base — a possible latent UP signal
worth revisiting in Exp 3.

**Side finding (important):** `trades.csv` history is corrupted by the settlement
attribution bugs fixed on 2026-07-04 (buy/settle token and market-ID mismatches;
engine balance says -$370 while row-level recomputation says +$12,959 —
irreconcilable). All pre-fix paper statistics, including the apparent "50% live
win rate," are unusable. Post-fix paper data is needed to measure the real-world
gap between test-set win rate and EV-gated live win rate (adverse selection).

**Decision: ADOPT.** Tighten the V2 conviction gate from 0.450 to 0.445
(`min_model_confidence` 0.05 → 0.055, now configurable via `MIN_MODEL_CONFIDENCE`).
Rationale: per-trade win rate 60.5% → 66.2% with CI separation from base rate,
breakeven price rises 0.60 → 0.66 (much more robust to paying up), and ~19% of
epochs still trade (≈55 trades/day of continuous markets). Fewer, stronger trades
also reduce exposure to adverse selection until clean paper data exists.

Implemented: `Config.MIN_MODEL_CONFIDENCE` (default 0.055) wired into
`EVCalculator`; verified prob 0.447 → HOLD, prob 0.442 → BUY_DOWN.

---

## Exp 2 — Regime filters at the adopted gate (2026-07-06)

**Question:** at `prob_up <= 0.445`, does conditioning on decision-time regime
features (volatility, spread, streaks, mean-reversion) improve win rate?

**Method:** screened 11 candidate filters on the train+cal period (1,408 gated
trades), then confirmed on the test period (325 gated trades). Adoption required
consistent direction in both periods AND survival of a clean semantic restatement.

Screen survivors and their fate:

| rule | screen Δwr | test Δwr | verdict |
|------|-----------|----------|---------|
| `spread_widening > 0` | +1.3pp | **−6.0pp** | sign flip — reject |
| `return_5ep > median` | +3.1pp | +5.1pp | looked good, but… |
| → semantic restatement `return_5ep > 0` | +1.5pp | **+0.0pp** | effect vanished — reject |
| `up_ratio_5ep ≥ 0.6` | +1.5pp | +5.0pp | n=90, CI ±9.5pp; same cut as the screen split, not independent — insufficient |

**Decision: REJECT (all filters).** The apparent mean-reversion edge did not
survive semantic restatement — classic multiple-comparisons artifact on a small
test set. No regime filter adopted; the strategy layer stays simple. Revisit only
with a larger dataset (>20K epochs).

---

## Exp 3 — Isotonic vs Platt calibration (2026-07-06)

**Question:** Platt scaling compressed all probabilities into [0.41, 0.47] while
actual outcome rates spanned 0.28–0.57 by decile (Exp 1). Does isotonic
(monotone, non-linear) calibration produce honest probabilities — and does it
unlock the top-decile UP signal?

**Method:** same base model (identical data/seed, AUC 0.5750), calibrator swapped;
compared on the untouched test holdout.

| calibrator | log-loss | Brier | prob range |
|-----------|----------|-------|------------|
| raw | 0.6909 | 0.2489 | [0.46, 0.53] |
| Platt (old) | 0.6843 | 0.2456 | [0.41, 0.47] |
| **isotonic** | **0.6771** | **0.2423** | [0.10, 0.53] |

Reliability by decile now tracks actuals (e.g. d0 predicted 0.32 vs actual 0.28).
Edge table under honest probabilities: `prob_down >= 0.58` → 137 trades at 75.2%
win rate; `>= 0.65` → 100 trades at 77.0%.

UP side: max calibrated prob_up is 0.533 with actual UP rate 55.8% at
`prob_up >= 0.52` (n=174, CI ±7.4pp) — real but too thin to trade; stays blocked
by both the confidence gate and the V2 BUY_UP suppression.

**Decision: ADOPT.**
- New `IsotonicLogOddsCalibrator` in [src/ml/calibration.py](src/ml/calibration.py)
  (drop-in `predict_proba` compatible; output clipped to [0.10, 0.90] to protect
  Kelly sizing from sparse isotonic tail steps).
- `train_model_v2.py` now fits isotonic; production artifact regenerated.
- `MIN_MODEL_CONFIDENCE` default re-anchored 0.055 → **0.08** for the new scale:
  V2 trades at `prob_up <= 0.42`, the high-conviction plateau (75.2% test WR,
  ≈23 trades/day). Verified end-to-end through PredictorV2 → EVCalculator.

---

## Exp 4 — Current-epoch (first-90s) hybrid features + 2.6× data (2026-07-06)

**Question:** V2 predicts at epoch start from prior epochs only, ignoring
everything happening in the epoch it bets on. Do first-90-seconds features of the
current epoch help? Also: the raw dataset grew to 22.2M rows (~21.6K epochs vs
8.4K) — does the old edge replicate on more data?

**Method:** new streaming labeler ([src/ml/label_dataset_v3.py](src/ml/label_dataset_v3.py),
constant-memory — the v2 labeler cannot hold 22M rows; also requires 5 *contiguous*
prior epochs, where v2 silently allowed gaps). A/B on 21,621 epochs, time-ordered
70/10/20, config selected by **cal AUC** (the old trainer selected by test AUC —
leakage), isotonic calibration, test scored once per arm.

| arm | test AUC | test log-loss | gate wr @ prob≤0.42 |
|-----|----------|---------------|---------------------|
| A: baseline 40 features | **0.5242** | 0.6920 | 55.8% (n=1648) |
| B: hybrid +8 first-90s features | **0.7543** | 0.5896 | 74.8% (n=1666) |

**Finding 1 (sobering):** the baseline collapsed from its reported 0.5750 to
0.5242 on 2.6× data with leakage-free selection. The old AUC was inflated by
test-set config selection and a small test window. Prior-epochs-only ≈ no edge.

**Finding 2:** the hybrid's gain is real but dominated by `curr_return_90s`
(importance ~74/100): intra-epoch momentum. **Honesty caveat:** at t+90s the
market has also seen the first 90s and repriced — the 75-80% win rate is against
*epoch-start* odds, not against t+90s prices. The tradeable edge is model-vs-market
disagreement at t+90s, which is exactly what the EV gate computes against live
prices. What we bank on is the much better-calibrated probability (log-loss
0.69 → 0.59) feeding that gate. **Paper trading must validate this before LIVE.**

**Decision: ADOPT.**
- Production model retrained on all 21.6K epochs, 46 features (dropped
  `prev_n_samples` / `curr_n_samples_90s`: train/serve cadence skew — training
  data is 100ms ticks, live prediction is ~2s ticks). Test AUC 0.7547,
  log-loss 0.5872, calibrated range [0.10, 0.90].
- [PredictorV2](src/ml/predictor_v2.py) now waits until **t+90s** into each epoch
  (returns 0.5 before that), then predicts once and caches. Duplicated feature
  construction replaced by shared `build_feature_row` / `build_curr_features`.
  Context buffer resets on epoch gaps (matches contiguous training data).
- Trainer selects configs by cal AUC (leakage fixed) and defaults to the v3 dataset.
- **BUY_UP suppression removed** from EVCalculator: the hybrid model signals both
  sides honestly (prob range reaches 0.90); the conviction gate + EV gate +
  market-disagreement filter are the guards now.

**Expectations reset:** trades now happen between t+90s and epoch end, both sides,
gated at |prob−0.5| ≥ 0.08. Live win rates will be far below the 75-80% table
numbers because entry prices at t+90s already reflect the partial move. Judge the
strategy on paper PnL against real prices, not on the offline win-rate table.

---
## Exp 5 — Historical price-join: strategy vs REAL Polymarket prices (2026-07-11)

**Question:** the paper engine's +580%/2d used inferred fills (fantasy risk).
What would the EV-gated strategy have earned buying at *actual executed
Polymarket prices* at t+90s?

**Method:** 536 out-of-sample test epochs (every 8th of the chronological test
split, June 2026), joined: model prediction (production artifacts) × real
UP-token trade print closest to t+90 (data-api, paginated) × actual outcome.
Simulated the exact production gates (conviction 0.08, min EV 3%), $10 flat stakes.

**Result: +$812 on $2,860 staked (+28.4%/$), 286 trades, WR 50.0% [44.2, 55.8]
at avg cost ~0.40.** Every entry-cost bucket non-negative; the cheap bucket
(cost < 0.35) delivered WR 28.3% vs 19% breakeven → +$497.

**Why it works (key insight):** the model does NOT beat the market as a
predictor (log-loss: market 0.5998, model 0.6012 — near-identical). But in
strong-disagreement epochs the truth lands *between* them:

| model−market | n | actual UP | model said | market said |
|---|---|---|---|---|
| < −0.15 | 139 | 0.547 | 0.382 | 0.667 |
| > +0.15 | 79 | 0.519 | 0.583 | 0.319 |

Both sides are wrong at the extremes; buying the cheap side of the disagreement
is +EV even though neither forecast is right. The edge is *disagreement
harvesting*, not prediction superiority.

**Caveats:** entry at the trade print (no spread/slippage penalty — a real FOK
crosses the spread, costing ~1–3¢/token ≈ 2–7pp of the 28.4%); flat stakes;
single June regime. Still: this replaces the fantasy +580% with a defensible
+20–25% per staked dollar estimate and greenlights the live canary economics.

**Follow-up candidate (Exp 6):** feed the t+90 market price to the model (or a
simple blend) — truth-between-forecasts suggests a meta-estimator beats both.

---
## Exp 6 — Market-price blend / stack (2026-07-12)

**Question:** Exp 5 showed truth lands between model and market at extremes.
Does blending the t+90 market price into the probability beat the model alone?

**Method:** extended price join — 534 cal epochs (fitting) + 1,070 test epochs
(scored once per candidate). Linear blend weight and logit-stack coefficients
fit on cal only. Strategy sim = production gates at real t+90 prices.

| candidate | test log-loss | trades | WR | PnL | ROI/$ |
|-----------|---------------|--------|-----|-----|-------|
| **model only (production)** | 0.5916 | 550 | 55.6% | **+$1,825** | **+33.2%** |
| market only | 0.6072 | 0 | — | $0 | — |
| blend w=0.45 | 0.5783 | 491 | 43.8% | +$1,143 | +23.3% |
| logit stack | **0.5709** | 583 | 58.8% | +$1,629 | +27.9% |

**Findings:**
1. The stack IS the best probability estimator (log-loss 0.5709 — both inputs
   carry weight: 0.58·model + 0.66·market in logit space). Truth-between confirmed.
2. **But better calibration traded worse.** The EV gate monetizes *disagreement
   with the market*; folding the market into p* dilutes exactly the signal being
   harvested. Model-only maximizes PnL.
3. Bonus: Exp 5's result replicated on 2× the test data — model-only
   +$1,825 / 550 trades / WR 55.6% [51.4, 59.7] / +33.2% per staked dollar
   at real prices (pre-spread).

**Decision: REJECT for the trading path** — production keeps the pure model
probability feeding the EV gate. The stack is noted as a superior probability
estimator if a future use needs calibrated risk (e.g. position sizing), but it
must not replace the disagreement signal. Test set now touched by 4 candidates;
no further rule iteration on this split.

---
## Exp 7 — Dry-run forward test, Jul 16–17 (2026-07-18)

**Question:** does the Exp 5 edge (June data, +28.4%/$) hold in a true forward
test — live signals, live books, zero look-ahead?

**Method:** 33h Nautilus dry run logged 132 would-submit orders (side, ask
price, stake) with no outcome knowledge. Outcomes resolved post-hoc from
Polymarket; PnL assumes 100% fill at the logged ask (optimistic bound).
Tool: [nautilus_spike/dryrun_report.py](nautilus_spike/dryrun_report.py).

**Result: WR 32.6%, −$82 on $1,320 staked (−6.2%/$).** The optimistic bound
is *negative* — real fills could only be worse.

| slice | n | WR | breakeven | PnL |
|---|---|---|---|---|
| UP side | 54 | **20.4%** | 32% | −$285 |
| DOWN side | 78 | 41.0% | 37% | **+$203** |
| cheap entries (<0.35) | 73 | **9.6%** | 18% | −$153 |
| mid (0.35–0.50) | 16 | 68.8% | 42% | +$102 |
| upper (0.50–1.00) | 43 | 58% | ~62% | −$31 |

**vs Exp 5 (June):** the cheap-entry bucket flipped from +28.3% WR (be 19%) to
9.6% (be 18%) — CIs barely overlap, this is a real degradation, not noise. In
deep-disagreement epochs this window, **the market was right and the model
wrong**, the opposite of June's truth-in-the-middle. DOWN-side and mid-price
trades remained profitable.

**Candidate explanations (unresolved):** (a) model staleness — trained through
Jun 26, running Jul 16 with regime-sensitive momentum features; (b) Exp 5's
window was one 15-day regime and the extreme-disagreement edge is
regime-dependent; (c) small-sample bad luck (n=132) — possible but the cheap
bucket alone argues against it.

**Decision: LIVE CANARY ON HOLD.** Do not spend real money on a signal whose
optimistic forward bound is negative. Path: (1) pull fresh VM data through
today, (2) retrain (staleness is the cheapest hypothesis to kill), (3) longer
dry run (3–5 days, n≥400), (4) gate: reconstructed forward PnL must be
positive with CI clear of zero before any live order. Do NOT hand-fit price-bucket
filters to this 132-order sample (Exp 2 lesson).

---
## Exp 8 Step 1 — Archive reconstruction validation (2026-07-18)

**Question:** can training data be rebuilt from Binance's free public aggTrades
archive (binance.com) and transfer to our live feed (binance.us)?

**Method:** both sources reduced to a per-second last-price grid feeding one
shared price-pure feature builder ([exp8/price_features.py](exp8/price_features.py),
20 features, no volumes/books — immune to the cross-exchange volume mismatch).
Apr–Jun overlap: 24,959 archive epochs vs 17,437 collected (archive has no
collection gaps).

**Results — all three gates passed:**
1. **Transfer (kill test):** train archive Apr–May → test real collected
   Jun 11–26: AUC **0.7467** vs same-source baseline 0.7473 — statistically a
   tie, with *better* log-loss (0.5900 vs 0.5965) from the archive's larger n.
   Reconstructed training transfers with zero loss.
2. **Label noise is micro-move only:** cross-source agreement 94.7% at
   |ret|≥10bps, coin-flip below 1bp (exchange print noise where no true
   direction exists).
3. **Archive labels = Polymarket ground truth: 29/29 (100%)** match against
   actual cached resolutions (Jul 16). Polymarket resolves on the global
   Binance price — the archive is the *correct* label source; our
   binance.us-collected labels were the noisy ones.

**Bonus:** the 20-feature price-only model performs at parity with the
46-feature hybrid (AUC 0.747–0.755) — order-book features add ≈nothing,
confirming the importance analysis.

**Decision: PROCEED to Step 2** — full 2023-01→2026-06 build (~42 months,
~350K epochs, 16× current training data) and retrain. The Oracle collector
becomes optional insurance rather than the critical path.

---
## Exp 8 Step 2 — Multi-year archive model (2026-07-19)

**Built:** 364,676 epochs (2023-01 → 2026-06, 42 archive months, 16.9× the
previous training data), perfectly balanced (UP 49.9%). Pipeline:
[exp8/build_multiyear.py](exp8/build_multiyear.py) — month-at-a-time grids with
epoch carry-over, 9 minutes total.

**Trained:** price-only 20-feature model, train <2026, cal 2026 Q1, test 2026 Q2.
Config selection was *stable to 3 decimals* across depths 4/6/8 (calAUC
0.7473/0.7472/0.7470) — at this data size the signal no longer depends on
hyperparameter luck.

| evaluation | n | AUC | log-loss |
|---|---|---|---|
| archive 2026 Q2 (out-of-time) | 24,964 | 0.7535 | 0.5877 |
| **real collected feed, Jun 11–26 (live parity)** | 4,346 | **0.7517** | **0.5832** |
| reference: 46-feat hybrid (Exp 4, own window) | 4,325 | 0.7547 | 0.5872 |

The multi-year price-only model **matches the hybrid's AUC and beats its
calibration on the real live feed**, despite training exclusively on
reconstructed archive data with ground-truth (Polymarket-aligned) labels.
Gate table on the real feed: |p−0.5|≥0.25 selects 26% of epochs at **87.5%**
accuracy. Same signal anatomy (curr_return_90s ≈ 70 importance).

Artifacts: `exp8/catboost_price_multiyear.cbm` + calibrator.

**Why this matters for Exp 7:** the model has now seen every BTC regime since
2023 (two bears, a halving bull, chop) — the regime-dependence hypothesis for
the July forward-test failure is directly addressed. But offline numbers do not
reopen the live question by themselves.

**Next (the Exp 7 gate, unchanged):** wire this model into the Nautilus dry-run
strategy (needs a per-second price-grid feature path instead of FeatureEngine),
run 3–5 days, reconstruct forward PnL with dryrun_report. Positive with CI clear
of zero → live canary reopens. Weekly retrain becomes a 15-minute cron
(download latest month → rebuild → retrain), no collector required.

---
## Exp 9 — Forward test of the multi-year model (2026-07-22)

**Question:** does the Exp 8 model (fresh, 3.5 years of regimes, best-ever
calibration on the live feed) pass the forward gate that the stale hybrid
failed in Exp 7?

**Method:** 3.9-day dry run (Jul 18–22), 440 would-submit orders, scored
against actual Polymarket resolutions at the logged ask (optimistic bound —
100% fill, no spread).

**Result: WR 46.8%, −$83 on $4,400 staked (−1.9%/$). GATE FAILED.**

| entry price | n | WR | breakeven | PnL |
|---|---|---|---|---|
| <0.35 | 136 | 19.9% | 21% | −$84 |
| 0.35–0.50 | 82 | 42.7% | 42% | +$41 |
| 0.50–0.65 | 114 | 53.5% | 57% | −$72 |
| 0.65–1.00 | 108 | 76.9% | 75% | +$32 |

**The decisive pattern: win rate ≈ breakeven price in every bucket.** This is
the signature of an efficient market — the Polymarket price at t+90 *is* the
outcome probability, and our model's disagreements with it carry no net
information. Model quality was not the problem (Exp 8 fixed staleness, regime
coverage, and calibration; the model even beats the market slightly on
log-loss) — "slightly better forecaster" is not a tradeable edge once the
spread is paid. June's +28%/$ (Exp 5) was either a transient inefficiency that
has since been arbitraged away, or a favorable window.

**Decision: LIVE CANARY STAYS CLOSED for the taker strategy.** Three forward
evidence points now exist (Exp 5 +28%, Exp 7 −6%, Exp 9 −2% optimistic-bound);
the honest read is no current edge net of costs. Remaining avenues, in order
of merit:
1. **Maker-side variant** — post resting limits instead of crossing (the
   ~2–4¢ spread flips from cost to earnings; a breakeven taker can be a
   profitable maker). Structurally sound but introduces adverse selection;
   needs its own forward test. The one continuation with clear economics.
2. **Keep the free weekly dry-run** as an inefficiency detector — June-like
   windows may recur; the infrastructure to exploit them is built and gated.
3. New alpha research (Polymarket order-flow via poly_data, cross-market) —
   open-ended.

Cost of this entire discovery process: $0 of trading losses.

---
## Exp 10 — Maker variant forward test (2026-08-08)

**Question:** does flipping from taker (crossing the spread — Exp 9: −1.9%/$)
to maker (resting bids at best_bid+tick, capped at fair−3¢, 8bps adverse-move
cancel guard) produce positive forward PnL?

**Method:** 1.7-day dry run (Aug 1–3, pre-TWAP rules), 204 quotes / 17 cancels,
scored two-bounded against real epoch prints
([nautilus_spike/maker_report.py](nautilus_spike/maker_report.py)):
conservative = traded through our bid, optimistic = traded at it.

**Result (conservative bound): fill rate 96%, WR 52.3%, +$460 on $1,950 filled
(+23.6%/$).** Optimistic: +27.6%/$. Mean +$2.36/trade,
**bootstrap 95% CI [−$0.01, +$4.95] — GATE MISSED by one cent.** ~97% of the
bootstrap mass is positive; the sample (204 quotes, one 1.7-day window) is
simply too small to clear the bar.

Findings:
- The maker hypothesis is strongly supported: same signal, same market,
  execution flipped → −1.9%/$ becomes +23.6%/$. The spread was the whole game.
- **Adverse-move guard is too slow:** all 17 cancelled quotes were filled
  BEFORE the cancel (−$105 total, −$6.2/fill — textbook adverse selection).
  The 8bps BTC-move signal lags the Polymarket book. Parameter kept unchanged
  for comparability; revisit after the TWAP re-run.
- DOWN-side carried the PnL (+$462 at WR 56.7%); UP-side flat (−$2). Same
  DOWN-tilt seen all project; noted, not acted on (Exp 2 lesson).
- Fill rate 96% at bid+tick in these markets — fill starvation is not a risk.

**Decision: RE-RUN under the new TWAP rules (Exp 12) with the TWAP-native
retrained model, 3–5 days / ≥400 quotes.** The Aug 7 resolution change makes
this mandatory regardless of the gate; it also *helps* the maker case (TWAP
blunts expiry manipulation = less end-of-epoch adverse selection) and
Polymarket's August LP rewards ($550K on 5-min markets) subsidize exactly this
behavior. Live canary remains closed until the gate passes cleanly.

---
## Exp 11 — TWAP label migration + retrain (2026-08-08)

**Trigger:** Polymarket moved 5-min markets to TWAP resolution (effective
2026-08-07 00:00 UTC; Chainlink BTC/USD 30s TWAP stream; ties → Up), killing
single-print settlement.

**Label determination (empirical, 267–276 post-change epochs vs actual
resolutions):** candidates — old close≥open 90.6%; TWAP(last30)≥open 91.4%;
**TWAP(last30) ≥ TWAP(trailing30 at start): 94.4%** ← adopted. All 17
disagreements sit at ≤1.3bps TWAP difference (median 0.3bps) — pure
Chainlink-vs-Binance proxy noise on effective ties; agreement is ~100% on
decided epochs.

**Implementation:** `epoch_aggregate` gains `twap_last30`; label(e) =
e.twap_last30 ≥ prev.twap_last30 (the trailing start reference is exactly the
previous epoch's tail). Pre-change model archived to `models/`.

**Retrain (364,676 epochs, same protocol): out-of-time test AUC 0.7535 →
0.7716, log-loss 0.5877 → 0.5713.** The TWAP label is *more* learnable — endpoint
smoothing removed settlement noise, as hypothesized. Gate t=0.25 selects 36%
of epochs at 84.2% accuracy.

**Strategy impact:** none structurally — same features, same artifact path;
maker/taker reports score against actual resolutions so they were already
rule-agnostic. Next: Exp 12, the confirming maker dry run under TWAP rules.

---
## Exp 12 — Maker forward test under TWAP rules (2026-08-11)

**Setup:** 2.8-day dry run (Aug 8–11), TWAP-native retrained model (Exp 11,
offline AUC 0.7716), same maker logic as Exp 10. 253 quotes, 19 cancels,
0 errors.

**Result: WR 38.9%, −$127 on $2,440 filled (−5.2%/$). GATE FAILED** — and it
reverses Exp 10 (+23.6%/$) on the adjacent window.

| run | n fills | mean PnL/trade | bootstrap 95% CI |
|---|---|---|---|
| Exp 10 (Aug 1–3, pre-TWAP) | 195 | +$2.36 | [−0.01, +4.95] |
| Exp 12 (Aug 8–11, TWAP) | 244 | −$0.52 | [−2.64, +1.80] |
| **pooled maker** | **439** | **+$0.76** | **[−0.85, +2.50]** |
| pooled, excluding cancel-flagged | 403 | +$1.53 | [−0.17, +3.33] |

**Every interval straddles zero.** Two adjacent weeks produced +23.6%/$ and
−5.2%/$ with n≈200 each — the between-window variance dwarfs any systematic
edge. That is the signature of no edge, not of an edge with noise.

**Diagnostic that explains everything — live calibration conditional on fill:**

| model said | Exp 10 actual | Exp 12 actual |
|---|---|---|
| 0.6–0.7 | 46.9% | 38.0% |
| 0.7–0.8 | 76.7% | 46.3% |
| 0.8–1.0 | 64.5% | 57.1% |

Offline this model is well calibrated (log-loss 0.5713, AUC 0.7716). *Conditional
on receiving a maker fill*, its 80%+ predictions deliver 57%. Mechanism: a
resting bid on UP fills precisely when UP is getting cheaper — we systematically
buy into the move against us. **Adverse selection is not a tunable parameter
here; it is the price of providing liquidity with a signal too weak to pay for
it.** Corroborating: cancel-flagged fills lost $9.37/trade (Exp 12) — the guard
correctly identifies toxic flow but cannot outrun it.

**Decision: STOP the directional trading pursuit.** Six forward tests
(Exp 5/7/9/10/12 + pooled) with no configuration producing a CI clear of zero.
Live canary stays closed; no capital deployed. Cost of reaching this conclusion:
$0 in trading losses.

Retained assets: Nautilus live/dry-run harness, archive→retrain pipeline
(15 min, no infra), TWAP-native labeling, two-bound maker reconstruction, and a
9-experiment record of methods that repeatedly caught false positives (Exp 2, 4,
6, 9) before they cost money.

---
## Exp 13 — Order flow from Polymarket's public data API (2026-08-19)

*Write-up added 2026-10-23; numbers reproduced by re-running
[exp13/analyze_flow.py](exp13/analyze_flow.py) on the saved trade pull.*

**Question:** does Polymarket's own order flow — who buys which side, and how
much, in the first 90 seconds — carry information beyond the t+90 price? And do
individual wallets show persistent skill?

**Data:** [exp13/fetch_flow.py](exp13/fetch_flow.py) pulled every trade the
public data API returns for 1,315 resolved btc-updown-5m epochs: 2,047,913
trades from 17,077 wallets.

**Results:**
- **The feed is incomplete.** If both sides of every fill were recorded, net
  flow and aggregate PnL would both be ~0. Instead mean |net flow|/volume =
  0.099 and the recorded trades net +$164 per epoch: the API does not return
  both sides of every fill, so a flow signal built on it partly measures the
  feed rather than the market.
- **Flow adds nothing beyond price** (1,310 epochs, decision at t+90s, fit on
  the first half, scored on the second): out-of-sample log-loss 0.5389 (price
  only) → 0.5390 (+ net flow) → 0.5392 (+ large-trade flow).
  corr(net flow, outcome − price) = +0.006.
- **No detectable wallet persistence at this scale:** across 655 wallets active
  in both halves, corr(period-1 ROI, period-2 ROI) = −0.06; the top decile fell
  from +15.8% to +2.8% out-of-sample (n=65).

**Decision: REJECT the data-API route.** The question is right but this data
cannot answer it. Moved to the complete on-chain fill history (poly_data, both
counterparties of every fill) in Exp 14.

---
## Exp 14 — Participant skill in Polymarket's own flow (2026-08-19)

**Data:** poly_data v2 pull — `processed/trades.csv` (60GB) + `data/markets.csv`.
49,327 btc-updown-5m markets, **48,156 with winners embedded in metadata**
(no API needed), 2026-02-12 → 2026-08-02 (172 days). Every fill carries both
counterparty addresses. Scanned in full: 225,664 distinct wallets.

**Structural finding (closes a door):** aggregate PnL is *exactly* zero and each
economic trade appears as complementary Up/Down legs of one mint (0.67+0.33,
same tx hash). In a binary market with complete-set minting, **net order-flow
imbalance is identically zero by construction** — every Up share bought is
matched by a Down share bought. The "flow imbalance as a feature" idea is void
on logic, not evidence. (Independently confirmed: corr(net flow, outcome) =
−0.04 on a live API sample.) Only *participant identity* can carry information.

**Skill persistence — identify on period 1 (Feb 12–May 8), validate on
period 2 (May 8–Aug 2), 9,220 wallets active in both:**

- corr(ROI p1, ROI p2) = **+0.138** (Spearman +0.177)
- monotonic decile gradient: worst p1 decile → −2.55% OOS; 8th decile → +2.04% OOS
- elite cohort (p1 ROI>2%, ≥$20k vol, ≥200 trades; n=430): p1 +4.40% →
  **p2 OUT-OF-SAMPLE +1.77% on $284M volume ($+5.0M)**, bootstrap 95% CI
  [$+3.4M, $+6.9M] — excludes zero; 60.7% individually still profitable

**Skill in these markets is real and persistent.** This is the first positive
forward result in the project. But the earners' profile is market-making, not
prediction: the top wallet has 70M trades at 0.25% ROI; the rest cluster at
1–3% ROI on 1–5M trades. That is the exact economics Exp 10/12 showed *we*
cannot capture — we get adversely selected out of the spread.

**Open question under test:** is elite positioning FOLLOWABLE — does the cohort's
net Up position by t+90 predict the outcome *beyond the t+90 price*? Elite set
defined on period 1 only; tested on period 2 epochs. Result pending.

---
### Exp 14 conclusion — skill is real, but NOT followable (2026-08-26)

**Near-miss worth recording:** the first followability model showed log-loss
0.5846 → 0.5693 and a "+8.5pp edge" on 9,724 epochs. It was **look-ahead
leakage**: it priced the trade at t+90 while using elite positioning through
t+150 — 60 seconds of future. The tell was in the inputs: `e90` (legitimately
available) correlated +0.019 with the residual; `e150` (contains the future)
correlated +0.163, and carried all the model weight.

**Leak-free test — price and cohort signal measured at the SAME instant,
23,379 period-2 epochs, cohort defined on period 1 only:**

| horizon | price alone | price + elite | Δ log-loss | corr(elite, residual) |
|---|---|---|---|---|
| t+90  | 0.5846 | 0.5845 | −0.0001 | +0.019 |
| t+150 | 0.5154 | 0.5152 | −0.0001 | +0.020 |
| t+210 | 0.4543 | 0.4541 | −0.0002 | +0.023 |

**Elite positioning carries zero information beyond the price, at every
horizon.** Their +$5.0M out-of-sample profit is earned *in the price they
transact at* — liquidity provision — not in a direction anyone can copy. Note
also how fast the market itself sharpens (0.585 → 0.454 by t+210): price
absorbs everything, immediately.

**Verdict on poly_data: it cannot make this bot profitable.** It did settle the
question at high power, and produced three durable findings — (1) net flow
imbalance is structurally void in minted binary markets; (2) participant skill
is real and persistent (corr +0.138, elite +1.77% OOS, CI excludes zero); (3)
that skill is unfollowable. Combined with Exp 9/12, the market is efficient
against every signal we can construct, and the profitable role in it —
market making — is the one we measured ourselves being adversely selected out
of.

**The stop decision from Exp 12 stands, now on much stronger evidence.**
Total trading losses across 14 experiments: $0.

---
## Exp 15 — Late-epoch favorite–longshot bias (2026-08-26)

**Question:** by t+210 the TWAP reference is fixed and only ~90s of BTC drift
remains. Does the market misprice near-certain outcomes?

**Method:** 15,238 period-2 epochs with both Polymarket t+210 prices and Binance
archive BTC paths. Bucketed by BTC's lead over the epoch's start reference.

| BTC lead @ t+210 | n | actual UP | market px210 | gap |
|---|---|---|---|---|
| −99..−15 bps | 1,049 | 1.6% | 2.9% | −1.3pp |
| −8..−3 | 2,606 | 17.8% | 18.2% | −0.4pp |
| −3..3 | 5,175 | 50.1% | 49.8% | +0.3pp |
| 3..8 | 2,605 | 83.8% | 82.5% | **+1.2pp** |
| 15..99 | 906 | 97.9% | 96.5% | **+1.4pp** |

**A real inefficiency exists — classic favorite–longshot bias**, consistent in
both directions and across every bucket: near-certain outcomes carry
**+1.3% EV/$** (UP-certain n=913 win 97.9% @ cost 0.966; DOWN-certain n=1,051
win 98.4% @ cost 0.971).

**But it is ≈ one tick.** Buying at the ask rather than the last-trade price
costs 1–3c, which consumes the entire edge (the Exp 9 wall); quoting passively
re-introduces the Exp 10/12 adverse selection. **Not harvestable here** — BTC 5m
is the most competitive market on the venue, and the bias is arbitraged down to
the tick.

**LP rewards checked as an alternative economic model:** score =
discount^ticks × size, split pro-rata against professional MMs, and the time
factor *penalizes* getting filled. Requires capital and size to matter; at
$10–100 scale the share is pennies. Not viable for us.

**Strategic implication (the useful part):** favorite–longshot bias is a
*market-psychology* effect, typically 5–15pp in less contested venues. Measuring
it at 1.3pp here quantifies exactly how competitive this market is — and gives
us a **transferable yardstick** for screening other markets before committing.

---
## Exp 16 — Cross-market efficiency survey (2026-08-28)

*Write-up added 2026-10-23 from the saved `survey_results.json` and survey log.*

**Question:** Exp 15 measured BTC 5m's favorite–longshot bias at ~1.3pp — about
one tick. Are the less-traded sibling markets less efficient?

**Method:** [exp16/survey.py](exp16/survey.py) sampled 250 recent resolved
markets in each of four families and compared the favorite's price 70% of the
way through each market with how often the favorite actually won.

| market | median volume | median trades | favorite 0.50–0.65: n, won vs implied | favorite 0.65–0.80: n, won vs implied |
|---|---|---|---|---|
| BTC 15m | $13,176 | 980 | 248: 49.2% vs 52.8% (−3.6pp) | — |
| ETH 5m | $2,364 | 345 | 207: 53.6% vs 56.0% (−2.4pp) | 42: 69.0% vs 68.5% (+0.5pp) |
| SOL 5m | $781 | 169 | 210: 58.1% vs 56.0% (+2.1pp) | 38: 60.5% vs 69.9% (−9.3pp) |
| XRP 5m | $331 | 128 | 146: 56.8% vs 57.3% (−0.5pp) | 89: 76.4% vs 69.6% (+6.9pp) |

**Findings:** no consistent bias — the sign flips across markets and price
bands, and the only gaps larger than ~4pp come from 38–89 markets each, where
the 95% interval is roughly ±9–16pp. Meanwhile median volume falls ~40× from
BTC 15m to XRP 5m, so even a real gap in the thin markets could only be traded
in tiny size.

**Decision: no crypto up/down family offers a measurable mispricing worth the
spread.** Every signal so far was built from the same public price data and
arbitraged to the tick, which pointed the search at a market where public data
takes real work to use: weather (Exp 17).

---
## Exp 17 — Weather markets: forecast-skill baseline (2026-08-28, IN PROGRESS)

**Correction to the previous session's dismissal:** Polymarket *does* run daily
high-temperature markets (12 cities/day, 11 integer-°C buckets each, NOAA
station resolution, $22–84K volume per event — higher than a BTC 5m market).
My earlier "weather is out on cadence" conclusion came from a tag search that
only surfaced long-horizon climate/disaster markets. Wrong.

**Why this market class is structurally different from BTC up/down:** every
failure in Exp 1–15 traced to using *the same information everyone else has*
(BTC price). Weather forecasts are public but require real work to use —
archived model runs, station-vs-gridpoint bias correction, uncertainty →
bucket-distribution conversion. Work is what stops an edge being arbitraged flat.

**Leak-free forecast source established:** Open-Meteo `previous-runs` API,
variable `temperature_2m_previous_day1` = what the model said 24h before the
target date (verified NOT analysis data: RMSE 0.68°C, consistent with genuine
day-ahead skill). Observed truth from the archive API. Both free.

**Forecast skill baseline (93 days × 12 cities, bias-corrected):**

| city | err sd | exact-bucket | clim sd |  | city | err sd | exact-bucket |
|---|---|---|---|---|---|---|---|
| madrid | 0.55 | **64.5%** | 3.36 | | tokyo | 0.98 | 45.2% |
| seoul | 1.11 | 48.4% | 2.87 | | miami | 1.03 | 44.1% |
| paris | 1.20 | 47.3% | 5.26 | | moscow | 1.03 | 43.0% |
| london | 0.90 | 40.9% | 4.59 | | beijing | 1.17 | 38.7% |
| nyc | 1.50 | 35.5% | 3.55 | | toronto | 1.49 | 32.3% |
| chicago | 1.53 | 31.2% | 3.68 | | los-angeles | 1.31 | 28.0% |

Mean 41.6% exact-bucket accuracy vs climatological sd 1.6–5.3°C. Skill varies
2× by city → city selection is itself an edge lever.

**Suggestive but NOT yet a result:** a live Madrid market priced its modal
bucket at 0.32 while Madrid's day-ahead forecast hits the exact bucket 64.5% of
the time. The comparison is not valid until market price and forecast are taken
at the SAME instant (markets sharpen near close). That like-for-like test is
built ([weather/pilot.py](weather/pilot.py): decision point = 24h before close
on both sides) and **blocked only on Polymarket network access** (DNS reverted;
needs 1.1.1.1).

**Next:** run the pilot, verify the bucket-mapping rule empirically against
resolved winners (round vs floor — getting it wrong manufactures a fake edge),
then compare market vs forecast log-loss. Gate before any capital: forecast must
beat market by materially more than the spread, out-of-sample.

---
### Exp 17 result — market beats our forecast decisively (2026-08-28)

**Data:** 331 resolved city-days (12 cities × ~30 days), market prices taken
24h before close, forecasts from `previous_day1` only. Bucket parsing verified
empirically against resolved winners (US cities use 2°F ranges, others 1°C
singles; NYC 24.7°C=76.5°F → winner "76-77°F" ✓, Madrid 32.7°C → "33°C" ✓).
Station offsets and forecast sd fitted on the TRAIN half; held-out scored once.

| held-out (n=166) | log-loss | modal-bucket hit |
|---|---|---|
| **market-implied** | **1.3742** | **43.4%** |
| forecast, global sd 1.5 | 1.8462 | 28.9% |
| forecast, per-city sd | 2.0325 | 27.7% |

**Sharpening made it worse** — the tell. Madrid has the sharpest forecast
(sd 0.55) yet the worst gap (Δ +1.455): a tight distribution centred on the
wrong value is far worse than a loose one. **Diagnosis: we forecast an
Open-Meteo gridpoint; the market resolves on a specific NOAA station.** Our
forecast is skillful for the wrong quantity (41.6% exact-bucket vs *gridpoint*
truth, but 27.7% vs *station* truth).

**Calibration of how good the market is:** 43.4% modal-bucket accuracy against
the station — i.e. the market forecasts the station at least as well as our best
model forecasts the (easier) gridpoint. These participants are competent.

**Verdict: this implementation fails; hypothesis not refuted but downgraded.**
A fair rematch needs station-level forecasts (NWS point forecasts / METAR-TAF
for the exact site), multi-model ensembles, and offsets fitted on ~90 days
rather than 15 — roughly a day of work, against opponents already doing it.
Longshot-band EV estimates (>15pp band showing +0.46/$) are NOT trusted: they
come from a demonstrably miscalibrated model, n=115 non-independent outcomes,
and flip sign across adjacent bands.

**No capital deployed. Running total across 17 experiments: $0 in trading losses.**

---
