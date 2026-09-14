# SP5 — the scorecard label: teach the model *how much* a decision was won by

**Status:** pre-registration written 2026-09-13, **before any arm was built or
scored**. Completion notes appended after.

**Goal.** Every experiment this project has run — eight feature blocks, a
25-configuration capacity search, a calibration audit — changed what the model
is *told*. None changed what it is *trained on*. A 30–27 sweep and a 29–28
split decision are currently the same training example: `y_winner = 1`. The
judges' scorecards, already scraped and sitting unparsed in
`data/raw/fight.csv`'s `details` column, say which was which.

---

## Why this is worth a pre-registration rather than a try

The prior is not good and should be stated first. Eight blocks shipped one.
The capacity search produced a winner that the fresh-seed re-score destroyed.
The odds-free ceiling is bounded: the market beats the deployed hybrid by
**+0.036** pooled log-loss and still leaves 36% of fights near a coin flip,
so nothing here can be worth more than a fraction of that.

What makes this one different from block nine is the mechanism, and the
mechanism is falsifiable:

- **79% of out-of-fold predictions sit in bands where the model is 52–59%
  accurate** (`models/calibration_audit.json` and the residual map). The model
  is not confidently wrong; it is mostly silent.
- It has no way to learn the difference between *"this fight was genuinely a
  toss-up"* and *"I failed to see who would win"*. Both are `y_winner = 1` at
  p ≈ 0.5, and the second is punished identically to the first.
- **930 of 4,461 parsed decisions (20.8%) had judges disagree with each
  other** — ground truth that those fights were close, currently discarded.

So the hypothesis has a specific predicted signature, not just a direction,
and §6 gates on it.

---

## 1. The data, and what is actually there

`details` on a decision row reads `David Therien 28 - 29. Greg Jackson 29 -
28. Nelson Hamilton 27 - 30.` Measured before writing this document (counting
rows is not measuring an arm):

| | |
|---|---|
| decisions in `data/raw/fight.csv` | 4,995 |
| with all three judge cards parseable | **4,461 (89.3%)** |
| distinct judges / judge-cards | 573 / 13,383 |
| fights where judges disagreed | 930 (20.8%) |
| per-card point margins | 1pt ×6,672, 3pt ×5,051, 4pt ×602, 2pt ×444, 5pt ×344, 0pt ×187 |

The 534 unparseable decisions (10.7%) are masked, not guessed at.

## 2. The label

`y_margin` — **the mean per-judge point margin, oriented A-minus-B**, where A
is the feature table's corner A *after* the md5-parity swap.

- Finishes carry no scorecard: `NaN`, masked. The auxiliary target therefore
  covers ~4,461 of 11,290 rows (39.5%).
- Mean per-judge rather than summed, so a fight scored by two judges (if the
  parser ever sees one) is on the same scale; rather than rounds-won, so 10-8
  rounds and point deductions need no reverse-engineering.

**Orientation is the known bug class here.** Out-of-fold predictions have been
mis-paired by row position twice in this repo. `y_margin` is built on
`fight_id` and swapped by the *same* `swap_corner` call that orients every
other label in `mma.features`, and §7 pins it.

**`y_margin` is a LABEL and never a feature.** It is the outcome of the fight
being predicted. It enters `features.parquet` beside `y_winner`, `y_method`
and `y_finish_round`, is registered in no feature block, and `mma.serving` is
not touched by this work at all — the auxiliary head is training-time only and
nothing at prediction time needs a scorecard.

## 3. The arms — fixed now, six of them

Both blend members get the information in the form each can use. XGBoost has
no multi-task head; the trees take it as a soft label instead.

| arm | member | change | λ |
|---|---|---|---|
| **T0** | torch | incumbent, `margin_scale = 0` | — |
| **T1** | torch | masked margin head, Huber loss | 0.1 |
| **T2** | torch | masked margin head, Huber loss | 0.3 |
| **T3** | torch | masked margin head, Huber loss | 1.0 |
| **X1** | xgb | soft label `sigmoid(mean_margin / 2)` on decisions, hard 0/1 on finishes | — |
| **X2** | xgb | soft label `sigmoid(mean_margin / 4)` (flatter) | — |

**Six arms is the whole search.** No λ outside `{0.1, 0.3, 1.0}`, no soft-label
temperature outside `{2, 4}`, no architecture changes. Adding any arm later
must be recorded in the completion notes as an addition, with the reason, and
counted in §5.

`margin_scale = 0` **must reproduce the incumbent bit-for-bit**. The margin
head is constructed only when λ > 0, so existing checkpoints stay loadable and
T0 is not a re-run of the incumbent but the incumbent itself.

## 4. The primary endpoint

**Pooled out-of-fold winner log-loss**, walk-forward, fold years 2018–2025,
via `scripts/run_walkforward.py` — the same harness, the same folds, the same
artefact shape as every other experiment here.

Not the margin head's own accuracy. The margin head is a *teacher*, and a
teacher that predicts margins beautifully while the winner marginal stands
still has failed.

## 5. The bar, and the multiplicity it has to survive

The standard v3 §5 bar: a candidate ships only if it beats its paired
incumbent by more than **max(0.003, 2σ_seed)**.

| member | σ_seed | 2σ | bar |
|---|---|---|---|
| torch (5-seed) | 0.000346 | 0.00069 | **0.003** |
| xgb (single fit) | 0.00125 | 0.0025 | **0.003** |

Six arms of selection optimism, at σ√(2 ln N) with N = 6:

| member | expected best-of-6 under the null |
|---|---|
| torch | 0.00066 |
| xgb | **0.00237** |

The xgb figure is 79% of the bar, which is not comfortable. So, as SP2.1's
rule 2 already requires and as SP2.1 itself demonstrated — its A1-XGB cleared
the bar at −0.0039 on seed 0 and failed the fresh-seed re-score at −0.0024 —
**anything that clears the bar must also clear it on fresh seeds** (torch
5–9; xgb `--model-seed` 1–3), measured against the fresh-seed paired
incumbent. Both are required to ship. One without the other is recorded as a
failure.

## 6. The mechanism gate — reported and gated

The hypothesis is not "margins help" but "margins help *in the band where the
model is currently silent*". So:

- **Pre-registered secondary:** the log-loss improvement restricted to
  out-of-fold rows with |p − 0.5| < 0.10 (n ≈ 979 of the exploration window's
  bands; recomputed on the full pooled set at scoring time).
- If an arm clears the bar **but its gain does not concentrate in that band**,
  it ships on the primary endpoint — the bar is the bar — but the completion
  notes must record that **the stated mechanism was wrong**, in those words.
  A right answer for the wrong reason is still a finding, and pretending
  otherwise is how a lucky arm becomes a story.

**Also reported, never gated:** pooled ECE, accuracy, per-fold deltas, the
margin head's own MAE on held-out decisions, and the split-decision subset
(n = 930) where the mechanism should bite hardest.

## 7. Guards

1. **Out of fold only.** Fold year *Y*'s predictions come from a model fitted
   on fights before *Y−1*, early-stopped on *Y−1*. The deployed model trains
   through the latest event and is in-sample on all of these; no headline uses
   it.
2. **`y_margin` never becomes a feature.** A test asserts it is absent from
   `mma.feature_blocks.columns_for(...)` for every registered block and from
   the served feature row.
3. **Orientation pinned.** A test builds a fixture whose scorecard favours the
   corner the md5 swap moves, and asserts `y_margin` follows the swap — the
   fixture fails if the label is joined positionally.
4. **Masking pinned.** A test asserts finishes and unparseable decisions
   contribute exactly zero gradient to the margin term, and that
   `margin_scale = 0` reproduces the incumbent loss to within float error.
5. **Serving untouched.** `mma.serving.feature_row` is byte-identical after
   this work; a test asserts the served column list is unchanged.
6. **Every arm reported, including the losers**, with per-fold numbers.

## 8. Decision rule

- **Ships** iff its 5-seed (torch) / seed-0 (xgb) result clears
  max(0.003, 2σ) against its paired incumbent **and** the fresh-seed re-score
  clears it too.
- **Does not ship** otherwise, and is recorded in `README.md` beside the other
  negatives with its numbers, as `rankings` and the capacity search were.
- **No arm is promoted on the strength of the margin head's own performance,
  the mechanism gate, or a slice.** The primary endpoint decides.

## 9. What this does not do

It does not touch the simulator, the hazard model, the blend weight or the
temperature. It does not use judge *identity* — the tendencies of the 573
named judges are a separate idea with a serving-parity problem (we know the
panel historically but do not scrape it prospectively) and are deliberately
out of scope here. It deploys nothing on its own; a passing arm becomes a
deployment only through the existing refit path.

---

## Completion notes

*(appended after the arms are scored — nothing above this line changes)*

**Scored 2026-09-13. Nothing ships. `models/walkforward/sp5_decision.json`.**

### The primary endpoint: five comparisons, five failures

| arm | member | change | pooled | delta | bar | |
|---|---|---|---|---|---|---|
| T0 | torch | incumbent | 0.6487 | — | — | |
| T1 | torch | λ=0.1 | 0.6493 | +0.00060 | 0.003 | fails |
| **T2** | torch | **λ=0.3** | **0.6484** | **−0.00030** | 0.003 | fails |
| T3 | torch | λ=1.0 | 0.6497 | +0.00100 | 0.003 | fails |
| X0 | xgb | incumbent | 0.6488 | — | — | |
| X1 | xgb | soft T=2 | 0.6502 | +0.00140 | 0.003 | fails |
| X2 | xgb | soft T=4 | 0.6542 | +0.00540 | 0.003 | fails |

The best arm is a tenth of the bar. It is also **less than half the selection
optimism a best-of-six search carries** — σ√(2 ln 6) is 0.00066 for torch —
so T2's −0.0003 is smaller than what six equivalent arms would produce from
nothing. The fresh-seed confirmation was therefore not run: §5 requires it of
an arm that clears, and none did.

### The mechanism was wrong, and it was wrong backwards

§6 pre-registered where the gain was predicted to land. Paired on `fight_id`
(n=4,856), T2 against T0:

| band | n | share | delta |
|---|---|---|---|
| coin-flip, \|p−0.5\| < 0.10 | 2,490 | 51.3% | **+0.00074** |
| decided, \|p−0.5\| ≥ 0.10 | 2,366 | 48.7% | **−0.00132** |

**The stated mechanism was wrong.** The margin label did not help where the
model is silent; it helped, very slightly, where the model was already
confident, and *hurt* in the coin-flip band it was designed for.

That reverses the argument the experiment was built on. The reasoning was
that a split decision is ground truth that a fight was close, so the label
should teach the model to tell a genuine toss-up from a fight it failed to
read. What the label actually carries is mostly the opposite: `|margin|`
averages 2.29 on unanimous decisions against 0.54 on splits, so most of its
variance is *how big a blowout was*. Learning to recognise blowouts sharpens
the decided end. It says nothing about the murky middle, and spending trunk
capacity on it costs a little there.

### Two things worth keeping

**T0 reproduced the incumbent bit-for-bit** — pooled winner log-loss,
accuracy, Brier, ECE, joint log-loss and both macro-F1s identical to the
committed `revalidation_torch_A` figures. The design goal that
`margin_scale = 0` is the incumbent rather than a re-run of it held exactly,
so the comparison is paired in the strongest available sense.

**More softening is monotonically worse on the trees**: T=4 (+0.0054) loses
four times as much as T=2 (+0.0014). Softening a label discards information
the trees were using, and the amount lost scales with how much you soften —
the same shape as SP2.2's isotonic result and this month's calibration audit,
where flattening bought calibration by spending discrimination.

### What this does not say

It does not say the scorecards are worthless — only that `y_margin` as an
auxiliary regression target and as a softened label both fail. Judge
*identity* (573 named judges, 13,383 cards) is untouched and was deliberately
out of scope (§9); it remains blocked on the serving-parity question, not on
this result.

The label, parser and table stay in the repo. `y_margin` costs nothing when
unused — `margin_scale` defaults to 0, no head is constructed, and the column
is in both members' `TARGETS` so neither can model it.
