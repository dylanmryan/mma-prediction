# `data/external/rankings.parquet` — official UFC divisional rankings

| | |
|---|---|
| Source | <https://www.kaggle.com/datasets/jerzyszocik/ufc-rankings-history> |
| Licence | CC0: Public Domain |
| Dataset version | 53 |
| Coverage | 2013-02-04 .. 2026-09-03 (533 weekly snapshots) |
| Rows | 86850 (from 89778 divisional source rows) |
| Divisions | 12, named exactly as our `weight_class` |
| Fighters | 630 of our 4581 (0.1375) appear at least once |

Rank 0 is the champion. Pound-for-pound rows are dropped: they are not a
division a bout takes place in, they re-list fighters already ranked in their
own division, and the source spells them four different ways.

Regenerate with `python scripts/build_rankings.py`; the raw CSV is
kagglehub-downloaded and never committed.

## Name matching, and what it costs

The source identifies fighters by NAME only, so this is the one derived table
that cannot be joined by id. It uses the prospective pipeline's never-guess
matcher (`mma.prospective.match_fighter_id`): exact unicode-normalised match,
then accent-folded, and anything ambiguous under either tier is left unmatched.

**631 of 656 distinct names matched**
(620 exact, 11 accent-folded), covering
0.9786 of the divisional source rows.

An unmatched name is dropped, so that fighter reads as *unranked* — the same
state as someone genuinely outside the top 15. That conflation is the honest
cost of a name-keyed source; it is not repaired with a fuzzy match, because a
wrong id is worse than a missing one. The names it affects are ring names the
UFC uses and our fights table does not ("Rampage Jackson", "Cris Cyborg",
"Minotauro Nogueira"), spellings ("Georges St. Pierre" vs "Georges St-Pierre"),
and names shared by two of our fighters, which are ambiguous by construction:

| name | why it did not match |
|---|---|
| `Bobby Green` | no fighter in fighters.parquet matches name 'Bobby Green' (exact or accent-folded) |
| `Brianna Van Buren` | no fighter in fighters.parquet matches name 'Brianna Van Buren' (exact or accent-folded) |
| `Bruno Silva` | name 'Bruno Silva' is ambiguous: matches 2 fighter ids |
| `Costas Philippou` | no fighter in fighters.parquet matches name 'Costas Philippou' (exact or accent-folded) |
| `Cris Cyborg` | no fighter in fighters.parquet matches name 'Cris Cyborg' (exact or accent-folded) |
| `Georges St. Pierre` | no fighter in fighters.parquet matches name 'Georges St. Pierre' (exact or accent-folded) |
| `Jan Błachowicz` | no fighter in fighters.parquet matches name 'Jan Błachowicz' (exact or accent-folded) |
| `Jean Silva` | name 'Jean Silva' is ambiguous: matches 2 fighter ids |
| `Jose Miguel Delgado` | no fighter in fighters.parquet matches name 'Jose Miguel Delgado' (exact or accent-folded) |
| `Klaudia Syguła` | no fighter in fighters.parquet matches name 'Klaudia Syguła' (exact or accent-folded) |
| `Lone’er Kavanagh` | no fighter in fighters.parquet matches name 'Lone’er Kavanagh' (exact or accent-folded) |
| `Melissa Dixon` | no fighter in fighters.parquet matches name 'Melissa Dixon' (exact or accent-folded) |
| `Michael McDonald` | name 'Michael McDonald' is ambiguous: matches 2 fighter ids |
| `Michael Venom Page` | no fighter in fighters.parquet matches name 'Michael Venom Page' (exact or accent-folded) |
| `Minotauro Nogueira` | no fighter in fighters.parquet matches name 'Minotauro Nogueira' (exact or accent-folded) |
| `Mirko Cro Cop` | no fighter in fighters.parquet matches name 'Mirko Cro Cop' (exact or accent-folded) |
| `Nina Ansaroff` | no fighter in fighters.parquet matches name 'Nina Ansaroff' (exact or accent-folded) |
| `Rafael Feijao` | no fighter in fighters.parquet matches name 'Rafael Feijao' (exact or accent-folded) |
| `Rampage Jackson` | no fighter in fighters.parquet matches name 'Rampage Jackson' (exact or accent-folded) |
| `Rocco Martin` | no fighter in fighters.parquet matches name 'Rocco Martin' (exact or accent-folded) |
| `Ronaldo Souza` | no fighter in fighters.parquet matches name 'Ronaldo Souza' (exact or accent-folded) |
| `Seohee Ham` | no fighter in fighters.parquet matches name 'Seohee Ham' (exact or accent-folded) |
| `Tecia Torres` | no fighter in fighters.parquet matches name 'Tecia Torres' (exact or accent-folded) |
| `Tim Johnson` | no fighter in fighters.parquet matches name 'Tim Johnson' (exact or accent-folded) |
| `Ulka Sasaki` | no fighter in fighters.parquet matches name 'Ulka Sasaki' (exact or accent-folded) |

Three of these are mechanical near-misses rather than genuinely different
people: `Jan Błachowicz` / `Jan Blachowicz` and `Klaudia Syguła` /
`Klaudia Sygula` differ by a stroked Latin letter, which NFKD does not
decompose the way it decomposes an accent, and `Lone’er Kavanagh` /
`Lone'er Kavanagh` differ by a curly versus a straight apostrophe. Extending
`mma.prospective.fold_accents` with a stroked-letter map and apostrophe
normalisation would close all three without weakening the never-guess rule
(a folded match must still be unique). It is left as a follow-up rather than
done here, because `fold_accents` is on the live prediction path and this
table is not.

## Point-in-time

Every row carries the date the ranking was PUBLISHED. `mma.rankings` joins the
most recent publication **strictly before** each fight's date
(`pandas.merge_asof(..., allow_exact_matches=False)`), so a fight can never see
a ranking that its own result produced. Unlike the pre-UFC snapshot next door,
this source is refreshed weekly, so its coverage does not decay away from the
fights the deployed model actually serves.
