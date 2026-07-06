# Quality-Gated Catalyst Rerank Design

## Problem

The current candidate stack selects liquid recent winners, then reranks them with broad
theme-news heat. This makes sector-adjacent names look like company-specific catalysts.
The live examples are 2162.T, 6584.T, and 8604.T: each was mechanically mapped to a hot
theme, but no served evidence proved a same-day company catalyst.

## Design

The catalyst layer becomes evidence-tiered:

- `company`: the news item explicitly links the symbol, or the metadata company name
  appears in the news title. This may receive the full catalyst weight.
- `sector`: the ticker only maps to a hot theme through metadata. This may receive only
  a small ordering nudge and must not create a theme-leader badge.
- `none`: no hot theme evidence.

The rerank layer adds quality gates before sorting:

- Company catalyst uses `company_news_weight=0.30`.
- Sector-only catalyst uses `sector_news_weight=0.08`.
- Sector-only evidence takes `sector_only_penalty=0.08` because it is exposure
  rather than company news.
- Sector-only evidence with `mom_20 >= 0.30` takes `sector_chase_penalty=0.25`.
- Low fundamentals reduce the blended score by up to `fundamental_penalty_weight=0.12`.
- Stale recent prices reduce the blended score by `stale_price_penalty=0.12` when the
  serialized candidate row exposes repeated unchanged recent closes.

The dashboard remains honest: displayed `score` stays the raw uncalibrated screener
score. New served fields explain ordering: `catalystEvidence`, `catalystEvidenceLevel`,
`companyCatalyzed`, and `sectorCatalyzed`.

## API Consistency

`/api/symbol/{ticker}/profile` must read the same freshest screener snapshot as
`/api/dashboard`; otherwise a current candidate can appear as `in_screener=false`.

## Governance

Rule 11.12 is extended: only `company` evidence can render the persuasive
news-catalyst / theme-leader badge. Sector-only evidence is an exposure label, not a
company catalyst.

## Tests

Unit tests must prove:

- A sector-only candidate is not `news_catalyzed` and cannot be a theme leader.
- A company-linked candidate keeps the catalyst badge path.
- A weak-fundamental sector-only candidate is pushed below a cleaner pure-screener
  candidate when scores are otherwise close.
- Profile lookup uses the freshest HTR screener snapshot.
