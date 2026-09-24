## What & why

<!-- What changes, and what problem it solves. -->

## Checklist

- [ ] `pytest tests` passes (or the failures are pre-existing and named below)
- [ ] `ruff check core tests` is clean
- [ ] No new files in `outputs/`, no `.pyc`/`.db`/`.DS_Store` added to git
- [ ] `experiments/` and `data/benchmark/` are untouched
- [ ] New thresholds went into `core/config.py`, not inline
- [ ] No metric units / GPS claimed where the data has none

## Reconstruction impact

<!-- Does this change point-cloud geometry? If so, give before/after point
     counts and confidence stats, and say which dataset you ran. -->

- [ ] No geometry change (refactor / docs / tooling only)
