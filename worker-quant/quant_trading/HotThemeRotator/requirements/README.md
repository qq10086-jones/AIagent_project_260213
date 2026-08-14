# Dependency locks (P37-03 step 2)

Four locks, one per install contract. All are generated — never hand-edited.

```powershell
python tools/compile_locks.py                       # recompile from the pinned environment
python tools/compile_locks.py --refresh-environment # re-snapshot this machine first
```

| file | install contract | what it is for |
|---|---|---|
| `bootstrap.txt` | pip + setuptools | the BUILD toolchain, installed first (see below) |
| `runtime.txt` | base `dependencies` | the daily operational lane (`tools/daily_routine.py`) |
| `fast.txt` | `.[test,dashboard]` | `pytest -m "not slow"` — the Rule 15.2 readiness gate |
| `slow.txt` | `.[test,research]` | `pytest -m slow` — the research regression lane |
| `dev.txt` | all extras | everything, including the standalone streamlit app |

Install one with:

```powershell
python -m pip install --require-hashes -r requirements/bootstrap.txt
python -m pip install --require-hashes -r requirements/fast.txt
python -m pip install --no-deps --no-build-isolation --no-index .
```

## bootstrap.txt is not optional (P37-03 step 3)

A fresh CPython 3.13 venv contains **`pip 24.2` and nothing else** - ensurepip
stopped shipping setuptools in 3.12. Measured, not assumed. So on a clean
machine:

```
pip install .                       -> build isolation downloads an UNLOCKED
                                       setuptools from PyPI, past every hash here
pip install --no-build-isolation .  -> ModuleNotFoundError: setuptools
```

Neither is acceptable, and "the venv usually has setuptools" is false. The build
toolchain is therefore locked like everything else and installed first, and the
project is then built with `--no-build-isolation --no-index` so nothing is
fetched outside the lock.

One wrinkle worth recording: plain `pip freeze` **omits** pip, setuptools and
wheel, so the first bootstrap lock resolved pip unconstrained and picked
`26.2.1` where this machine runs `25.3`. The environment snapshot now uses
`pip freeze --all`, and the build toolchain is pinned to versions that have
actually run here.

## What these locks mean, and what they do not

**They reproduce the environment the suite passed in — not the newest set of
packages that satisfies `pyproject.toml`.** Those are different answers. An
unconstrained resolve of this project today picks `yfinance 1.6.0` and
`beautifulsoup4 4.15.0`, neither of which has ever run here; locking that would
be invented compatibility with more decimal places. So every resolution is
constrained by `verified-environment.txt`, and `tools/compile_locks.py` fails
if any pin differs from it.

**They are for CPython 3.13 on x86_64 Windows.** That is stated in each file's
resolution and is not incidental: hashes pin specific wheels, and wheels are
per-platform. `pyproject.toml` used to declare `requires-python = ">=3.10"` on
the strength of a syntax scan; that floor was **not locked and not tested**, so
step 3 narrowed it to `>=3.13`, which is the only interpreter this code has ever
run on. Widen it again behind a verified matrix, never behind a scan.

**Installed and exercised from empty environments (P37-03 step 3).**
`python tools/verify_clean_environments.py` builds a fresh venv per lane,
installs from these locks with `--require-hashes`, and then runs the lane.
See the P37-03 entry in `docs/01_TASKS.md` for the recorded evidence. What is
still NOT claimed: any platform or interpreter other than CPython 3.13 on
x86_64 Windows, and any install of `api/` or `tools/` - those are deliberately
not packaged and run from the checkout.

## verified-environment.txt

The `pip freeze` of the interpreter that ran the smoke lane, used as a
constraint. It is an **input**, not an install target: it is the whole machine,
including packages that have nothing to do with this project.

It is taken with `python -m pip freeze` from the running interpreter, never with
`uv pip freeze`. That distinction cost a wrong lock once: **uv defaults to its
own managed interpreter**, a CPython 3.12.11 under `AppData` holding
huggingface/transformers, which reports `requests==2.32.5` where the interpreter
that runs pytest has `2.34.2`. A lock built that way pins a machine nobody runs.
`tools/compile_locks.py` therefore always passes an explicit `--python-version`
and `--python-platform`.

## Things the locks confirmed independently

- **`numba` is nobody's declared dependency.** It appears in `slow.txt` only,
  at `0.65.1`, pulled in by `vectorbt`. Step 1 concluded this from a static
  scan (no module imports numba); the resolver agrees.
- **The fast lane genuinely needs the `dashboard` extra.** `pydantic` and
  `starlette` are in `fast.txt` because the test suite reaches `api/`.
- **`vectorbt` and `numba` are absent from `fast.txt`,** which is what makes
  the fast lane cheap to install and is why the deferred-import boundary in
  `backtesting/vectorbt_spike.py` is load-bearing rather than stylistic.

`tests/unit/test_dependency_locks.py` re-checks all of this offline on every
smoke run.
