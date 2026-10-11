# Contributing

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,model]"
cp .env.example .env     # add your own FRED key; never commit .env
tickercast all           # rebuild data locally (data/ is git-ignored)
```

## Workflow

1. Branch from `main`: `git checkout -b feature/short-name`.
2. Make changes. Settings belong in `configs/data.yaml`, not hard-coded.
3. Run `pytest` (offline, under 5 seconds).
4. If you changed data or features, run `tickercast all` and commit the updated `reports/`
   so reviewers can see the effect.
5. Open a pull request. Describe what changed and paste the report summary lines.

## Rules for new features

- Use only data available at the close of the row's date. Backward-looking windows only;
  no `shift(-k)` outside `features/targets.py`.
- Lower-frequency data (quarterly, monthly, events) must carry the date it became
  **public**, joined with an as-of join.
- Target columns must start with `fwd_` or `label_`. Everything else is treated as a model input.
- Document the feature in `docs/03_features.md` (what it is, the formula, why it might help).
- Make sure the truncation test still passes (`tickercast validate-features`).

## Never commit

`data/`, `.env`, model checkpoints (`*.pt`). These are already in `.gitignore`.
