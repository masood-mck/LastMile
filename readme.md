Databricks apps: Compute -> apps or https://adb-1011835552418770.10.azuredatabricks.net/apps-v2?o=1011835552418770

## What To Commit

Commit these files:

- `apps/lm_cost/lm_cost_app.py`
- `apps/lm_cost/precompute_lm_cost_all.py`
- `apps/lm_cost/precompute_lm_cost_all.ipynb`
- `apps/lm_cost/preprocess_lm_cost_data.ipynb`
- `apps/lm_cost/precompute_lm_cost_scorecard.ipynb`
- `apps/lm_cost/Dockerfile`
- `README.md`
- `.gitignore`

Do not commit generated artifacts from `apps/lm_cost/data`:

- `LM_CS_slim_prepared.pkl` (very large)
- `LM_CS_slim_prepared.parquet`
- `LM_CS_slim_preprocess_manifest.json`
- `LM_CS_scorecard_default.pkl`
- `LM_CS_scorecard_default_manifest.json`

These generated files are now ignored by git.

## Docker Build Behavior

The Dockerfile now precomputes artifacts at build time when `data/LM_CS_slim.csv.gz` is present:

- It runs `python precompute_lm_cost_all.py --input data/LM_CS_slim.csv.gz --outdir data`
- This keeps app startup fast in the container without committing generated artifacts to git.

If `data/LM_CS_slim.csv.gz` is not present at build time, the app still runs and falls back to live computation.

## LM Cost App: Local Run Instructions

### 1) Install dependencies

From the repository root:

```powershell
Set-Location "apps/lm_cost"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Generate precomputed data artifacts

Run the consolidated preprocessing script (writes outputs into `apps/lm_cost/data` by default):

```powershell
python precompute_lm_cost_all.py
```

Optional custom paths:

```powershell
python precompute_lm_cost_all.py --input "full_path_to_source_csv_or_csv_gz" --outdir "full_path_to_output_folder"
```

Expected outputs:

- `LM_CS_slim_prepared.pkl`
- `LM_CS_slim_prepared.parquet` (if parquet engine is available)
- `LM_CS_slim_preprocess_manifest.json`
- `LM_CS_scorecard_default.pkl`
- `LM_CS_scorecard_default_manifest.json`

### 3) Run Streamlit app

From `apps/lm_cost`:

```powershell
streamlit run lm_cost_app.py
```

The app auto-loads precomputed artifacts from `apps/lm_cost/data` when available, and falls back to live computation if artifacts are missing or filters/thresholds are changed from defaults.

### If push previously failed due large files

If a local commit accidentally included generated artifacts, remove and recommit before pushing:

```powershell
git rm --cached apps/lm_cost/data/LM_CS_slim_prepared.pkl apps/lm_cost/data/LM_CS_slim_prepared.parquet apps/lm_cost/data/LM_CS_slim_preprocess_manifest.json apps/lm_cost/data/LM_CS_scorecard_default.pkl apps/lm_cost/data/LM_CS_scorecard_default_manifest.json
git commit --amend --no-edit
git push
```

### 4) Optional notebook runners

You can also run the notebook helpers:

- `apps/lm_cost/preprocess_lm_cost_data.ipynb`
- `apps/lm_cost/precompute_lm_cost_scorecard.ipynb`

Both now call the consolidated script `precompute_lm_cost_all.py`.

