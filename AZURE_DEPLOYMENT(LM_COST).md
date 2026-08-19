# LM Cost - Azure Migration and Deployment Runbook

Runbook for migrating and deploying the Streamlit LM Cost app (`apps/lm_cost/lm_cost_app.py`) to Azure.

This guide assumes the app already runs in Databricks Apps and you now want it in Azure.

---

## 1. Current App Baseline (from repo)

- App entry: `apps/lm_cost/lm_cost_app.py`
- Databricks app command is already defined in `apps/lm_cost/app.yaml`:
  - `streamlit run lm_cost_app.py --server.port $DATABRICKS_APP_PORT --server.address 0.0.0.0 --server.headless true`
- Python dependencies in `apps/lm_cost/requirements.txt`:
  - `streamlit`, `plotly`, `pandas`, `numpy`, `openpyxl`, `scikit-learn`
- Data input pattern in app code:
  - Databricks table via env var `NORM_DATA_TABLE`
  - Databricks path via env var `NORM_DATA_PATH`
  - Local fallback file: `apps/lm_cost/data/LM_CS_slim.csv.gz` (or `.csv`)

---

## 2. Migration Choice

Use one of these two paths:

1. **Azure Databricks Apps (recommended, lowest change):**
   - Keep app architecture and env vars almost unchanged.
   - Best if data remains in Unity Catalog / Volumes.

2. **Azure App Service (containerized Streamlit):**
   - Best if you need private endpoint app access pattern similar to FTL app.
   - Requires Docker image build and app service networking setup.

If unsure, start with Option 1 first.

---

## 3. Option 1 - Deploy to Azure Databricks Apps

### 3.1 Prerequisites

- Azure Databricks workspace exists.
- Repo is connected in Azure Databricks Repos.
- You have permission to create Apps and read source data tables/volumes.

### 3.2 Move code to Azure Databricks workspace

1. In Azure Databricks, open Repos.
2. Link to repo: `https://github.com/masood-mck/LastMile.git`
3. Checkout the deployment branch (usually `main`).

### 3.3 Configure runtime inputs

Choose one input source:

1. Unity Catalog table
   - Set env var: `NORM_DATA_TABLE=<catalog>.<schema>.<table>`

2. Unity Catalog Volume or DBFS path
   - Set env var: `NORM_DATA_PATH=/Volumes/<catalog>/<schema>/<volume>/LM_CS_slim.csv`

Do not set both unless you intentionally want table to win (the code checks table first).

### 3.4 Create the app in Azure Databricks

1. Go to **Compute > Apps** (or **Workspace > Apps**, depending on UI version).
2. Create app from repo path: `apps/lm_cost`.
3. App config file: `apps/lm_cost/app.yaml`.
4. Add environment variables:
   - `NORM_DATA_TABLE` or `NORM_DATA_PATH`
5. Deploy and start.

### 3.5 Validate

1. Open app URL.
2. Confirm page title shows "Last Mile Cost Outlier Detection".
3. Check key visuals/tables render.
4. Verify no errors in app logs.

### 3.6 Rollback

1. Re-deploy previous git commit/branch in the app.
2. Revert env var changes.

---

## 4. Option 2 - Deploy to Azure App Service (Private, FTL-style)

Use this when your security model needs private endpoint-only access from corporate network.

### 4.1 Azure resources

- Subscription ID: `912590af-f1f7-4844-9c9b-75a04f4fd0b7`
- Resource Group: `rg-vnet-eastus-mt-sco-prod-gen2`
- Region: `eastus`
- ACR: `ftlhubacr`
- App Service Plan: `lm-cost-plan` (Linux Premium V3, P1v3)
- Web App (Linux container): `lm-cost`
- Private endpoint name: `lm-cost-pe`
- VNet: `vnet-eastus-mt-sco-prod`
- Inbound subnet: `common`
- Private DNS zone: `privatelink.azurewebsites.net`

These defaults intentionally mirror the FTL deployment baseline so networking and policy behavior are predictable.

### 4.2 Add Dockerfile under `apps/lm_cost`

Use this Dockerfile:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["sh", "-c", "streamlit run lm_cost_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
```

### 4.3 Build image in ACR

From directory `apps/lm_cost`:

```bash
az account set --subscription 912590af-f1f7-4844-9c9b-75a04f4fd0b7
az acr build --registry ftlhubacr --image lm-cost:latest .
```

### 4.4 Create web app (follow policy-compliant path)

If policies block CLI create, use Portal wizard like FTL and set at create-time:

- HTTPS only = On
- Public network access = Disabled
- Container image = `ftlhubacr.azurecr.io/lm-cost:latest`
- Container port = `8501`
- Private endpoint = Enabled

### 4.5 Set app settings

Set one of:

- `NORM_DATA_TABLE=<catalog>.<schema>.<table>`
- `NORM_DATA_PATH=<csv-path>`

Optional Streamlit proxy hardening if needed:

- `STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION=false`
- `STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false`
- `STREAMLIT_SERVER_ENABLE_CORS=false`

### 4.6 Restart and validate

```bash
az webapp restart -g rg-vnet-eastus-mt-sco-prod-gen2 -n lm-cost
az webapp log tail -g rg-vnet-eastus-mt-sco-prod-gen2 -n lm-cost
```

Validate app load and interaction after restart.

---

## 5. Data and Access Checklist

Before go-live, verify:

- Identity used by app can read table/path.
- Any secrets are in Azure Key Vault or Databricks secrets, not in code.
- Data freshness process is defined (manual or scheduled).
- Monitoring/alerting is enabled for app failures.

---

## 6. Recommended Migration Sequence

1. Deploy to Azure Databricks Apps first (fast path).
2. Validate business results and performance.
3. If private endpoint web app is required, containerize and move to App Service.
4. Keep Databricks app as fallback until App Service is stable.

---

## 7. Quick Cutover Plan

1. Freeze changes to app logic.
2. Deploy target Azure environment.
3. Run parallel validation (old vs new) for at least one business cycle.
4. Switch users to Azure endpoint.
5. Monitor for 1-2 weeks.
6. Decommission old endpoint after sign-off.

---

## 8. One-Command Deployment Script

Script added in repo:

- `deploy_lm_cost_appservice.ps1`

Example usage:

```powershell
./deploy_lm_cost_appservice.ps1
```

With runtime data source:

```powershell
./deploy_lm_cost_appservice.ps1 -NormDataTable mycatalog.myschema.mytable
# or
./deploy_lm_cost_appservice.ps1 -NormDataPath /Volumes/catalog/schema/volume/LM_CS_slim.csv
```

Notes:

- The script builds and pushes `lm-cost:latest` to ACR.
- It creates App Service plan/web app if missing.
- If policy blocks CLI create because public access must be disabled at creation, create the web app once in Portal with private networking and re-run the script to continue updates.
