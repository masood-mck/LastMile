# FTL Hub — Azure Private Web App Deployment

Deployment runbook for the Streamlit FTL forecasting dashboard (`app_dockops_7.py`) as a
**fully private** (no public internet access) containerized Linux Web App on Azure App Service.
The app is reachable **only** from the McKesson corporate network via a Private Endpoint.

---

## 1. Goal & Architecture

- **App:** Streamlit dashboard (`app_dockops_7.py`) packaged as a Linux container.
- **Registry:** Azure Container Registry (ACR) holds the built image.
- **Compute:** Azure App Service (Linux, container) on a Premium V3 plan.
- **Privacy:** Public network access **Disabled**; inbound only via a **Private Endpoint**
  into a corporate VNet, resolved by an **Azure Private DNS Zone**.

```
Developer machine ──> ACR (ftlhubacr) ──image──> App Service (ftl-hub)
                                                      │
                                       Private Endpoint (ftl-hub-pe)
                                                      │
                                     VNet: vnet-eastus-mt-sco-prod
                                       subnet: common (10.169.115.0/27)
                                                      │
                              Azure Private DNS Zone (privatelink.azurewebsites.net)
                                                      │
                                   McKesson corporate network users
```

---

## 2. Environment / Identifiers

| Item | Value |
|---|---|
| Subscription ID | `912590af-f1f7-4844-9c9b-75a04f4fd0b7` |
| Tenant (McKesson) | `da67ef1b-ca59-4db2-9a8c-aa8d94617a16` |
| Resource Group | `rg-vnet-eastus-mt-sco-prod-gen2` |
| Region | East US |
| ACR name | `ftlhubacr` (login server `ftlhubacr.azurecr.io`, SKU Basic, admin-enabled) |
| Image:Tag | `ftlhubacr.azurecr.io/ftl-dashboard:latest` |
| Image digest | `sha256:92422abac891488b4af9f0a03574deada9a0b75dda1b967224aad638d81da44d` |
| App Service Plan | `ftl-hub-plan` (Linux, SKU **P1v3 / Premium V3**, East US) |
| Web App name | `ftl-hub` |
| **App URL (Default domain)** | `https://ftl-hub-f0fjenb4g6eue7ab.eastus-01.azurewebsites.net` |
| Container port | 8501 |
| VNet | `vnet-eastus-mt-sco-prod` (`10.169.115.0/26`) |
| Private endpoint | `ftl-hub-pe` in subnet `common` (`10.169.115.0/27`) |
| Private endpoint IP | `10.169.115.7` |
| Private DNS | Azure Private DNS Zone (`privatelink.azurewebsites.net`) |
| Cloud Shell user | `sgt4gul` (Bash) |

---

## 3. Application Container

### Dockerfile
- Base image: `python:3.12-slim`
- `EXPOSE 8501`
- Startup: `CMD streamlit run app_dockops_7.py --server.port=${PORT} --server.address=0.0.0.0`
- Streamlit binds to `${PORT}`. App Service injects `PORT` from `WEBSITES_PORT`.
  Because the Portal Container tab **Port = 8501** was set, `WEBSITES_PORT=8501` is
  configured automatically. **Do NOT** also set `WEBSITES_PORT` manually.

### .dockerignore
Excludes: `.git`, `.venv`, `__pycache__`, `notebook`, `*.md`, `data/temp`,
`data/rootcause`, `.vscode`.
> Note: `.dockerignore` is honored by `az acr build` (it uploads the build context and
> applies the ignore rules). It is **not** applied by `Compress-Archive`.

---

## 4. Build Context Packaging (local, PowerShell)

A curated ~13.3 MB zip was created from the project root (excluding heavy folders like
`.venv`, `.git`, `__pycache__`). Earlier a full-repo zip exceeded the 100 MB Cloud Shell
upload limit — the curated zip below is the correct artifact.

```powershell
$items = @('Dockerfile','.dockerignore','ftl_dashboard.py','requirements.txt','scripts','.streamlit','data')
$zip = Join-Path $env:USERPROFILE 'ftl-hub-deploy.zip'
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path $items -DestinationPath $zip -Force
"{0:N1} MB -> {1}" -f ((Get-Item $zip).Length/1MB), $zip
```

- **Output artifact:** `C:\Users\mghasemi\ftl-hub-deploy.zip` (~13.3 MB)
- Located in the user profile folder, **not** the project folder.
- **App entry file:** `ftl_dashboard.py` (the file you run locally with
  `streamlit run ftl_dashboard.py`). The Dockerfile `COPY`/`CMD` must reference this same
  name. Older names (`app_dockops_7.py`, `ftl_dashboard_app.py`) are stale — packaging or
  copying the wrong one causes the build to fail at `COPY ... file does not exist` (see §11).
- **Locked-file gotcha:** if `Compress-Archive` fails with *"being used by another process"*
  on `data\Input\accumax_cost.csv`, close Excel / pause OneDrive (they hold an exclusive
  lock `Compress-Archive` can't share), then re-run. A `Copy-Item` of the same file still
  works because copy uses shared-read.

---

## 5. Image Build in Azure Cloud Shell

Uploaded `ftl-hub-deploy.zip` to Cloud Shell, unzipped into `~/ftl`, then built the image
directly in ACR (no local Docker needed):

```bash
# In Cloud Shell (Bash), working dir ~/ftl after unzip
az acr build \
  --registry ftlhubacr \
  --image ftl-dashboard:latest \
  .
```

- Build **"Run ID: ca1 was successful"**.
- Verified the pushed tag:

```bash
az acr repository show-tags -n ftlhubacr --repository ftl-dashboard -o table
# -> latest
```

---

## 6. App Service Plan (already created)

```bash
# Premium V3, Linux, East US
az appservice plan show \
  -g rg-vnet-eastus-mt-sco-prod-gen2 \
  -n ftl-hub-plan
# provisioningState: Succeeded, reserved: true (Linux), sku: P1v3 (PremiumV3)
```

Premium V3 is required because **Private Endpoints** and **Always On** are not available on
Free/Basic tiers.

---

## 7. Why the Web App Was Created via the Azure Portal (not CLI)

McKesson Azure Policies blocked `az webapp create`:

1. **`AppendPoliciesFieldsExist`** on `Microsoft.Web/sites/httpsOnly` (policy
   `Core.Web_Assign`) — the app must have `httpsOnly = true`. Attempted fix:
   add `--https-only true`.
2. **`RequestDisallowedByPolicy`** — policy *"14. Public Network Access (Core.PublicNet)"*
   **denies** creation unless `properties.publicNetworkAccess = Disabled` at creation time.
   `az webapp create` cannot set this in the same call.

**Resolution:** create the Web App through the **Portal wizard**, whose Networking tab sets
public access **Off** at creation — satisfying the policy and the private requirement.

---

## 8. Portal Web App Creation Wizard

### Basics tab
- **Name:** `ftl-hub`
- **Publish:** Container
- **Operating System:** Linux
- **Region:** East US
- **App Service Plan:** existing `ftl-hub-plan` (P1v3 / Premium V3)
> Pitfall caught: the wizard initially defaulted to a **new Free plan**
> (`ASP-rgvneteastusmtscoprodgen2-b910`) in **Canada Central**. Corrected back to Region =
> East US and the existing `ftl-hub-plan`.

### Container tab
- **Image Source:** Azure Container Registry
- **Registry:** `ftlhubacr`
- **Image:** `ftl-dashboard`
- **Tag:** `latest`
- **Authentication:** Admin credentials
- **Port:** `8501`
- **Startup Command:** *(blank — Dockerfile CMD is used)*

### Tags tab
- Cleared a stray `latest` value that appeared in the tag field.

### Networking tab
Configured for a private-only app:

| Setting | Value | Reason |
|---|---|---|
| Enable public access | **Off** | blocks the internet + satisfies PublicNet policy |
| Enable virtual network integration (master toggle) | **On** | unlocks the VNet selector + Private Endpoint options |
| Virtual Network | `vnet-eastus-mt-sco-prod` (`10.169.115.0/26`) | corporate VNet |
| Inbound → Enable private endpoints | **On** | inbound access only via the endpoint |
| Private endpoint name | `ftl-hub-pe` | |
| Inbound subnet | `common` (`10.169.115.0/27`) | see subnet note below |
| DNS | **Azure Private DNS Zone** | auto-manages `privatelink.azurewebsites.net` |
| Outbound → Enable VNet integration | **Off** | app only reads local CSVs; no outbound VNet needed |

**Subnet note:** The VNet `10.169.115.0/26` (`.0–.63`, 64 addresses) was **full** — its
entire space is consumed by two `/27` subnets (`common` = `.0–.31`, `wmsconnect_snet` =
`.32–.63`), so a new subnet could not be created. A Private Endpoint only consumes **one IP**
and can share an existing subnet, so `common` was selected. (`common` must not be delegated
to another service — it is not.)

### Review + Create — Summary
- Subscription: `912590af-f1f7-4844-9c9b-75a04f4fd0b7`
- Resource Group: `rg-vnet-eastus-mt-sco-prod-gen2`
- Name: `ftl-hub` · Publish: Container
- Site Container: `ftl-dashboard`
- Image:Tag: `ftlhubacr.azurecr.io/ftl-dashboard:latest`
- Server URL: `https://ftlhubacr.azurecr.io` · Port: `8501`
- Plan: `ftl-hub-plan` · Linux · East US · Premium V3 · Small · 8 GB
- Basic authentication: Disabled *(expected — legacy publishing creds only; ACR deploy is unaffected)*
- Networking: VNet `vnet-eastus-mt-sco-prod`, Private endpoint `ftl-hub-pe`, subnet `common`,
  Private DNS = Azure Private DNS Zone

---

## 9. Post-Creation Steps (DONE)

1. **Configuration → General settings:**
   - **Always On = On** ✅
   - **Web sockets** — the platform *toggle* is not shown on Linux container apps
     (Windows-only), and the Linux front end proxies WebSockets by default, so the
     initial page renders. **However**, the App Service proxy mishandles WebSocket
     per-message compression, which freezes Streamlit on the first interaction and
     returns a 503 — this required a Streamlit-level fix (see §9d).
   - Do **not** add `WEBSITES_PORT` manually (Port 8501 already sets it).
2. **Container start verified** via Log stream ✅ — Streamlit booted:
   ```
   Local URL: http://localhost:8501
   Network URL: http://169.254.129.4:8501
   External URL: http://20.246.210.161:8501   <- ignore (Streamlit's IP guess; app is private)
   ```
   CLI equivalent: `az webapp log tail -g rg-vnet-eastus-mt-sco-prod-gen2 -n ftl-hub`

---

## 9a. Correct App URL (Secure unique default hostname)

The app was created with **Secure unique default hostname = Enabled**. This means the plain
`ftl-hub.azurewebsites.net` is intentionally **NOT** bound to the app (anti-subdomain-takeover).
Hitting it returns an Azure **404 "Web Site not found"**.

**The real app URL is the Default domain (Overview page):**

```
https://ftl-hub-f0fjenb4g6eue7ab.eastus-01.azurewebsites.net
```

Use this hostname everywhere (browser, DNS, tests). `ftl-hub.azurewebsites.net` will 404.

---

## 9b. Private Endpoint Verification (VPN, no admin rights)

On the McKesson VPN, corporate DNS does **not** resolve the private hostname yet, so testing
used `curl.exe --resolve` (no admin needed) to override DNS for one request and send the
correct Host header + SNI:

```powershell
curl.exe -v --resolve ftl-hub-f0fjenb4g6eue7ab.eastus-01.azurewebsites.net:443:10.169.115.7 `
  https://ftl-hub-f0fjenb4g6eue7ab.eastus-01.azurewebsites.net
```

**Result: `HTTP/1.1 200 OK`, `Server: uvicorn`, Streamlit HTML** ✅
This proves: VPN routes to the private endpoint `10.169.115.7`, TLS is valid
(`*.azurewebsites.net` cert), and the app serves. The **only** remaining gap is corporate
DNS resolution for normal browser access.

> Note: the wrong hostname (`ftl-hub.azurewebsites.net`) returned **404** in the same test —
> that is the unique-hostname behavior in §9a, not an app fault.

---

## 9c. Make It Resolvable for All Users (network team action)

Browsers on the corporate network get `DNS_PROBE_FINISHED_NXDOMAIN` because McKesson DNS does
not forward the Azure private zone. The app is fine; only DNS forwarding is missing.

**Request to the network / cloud team:**

> Please make on-prem/VPN clients resolve the Azure private endpoint for app **`ftl-hub`**.
> Add DNS forwarding for **`privatelink.azurewebsites.net`** to Azure — via an **Azure DNS
> Private Resolver inbound endpoint** (or a DNS forwarder VM) in `vnet-eastus-mt-sco-prod`,
> targeting `168.63.129.16`. The Private DNS zone is already linked to the VNet and holds the
> A record `ftl-hub-f0fjenb4g6eue7ab.eastus-01.azurewebsites.net` → `10.169.115.7`.

After that, users simply browse to
`https://ftl-hub-f0fjenb4g6eue7ab.eastus-01.azurewebsites.net`.

**No-admin local workaround** (per-machine, temporary): not possible via the `hosts` file
without admin rights. Use the `curl.exe --resolve` command in §9b to verify connectivity;
full browser use requires the DNS forwarder above.

---

## 9d. Fix: App Freezes / 503 on Tab Switch

**Symptom:** The app loads and the default **Action Center** tab works, but switching to the
**📅 Actuals & Forecast** tab **freezes**, then returns a **503** ("bad connection").

### Step 1 (tried first, did NOT fix it): WebSocket / proxy settings

The initial theory was Azure's reverse proxy mishandling WebSocket per-message compression
plus an XSRF/CORS origin mismatch. These settings were applied but the 503 **persisted**, so
this was **not** the cause. They are still correct hardening for a proxied Streamlit app and
are left in place (`.streamlit/config.toml` + equivalent App settings):

```toml
[server]
enableWebsocketCompression = false
enableCORS = false
enableXsrfProtection = false   # safe: app is private (VNet-only)
```
```bash
az webapp config appsettings set -g rg-vnet-eastus-mt-sco-prod-gen2 -n ftl-hub --settings \
  STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION=false \
  STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
  STREAMLIT_SERVER_ENABLE_CORS=false
```

### Step 2 (actual root cause): AG Grid Enterprise crashes the container worker

**Diagnosis by elimination:**
- The initial paint works, so the WebSocket transport is fine.
- On a tab switch Streamlit reruns the whole script; the *only* thing that differs is which
  `if _active_tab == …` block runs.
- A **503 means the worker process died** — a Python exception would render an error *inside*
  the page, not a 503.
- **Action Center** and **Consolidation Signals** both render with native
  `st.dataframe` + Plotly and work → Arrow serialization is fine in the container.
- The **only** tab with a unique element is **Actuals & Forecast**, which renders an
  **AG Grid Enterprise** tree-grid via `streamlit-aggrid`
  (`AgGrid(..., enable_enterprise_modules=True, treeData=True)`).
- That custom component's native stack resolves differently in the fresh-built
  `python:3.12-slim` container than in the local venv (Python 3.14), and crashes the worker
  when the tab renders → freeze → 503. A native segfault like this **cannot** be caught with
  `try/except`.

**Fix (in `app_dockops_7.py`):** render that table with native `st.dataframe` instead of the
custom component. Controlled by a flag so AG Grid can be re-enabled later if its container
stack is verified:

```python
USE_AGGRID = False   # native st.dataframe is the stable path in the Azure container
```

When `USE_AGGRID = False` (default) the "Actuals & Forecast" table is drawn with
`st.dataframe` (tree path flattened into an indented "Destination / Segment" column). When
`True`, AG Grid is attempted but wrapped in `try/except` with the same native fallback.

**Apply — rebuild the image** (the code is baked in, so a rebuild is required):

```bash
az acr build --registry ftlhubacr --image ftl-dashboard:latest .
az webapp restart -g rg-vnet-eastus-mt-sco-prod-gen2 -n ftl-hub
```

### Confirm from logs

Stream logs, then switch to **Actuals & Forecast** and watch what prints:

```bash
az webapp log tail -g rg-vnet-eastus-mt-sco-prod-gen2 -n ftl-hub
```

- A native crash / worker exit with **no Python traceback** confirms the AG Grid cause
  (fixed by `USE_AGGRID = False`).
- `Killed` / `MemoryError` would mean OOM (not the case — data is tiny).

---


## 10. Notes & Lessons Learned

- On this managed machine, only browser/Portal/Cloud Shell (as `sgt4gul`) works reliably;
  local `az` CLI was not usable (`az version` exit code 1).
- McKesson policies force `httpsOnly = true` and `publicNetworkAccess = Disabled`; both are
  satisfied by the Portal private-app path.
- "Basic authentication Disabled" on the app is expected and does not affect the ACR-based
  container deployment.
- Private Endpoint requires Basic tier or higher — Premium V3 (P1v3) was chosen.
- A Private Endpoint can share an existing, non-delegated subnet (it uses a single IP);
  a dedicated empty/delegated subnet is only needed for **outbound** VNet integration
  (which is Off here).
- **Web sockets** has no platform toggle on Linux container apps (Windows-only) and Linux
  proxies them by default — but the App Service proxy breaks WebSocket **compression**, which
  freezes Streamlit on the first interaction and 503s. Fix: `enableWebsocketCompression=false`
  (plus `enableXsrfProtection=false`, `enableCORS=false`) — see §9d.
- **Secure unique default hostname** means `ftl-hub.azurewebsites.net` returns 404 — the real
  URL is the regional Default domain
  `ftl-hub-f0fjenb4g6eue7ab.eastus-01.azurewebsites.net`.
- A `404 "Web Site not found"` reaching an `uvicorn`/App Service front end = wrong hostname,
  not a routing failure. A **200** over the private IP confirms VPN + private endpoint work.
- Editing the Windows `hosts` file needs admin rights; `curl.exe --resolve host:443:IP` is a
  no-admin way to test a private endpoint with the correct Host header and TLS SNI.
- Corporate DNS must forward `privatelink.azurewebsites.net` to Azure for browser access;
  otherwise clients get `DNS_PROBE_FINISHED_NXDOMAIN` even on VPN.

---

## 11. Updating the App (Redeploy a New Image)

Use this whenever the code (`ftl_dashboard.py`, `scripts/`) or data changes. The image is
built fresh in ACR and the Web App is restarted to pull it. No local Docker needed.

### Step 1 — Package the build context (local, PowerShell, project root)

```powershell
$items = @('Dockerfile','.dockerignore','ftl_dashboard.py','requirements.txt','scripts','.streamlit','data')
$zip = Join-Path $env:USERPROFILE 'ftl-hub-deploy.zip'
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path $items -DestinationPath $zip -Force
"{0:N1} MB -> {1}" -f ((Get-Item $zip).Length/1MB), $zip
```

> If `Compress-Archive` errors with *"being used by another process"*, close Excel / pause
> OneDrive (they lock `data\Input\accumax_cost.csv`), then re-run — see §4.

### Step 2 — Upload to Cloud Shell

Open **Azure Cloud Shell (Bash)** and use its **Upload** button to send
`ftl-hub-deploy.zip`. Freshly uploaded files land in your home dir (`~`).

### Step 3 — Select the correct subscription (common failure)

Cloud Shell may default to the wrong subscription, giving
*"The resource with name 'ftlhubacr' ... could not be found"*. Point it at the ACR's
subscription first:

```bash
az account set --subscription 912590af-f1f7-4844-9c9b-75a04f4fd0b7
az account show --query "{name:name, id:id}" -o table      # expect: mt-sco-prod
az acr show -n ftlhubacr --query "{name:name, rg:resourceGroup, login:loginServer}" -o table
```

### Step 4 — Unzip into a build folder, then build (common failure)

`az acr build` needs the current dir to contain `Dockerfile`, else
*"Unable to find './Dockerfile'"*. Unzip and `cd` in first:

```bash
cd ~
rm -rf ftl && mkdir ftl
unzip -o ftl-hub-deploy.zip -d ftl
cd ftl
ls -la                      # must list Dockerfile, ftl_dashboard.py, scripts/, data/
az acr build --registry ftlhubacr --image ftl-dashboard:latest .
```

> **Run these one line at a time.** Pasting the whole block at once can leave the shell in
> `~` instead of `~/ftl` (so `az acr build` reports *"Unable to find './Dockerfile'"* even
> though the unzip succeeded). Confirm you are in the right place first:
> `cd ~/ftl && ls -la` should show `Dockerfile` at the top level before you build. If it is
> nested, run `find ~/ftl -name Dockerfile` and `cd` into that directory.

> **`COPY ... file does not exist` build failure:** the Dockerfile `COPY`/`CMD` name must
> match the packaged app file. Current app = **`ftl_dashboard.py`**. If you ever rename the
> app, update BOTH the Dockerfile (`COPY ftl_dashboard.py .` and
> `CMD streamlit run ftl_dashboard.py ...`) and the `$items` list in Step 1.

### Step 5 — Restart the Web App to pull the new `latest`

```bash
az webapp restart -g rg-vnet-eastus-mt-sco-prod-gen2 -n ftl-hub
az webapp log tail -g rg-vnet-eastus-mt-sco-prod-gen2 -n ftl-hub   # watch it boot
```

Then browse to `https://ftl-hub-f0fjenb4g6eue7ab.eastus-01.azurewebsites.net` (VPN + DNS
forwarding per §9a/§9c).

---

## 12. Data Pipeline — Local (flat files) vs Cloud (Databricks SQL)

The app reads flat files from `data/`. There are **two ways** to produce them; the
application code is identical for both (it always reads plain files).

### Inputs and their sources

| Logical input | Flat file (read by app) | SQL source (cloud only) |
|---|---|---|
| Accumax loads + cost | `data/Input/accumax_cost.csv` | `data/Input/accumax_cost.sql` |
| Picks | `data/Input/picks.xlsx` (sheet `result`) | `data/Input/picks.sql` |
| Bio processing study | `data/Input/case_bulk_pick.xlsx` | *(none — flat file only)* |

Generated by the pipeline (read by the dashboard): `data/output/LOADS_CONSOLIDATED.xlsx`
and `data/output/pred_v3_*.csv`.

### Local (laptop) — flat files as-is

No Databricks connectivity. Use the flat files already in `data/Input`, then run the
pipeline:

```powershell
python app_data_prep.py
```

### Cloud (Databricks) — refresh from SQL, then run the same pipeline

Databricks is the cloud workstation: check the git repo out into a **Databricks Repo**
(no zip needed). A live `spark` session lets the `.sql` files run natively.

```bash
# On Databricks (repo checked out, `spark` available):
git pull
python refresh_inputs_from_sql.py   # runs *.sql via spark.sql -> writes the flat files
python app_data_prep.py             # SAME pipeline as local -> data/output/*
# commit the refreshed data/ back to git
git add data && git commit -m "refresh data" && git push
```

- `refresh_inputs_from_sql.py` executes each `<name>.sql` and writes the exact flat file
  the app reads (`accumax_cost.csv`; `picks.xlsx` with a `result` sheet). Inputs without a
  `.sql` (`case_bulk_pick.xlsx`) are left untouched.
- It is **cloud-only**: run locally it exits with a message (no Spark). Nothing in
  `scripts/` or `ftl_dashboard.py` changes — only *how the flat files are produced*.

### Rebuild the image straight from git (drops the zip in §11 Steps 1–2)

Because the refreshed `data/` is committed to git, `az acr build` can build directly from
the repo — no local packaging or Cloud Shell upload:

```bash
az account set --subscription 912590af-f1f7-4844-9c9b-75a04f4fd0b7
az acr build --registry ftlhubacr --image ftl-dashboard:latest \
  https://github.com/masood-mck/FTL_forecasting_consolidation.git#main:.
az webapp restart -g rg-vnet-eastus-mt-sco-prod-gen2 -n ftl-hub
```

Replace `main` with your branch if you build from a different one. The trailing `:.`
sets the build context to the repo root (where the `Dockerfile` lives). For a **private**
repo, pass a token in the URL
(`https://<user>:<PAT>@github.com/masood-mck/FTL_forecasting_consolidation.git#main:.`)
or run `az acr build` from a machine already authenticated to the repo. The zip flow in
§11 remains valid as a fallback when building from a local snapshot.
