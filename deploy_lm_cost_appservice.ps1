param(
    [string]$SubscriptionId = "912590af-f1f7-4844-9c9b-75a04f4fd0b7",
    [string]$ResourceGroup = "rg-vnet-eastus-mt-sco-prod-gen2",
    [string]$Location = "eastus",
    [string]$AcrName = "ftlhubacr",
    [string]$AppServicePlan = "lm-cost-plan",
    [string]$WebAppName = "lm-cost",
    [string]$ImageName = "lm-cost",
    [string]$ImageTag = "latest",
    [int]$ContainerPort = 8501,
    [string]$NormDataTable = "",
    [string]$NormDataPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Run-AzCli {
    param([Parameter(Mandatory = $true)][string]$Command)
    Write-Host "> $Command" -ForegroundColor Cyan
    $result = Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $Command"
    }
    return $result
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$appPath = Join-Path $repoRoot "apps\lm_cost"

if (-not (Test-Path $appPath)) {
    throw "Path not found: $appPath"
}

Write-Host "Setting Azure subscription..." -ForegroundColor Yellow
Run-AzCli "az account set --subscription $SubscriptionId" | Out-Null

Write-Host "Checking ACR..." -ForegroundColor Yellow
$acrLoginServer = Run-AzCli "az acr show -n $AcrName --query loginServer -o tsv"
if (-not $acrLoginServer) {
    throw "Unable to resolve ACR login server for $AcrName"
}

$imageRef = "$acrLoginServer/$($ImageName):$($ImageTag)"

Write-Host "Building image in ACR..." -ForegroundColor Yellow
Push-Location $appPath
try {
    Run-AzCli "az acr build --registry $AcrName --image $ImageName`:$ImageTag ." | Out-Null
}
finally {
    Pop-Location
}

Write-Host "Ensuring Linux Premium plan exists..." -ForegroundColor Yellow
$planExists = Run-AzCli "az appservice plan show -g $ResourceGroup -n $AppServicePlan --query name -o tsv 2>$null"
if (-not $planExists) {
    Run-AzCli "az appservice plan create -g $ResourceGroup -n $AppServicePlan --is-linux --sku P1v3 --location $Location" | Out-Null
}

Write-Host "Ensuring web app exists..." -ForegroundColor Yellow
$webAppExists = Run-AzCli "az webapp show -g $ResourceGroup -n $WebAppName --query name -o tsv 2>$null"
if (-not $webAppExists) {
    try {
        Run-AzCli "az webapp create -g $ResourceGroup -p $AppServicePlan -n $WebAppName --deployment-container-image-name $imageRef --https-only true" | Out-Null
    }
    catch {
        Write-Host "Web app creation via CLI failed (likely policy around publicNetworkAccess)." -ForegroundColor Red
        Write-Host "Create '$WebAppName' once in Portal as private app, then rerun this script." -ForegroundColor Red
        throw
    }
}

Write-Host "Applying container and app settings..." -ForegroundColor Yellow
Run-AzCli "az webapp config container set -g $ResourceGroup -n $WebAppName --container-image-name $imageRef" | Out-Null
Run-AzCli "az webapp config appsettings set -g $ResourceGroup -n $WebAppName --settings WEBSITES_PORT=$ContainerPort STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION=false STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false STREAMLIT_SERVER_ENABLE_CORS=false" | Out-Null

if ($NormDataTable) {
    Run-AzCli "az webapp config appsettings set -g $ResourceGroup -n $WebAppName --settings NORM_DATA_TABLE=$NormDataTable" | Out-Null
}

if ($NormDataPath) {
    Run-AzCli "az webapp config appsettings set -g $ResourceGroup -n $WebAppName --settings NORM_DATA_PATH=$NormDataPath" | Out-Null
}

Write-Host "Restarting web app..." -ForegroundColor Yellow
Run-AzCli "az webapp restart -g $ResourceGroup -n $WebAppName" | Out-Null

$defaultHost = Run-AzCli "az webapp show -g $ResourceGroup -n $WebAppName --query defaultHostName -o tsv"

Write-Host "Deployment complete." -ForegroundColor Green
Write-Host "Image: $imageRef"
Write-Host "App: https://$defaultHost"
Write-Host "Tip: if app is private-only, access may require VPN + private DNS forwarding."