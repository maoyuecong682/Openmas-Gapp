# OpenMAS-Gapp Q10 local path setup.
# Usage: . D:\Openmas-Gapp\config\q10.paths.ps1

$env:OPENMAS_ROOT = "D:\Openmas-Gapp"
$env:OPENMAS_CODE_ROOT = "$env:OPENMAS_ROOT\OpenMAS-Gapp-framework-release-20260902-v3"
$env:OPENMAS_DATA_ROOT = "$env:OPENMAS_ROOT"
$env:OPENMAS_Q10_DATA_ROOT = "$env:OPENMAS_ROOT\q10_datasets"
$env:OPENMAS_Q10_RAW_ROOT = "$env:OPENMAS_Q10_DATA_ROOT\raw\financial"
$env:OPENMAS_Q10_NORMALIZED_ROOT = "$env:OPENMAS_Q10_DATA_ROOT\normalized"
$env:OPENMAS_Q10_PILOT_ROOT = "$env:OPENMAS_Q10_DATA_ROOT\pilot"
$env:OPENMAS_Q10_MANIFEST_ROOT = "$env:OPENMAS_Q10_DATA_ROOT\manifests"

$env:HF_HOME = "$env:OPENMAS_ROOT\.cache\huggingface"
$env:HF_HUB_CACHE = "$env:HF_HOME\hub"
$env:HUGGINGFACE_HUB_CACHE = "$env:HF_HUB_CACHE"
$env:HF_DATASETS_CACHE = "$env:HF_HOME\datasets"
$env:HF_ASSETS_CACHE = "$env:HF_HOME\assets"
$env:PIP_CACHE_DIR = "$env:OPENMAS_ROOT\.cache\pip"

# Keep network settings aligned with the currently reachable local proxy.
$env:Q10_HTTP_PROXY = "http://127.0.0.1:7897"
$env:HTTP_PROXY = $env:Q10_HTTP_PROXY
$env:HTTPS_PROXY = $env:Q10_HTTP_PROXY
$env:ALL_PROXY = $env:Q10_HTTP_PROXY
$env:Q10_HF_DATASETS_ENDPOINT = "https://datasets-server.huggingface.co"

$env:OPENMAS_Q10_OUTPUT_ROOT = "$env:OPENMAS_ROOT\outputs\q10_financial"
$env:OPENMAS_Q10_RUN_ROOT = "$env:OPENMAS_Q10_OUTPUT_ROOT\runs"
$env:OPENMAS_Q10_FIGURE_ROOT = "$env:OPENMAS_Q10_OUTPUT_ROOT\figures"
$env:OPENMAS_Q10_TABLE_ROOT = "$env:OPENMAS_Q10_OUTPUT_ROOT\tables"
$env:OPENMAS_Q10_TRACE_ROOT = "$env:OPENMAS_Q10_OUTPUT_ROOT\traces"
$env:OPENMAS_Q10_AUDIT_ROOT = "$env:OPENMAS_Q10_OUTPUT_ROOT\audit"
$env:OPENMAS_Q10_LOG_ROOT = "$env:OPENMAS_ROOT\logs\q10_financial"
$env:OPENMAS_Q10_TMP_ROOT = "$env:OPENMAS_ROOT\tmp\q10_financial"
$env:PYTHONPATH = "$env:OPENMAS_CODE_ROOT"

# Load local secrets without printing them or placing them in source control.
$localSecrets = "$env:OPENMAS_ROOT\secrets\q9_api_keys.local.ps1"
if (Test-Path $localSecrets) {
    foreach ($line in Get-Content -LiteralPath $localSecrets) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) { continue }
        if ($trimmed -match '^\$env:([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') {
            $secretName = $Matches[1]
            $secretValue = $Matches[2].Trim()
            $secretValue = $secretValue.Trim('"', "'")
            [Environment]::SetEnvironmentVariable($secretName, $secretValue, "Process")
        }
    }
}

# Q10 uses the same provider-facing key contract as Q9.
$openMasKey = [Environment]::GetEnvironmentVariable("OPENMAS_LLM_API_KEY", "Process")
if (-not $openMasKey) {
    $deepSeekKey = [Environment]::GetEnvironmentVariable("DEEPSEEK_API_KEY", "Process")
    if ($deepSeekKey) {
        [Environment]::SetEnvironmentVariable("OPENMAS_LLM_API_KEY", $deepSeekKey, "Process")
    }
}
