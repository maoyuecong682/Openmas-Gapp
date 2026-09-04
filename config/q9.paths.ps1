# OpenMAS-Gapp Q9 local path setup.
# Usage from PowerShell:
#   . D:\Openmas-Gapp\config\q9.paths.ps1

$env:OPENMAS_ROOT = "D:\Openmas-Gapp"
$env:OPENMAS_CODE_ROOT = "$env:OPENMAS_ROOT\OpenMAS-Gapp-framework-release-20260902-v3"

# Dataset root passed to GraphHarnessEngine / openmas-bench CLI.
# DatasetAdapter.source_file values are relative to this root, e.g.
# q9_datasets/normalized/medqa.jsonl and q9_datasets/normalized/pubmedqa.jsonl.
$env:OPENMAS_DATA_ROOT = "$env:OPENMAS_ROOT"
$env:OPENMAS_Q9_DATA_ROOT = "$env:OPENMAS_ROOT\q9_datasets"
$env:OPENMAS_Q9_RAW_ROOT = "$env:OPENMAS_Q9_DATA_ROOT\raw\medical"
$env:OPENMAS_Q9_NORMALIZED_ROOT = "$env:OPENMAS_Q9_DATA_ROOT\normalized"
$env:OPENMAS_Q9_MANIFEST_ROOT = "$env:OPENMAS_Q9_DATA_ROOT\manifests"

# Hugging Face / dataset download cache locations.
$env:HF_HOME = "$env:OPENMAS_ROOT\.cache\huggingface"
$env:HF_HUB_CACHE = "$env:HF_HOME\hub"
$env:HUGGINGFACE_HUB_CACHE = "$env:HF_HUB_CACHE"
$env:HF_DATASETS_CACHE = "$env:HF_HOME\datasets"
$env:HF_ASSETS_CACHE = "$env:HF_HOME\assets"
$env:PIP_CACHE_DIR = "$env:OPENMAS_ROOT\.cache\pip"

# Network proxy for remote datasets and model endpoints.
# The inherited default proxy can point at an unused local port; use the
# currently reachable local proxy port for this workspace.
$env:Q2_HTTP_PROXY = "http://127.0.0.1:7897"
$env:HTTP_PROXY = $env:Q2_HTTP_PROXY
$env:HTTPS_PROXY = $env:Q2_HTTP_PROXY
$env:ALL_PROXY = $env:Q2_HTTP_PROXY

# Experiment artifacts.
$env:OPENMAS_Q9_OUTPUT_ROOT = "$env:OPENMAS_ROOT\outputs\q9_medical"
$env:OPENMAS_Q9_RUN_ROOT = "$env:OPENMAS_Q9_OUTPUT_ROOT\runs"
$env:OPENMAS_Q9_TABLE_ROOT = "$env:OPENMAS_Q9_OUTPUT_ROOT\tables"
$env:OPENMAS_Q9_TRACE_ROOT = "$env:OPENMAS_Q9_OUTPUT_ROOT\traces"
$env:OPENMAS_Q9_AUDIT_ROOT = "$env:OPENMAS_Q9_OUTPUT_ROOT\audit"
$env:OPENMAS_Q9_LOG_ROOT = "$env:OPENMAS_ROOT\logs\q9_medical"
$env:OPENMAS_Q9_TMP_ROOT = "$env:OPENMAS_ROOT\tmp\q9_medical"

# Make source-tree imports work without installing the package.
$env:PYTHONPATH = "$env:OPENMAS_CODE_ROOT"

# Optional local secrets file. Keep actual keys out of committed docs.
#
# Load simple lines in the form:
#   $env:NAME = "value"
#   $env:NAME = value
#
# Parsing the file instead of dot-sourcing it avoids accidentally echoing a
# malformed secret as a PowerShell command.
$localSecrets = "$env:OPENMAS_ROOT\secrets\q9_api_keys.local.ps1"
if (Test-Path $localSecrets) {
    foreach ($line in Get-Content -LiteralPath $localSecrets) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) {
            continue
        }
        if ($trimmed -match '^\$env:([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') {
            $secretName = $Matches[1]
            $secretValue = $Matches[2].Trim()
            while ($secretValue.Length -gt 0 -and
                   ($secretValue.StartsWith('"') -or $secretValue.StartsWith("'"))) {
                $secretValue = $secretValue.Substring(1)
            }
            while ($secretValue.Length -gt 0 -and
                   ($secretValue.EndsWith('"') -or $secretValue.EndsWith("'"))) {
                $secretValue = $secretValue.Substring(0, $secretValue.Length - 1)
            }
            [Environment]::SetEnvironmentVariable($secretName, $secretValue, "Process")
        }
    }
}

# Keep the benchmark-facing key variables aligned for OpenAI-compatible
# providers.  Some local files were created before the provider was switched
# from DashScope to DeepSeek, so a DeepSeek-shaped key may still be stored
# under the old variable name.
$deepSeekKey = [Environment]::GetEnvironmentVariable("DEEPSEEK_API_KEY", "Process")
$openMasKey = [Environment]::GetEnvironmentVariable("OPENMAS_LLM_API_KEY", "Process")
$q1Key = [Environment]::GetEnvironmentVariable("Q1_LLM_API_KEY", "Process")
$legacyDashScopeKey = [Environment]::GetEnvironmentVariable("DASHSCOPE_API_KEY", "Process")
if (-not $openMasKey) {
    if ($deepSeekKey) {
        [Environment]::SetEnvironmentVariable("OPENMAS_LLM_API_KEY", $deepSeekKey, "Process")
    } elseif ($legacyDashScopeKey -and -not $legacyDashScopeKey.StartsWith("sk-ws-")) {
        [Environment]::SetEnvironmentVariable("OPENMAS_LLM_API_KEY", $legacyDashScopeKey, "Process")
    }
}
if (-not $q1Key) {
    $openMasKey = [Environment]::GetEnvironmentVariable("OPENMAS_LLM_API_KEY", "Process")
    if ($openMasKey) {
        [Environment]::SetEnvironmentVariable("Q1_LLM_API_KEY", $openMasKey, "Process")
    }
}
