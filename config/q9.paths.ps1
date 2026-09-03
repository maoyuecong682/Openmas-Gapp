# OpenMAS-Gapp Q9 local path setup.
# Usage from PowerShell:
#   . D:\Openmas-Gapp\config\q9.paths.ps1

$env:OPENMAS_ROOT = "D:\Openmas-Gapp"
$env:OPENMAS_CODE_ROOT = "$env:OPENMAS_ROOT\OpenMAS-Gapp-framework-release-20260902-v3"

# Dataset root passed to GraphHarnessEngine / openmas-bench CLI.
# DatasetAdapter.source_file values are relative to this root, e.g.
# q2_datasets/normalized/medqa.jsonl and q2_datasets/normalized/pubmedqa.jsonl.
$env:OPENMAS_DATA_ROOT = "$env:OPENMAS_ROOT"
$env:OPENMAS_Q9_DATA_ROOT = "$env:OPENMAS_ROOT\q2_datasets"
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
$localSecrets = "$env:OPENMAS_ROOT\secrets\q9_api_keys.local.ps1"
if (Test-Path $localSecrets) {
    . $localSecrets
}
