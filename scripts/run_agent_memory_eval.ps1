param(
    [string]$Suite = "configs/suites/longmemeval_smoke.yaml",
    [string[]]$Backend = @(),
    [Nullable[int]]$Limit = $null,
    [switch]$NoEval,
    [switch]$DryRun,
    [switch]$Quiet
)

$argsList = @("-m", "agent_memory_eval", "run", $Suite)
foreach ($name in $Backend) {
    $argsList += @("--backend", $name)
}
if ($null -ne $Limit) {
    $argsList += @("--limit", "$Limit")
}
if ($NoEval) {
    $argsList += "--no-eval"
}
if ($DryRun) {
    $argsList += "--dry-run"
}
if ($Quiet) {
    $argsList += "--quiet"
}

python @argsList
