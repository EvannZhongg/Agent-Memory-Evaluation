param(
    [string]$Config = "configs/experiments/longmemeval_oracle_no_memory.yaml",
    [int]$Limit = 1,
    [switch]$DryRun
)

$argsList = @("-m", "agent_memory_eval", "run", $Config, "--limit", "$Limit")
if ($DryRun) {
    $argsList += "--dry-run"
}

python @argsList

