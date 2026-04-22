# Run iterations until Monday 12:00 PM
$targetTime = Get-Date -Year 2026 -Month 4 -Day 20 -Hour 12 -Minute 0 -Second 0

Write-Output "Training loop started. Target end time (Monday): $($targetTime.ToString('yyyy-MM-dd HH:mm:ss'))"

# Resume from the last observed index (136)
$i = 136
while ((Get-Date) -lt $targetTime) {
    Write-Output "--- ITERATION $i (UTC $($([DateTime]::UtcNow).ToString('HH:mm'))) ---"
    uv run python autoresearch/train.py
    if ($LASTEXITCODE -ne 0) {
        Write-Output "Training failed at iteration $i"
        # Wait a bit longer on failure before retry (e.g. 5 mins)
        Start-Sleep -Seconds 300
        continue
    }
    $i++
    # 60s cooldown
    Start-Sleep -Seconds 60
}

Write-Output "Target time reached (Monday 12 PM) or training finished."
