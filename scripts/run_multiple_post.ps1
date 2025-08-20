<#
Script to run PTSD ML pipeline multiple times and collect statistics (PowerShell version)

Usage:
  - Positional:  .\run_multiple_experiments.ps1 5
  - Named:       .\run_multiple_experiments.ps1 -NumRuns 5

This mirrors the behavior of run_multiple_experiments.sh.
#>

param(
    [int]$NumRuns
)

if (-not $PSBoundParameters.ContainsKey('NumRuns') -and $args.Count -ge 1) {
    $NumRuns = [int]$args[0]
}
if (-not $NumRuns) { $NumRuns = 5 }

# File names
$PYTHON_SCRIPT = "ml_pipelines_post.py"
$RESULTS_DIR = "experiment_results_post"
$SUMMARY_FILE = "experiment_summary_post.txt"

# Choose python executable
$pythonCmd = if (Get-Command python -ErrorAction SilentlyContinue) { 'python' } elseif (Get-Command py -ErrorAction SilentlyContinue) { 'py' } else { 'python' }

# Create results directory
New-Item -ItemType Directory -Force -Path $RESULTS_DIR | Out-Null

# Arrays to store metrics
$aucValues = @()
$recallClass0Values = @()
$recallClass1Values = @()
$bestModels = @()

# Arrays to store timing information
$runTimes = @()

function Format-Time([int]$seconds) {
    $ts = [TimeSpan]::FromSeconds($seconds)
    if ($ts.Hours -gt 0) {
        return ("{0}h {1}m {2}s" -f $ts.Hours, $ts.Minutes, $ts.Seconds)
    } elseif ($ts.Minutes -gt 0) {
        return ("{0}m {1}s" -f $ts.Minutes, $ts.Seconds)
    } else {
        return ("{0}s" -f $ts.Seconds)
    }
}

function Get-Stats([double[]]$values) {
    if (-not $values -or $values.Count -eq 0) { return $null }
    $count = $values.Count
    $sum = ($values | Measure-Object -Sum).Sum
    $mean = $sum / $count
    $sumsq = ($values | ForEach-Object { $_ * $_ } | Measure-Object -Sum).Sum
    $var = $sumsq / $count - $mean * $mean
    if ($var -lt 0) { $var = 0 }
    $std = [Math]::Sqrt($var)
    $min = ($values | Measure-Object -Minimum).Minimum
    $max = ($values | Measure-Object -Maximum).Maximum
    return [pscustomobject]@{ Mean=$mean; Std=$std; Min=$min; Max=$max }
}

function Add-ContentRetry([string]$Path, [string]$Value, [int]$Retries = 10, [int]$DelayMs = 100) {
    for ($attempt = 1; $attempt -le $Retries; $attempt++) {
        try {
            Add-Content -Path $Path -Value $Value
            return
        } catch {
            if ($attempt -eq $Retries) { throw }
            Start-Sleep -Milliseconds $DelayMs
        }
    }
}

Write-Host "=================================="
Write-Host "Running PTSD ML Pipeline Experiments"
Write-Host ("Number of runs: {0}" -f $NumRuns)
Write-Host "=================================="

# Clear previous summary
Set-Content -Path $SUMMARY_FILE -Value "" -Encoding UTF8

# Run experiments
$experimentStart = Get-Date

for ($i = 1; $i -le $NumRuns; $i++) {
    Write-Host ""
    $progressPercent = [int](($i - 1) * 100 / $NumRuns)
    Write-Host ("--- Running Experiment {0}/{1} [{2}%] ---" -f $i, $NumRuns, $progressPercent)

    if ($i -gt 1) {
        $totalElapsed = [int]((Get-Date) - $experimentStart).TotalSeconds
        $avgTimePerRun = [int]($totalElapsed / ($i - 1))
        $remainingRuns = $NumRuns - $i + 1
        $etaSeconds = $avgTimePerRun * $remainingRuns
        $finishAround = (Get-Date).AddSeconds($etaSeconds).ToString('HH:mm')
        Write-Host ("Progress: [{0}%] | Avg time/run: {1}" -f $progressPercent, (Format-Time $avgTimePerRun))
        Write-Host ("ETA: {0} (finishing around {1})" -f (Format-Time $etaSeconds), $finishAround)
    }

    $startMsg = ("{0}: Starting run {1}" -f (Get-Date), $i)
    Write-Host $startMsg
    Add-ContentRetry -Path $SUMMARY_FILE -Value $startMsg

    # Output files for this run
    $outputFile = Join-Path $RESULTS_DIR ("run_{0}_output.txt" -f $i)
    $finalOutput = Join-Path $RESULTS_DIR ("run_{0}_final.txt" -f $i)

    # Record start time for this run
    $runStart = Get-Date

    # Run the Python script and capture output
    Write-Host ("Executing: {0} {1}" -f $pythonCmd, $PYTHON_SCRIPT)
    if ($pythonCmd -eq 'py') {
        & py -3 $PYTHON_SCRIPT *> $outputFile
    } else {
        & $pythonCmd $PYTHON_SCRIPT *> $outputFile
    }
    $exitCode = $LASTEXITCODE

    # Timing
    $runEnd = Get-Date
    $runDuration = [int]($runEnd - $runStart).TotalSeconds
    $runTimes += $runDuration

    if ($exitCode -eq 0) {
        Write-Host ("Run {0} completed successfully in {1}" -f $i, (Format-Time $runDuration))

        # Extract metrics from the last 100 lines
        (Get-Content -Path $outputFile -Tail 100) | Set-Content -Path $finalOutput

        # Extract values via regex
        $bestModelMatch = Select-String -Path $finalOutput -Pattern 'Best Model \(by Test AUC\): (.+)$' -SimpleMatch:$false | Select-Object -First 1
        $aucMatch = Select-String -Path $finalOutput -Pattern 'Best Test AUC: ([0-9.]+)' -SimpleMatch:$false | Select-Object -First 1
        $recall0Match = Select-String -Path $finalOutput -Pattern 'Best Test Recall \(Class 0\): ([0-9.]+)' -SimpleMatch:$false | Select-Object -First 1
        $recall1Match = Select-String -Path $finalOutput -Pattern 'Best Test Recall \(Class 1\): ([0-9.]+)' -SimpleMatch:$false | Select-Object -First 1

        $bestModel = if ($bestModelMatch) { $bestModelMatch.Matches[0].Groups[1].Value.Trim() } else { $null }
        $auc = if ($aucMatch) { $aucMatch.Matches[0].Groups[1].Value.Trim() } else { $null }
        $recall0 = if ($recall0Match) { $recall0Match.Matches[0].Groups[1].Value.Trim() } else { $null }
        $recall1 = if ($recall1Match) { $recall1Match.Matches[0].Groups[1].Value.Trim() } else { $null }

        if ($bestModel -and $auc -and $recall0 -and $recall1) {
            $aucValues += $auc
            $recallClass0Values += $recall0
            $recallClass1Values += $recall1
            $bestModels += $bestModel

            Write-Host ("  Best Model: {0}" -f $bestModel)
            Write-Host ("  AUC ROC: {0}" -f $auc)
            Write-Host ("  Recall (Class 0): {0}" -f $recall0)
            Write-Host ("  Recall (Class 1): {0}" -f $recall1)

            Add-ContentRetry -Path $SUMMARY_FILE -Value ("Run {0}: Model={1}, AUC={2}, Recall0={3}, Recall1={4}" -f $i, $bestModel, $auc, $recall0, $recall1)
        } else {
            Write-Host ("Warning: Could not extract all metrics from run {0}" -f $i)
            Write-Host ("  AUC: '{0}', Recall0: '{1}', Recall1: '{2}', Model: '{3}'" -f $auc, $recall0, $recall1, $bestModel)
            Add-ContentRetry -Path $SUMMARY_FILE -Value ("Run {0}: FAILED to extract metrics" -f $i)
        }
    } else {
        Write-Host ("Run {0} failed with exit code {1} after {2}" -f $i, $exitCode, (Format-Time $runDuration))
        Add-ContentRetry -Path $SUMMARY_FILE -Value ("Run {0}: FAILED with exit code {1}" -f $i, $exitCode)
    }

    $endMsg = ("{0}: Completed run {1}" -f (Get-Date), $i)
    Write-Host $endMsg
    Add-ContentRetry -Path $SUMMARY_FILE -Value $endMsg
}

# Calculate total experiment time
$totalExperimentSeconds = [int]((Get-Date) - $experimentStart).TotalSeconds

Write-Host ""
Write-Host "=================================="
Write-Host "EXPERIMENT SUMMARY"
Write-Host "=================================="
Write-Host ("Total experiment time: {0}" -f (Format-Time $totalExperimentSeconds))

if ($aucValues.Count -gt 0) {
    Write-Host ("Successful runs: {0}/{1}" -f $aucValues.Count, $NumRuns)
    Write-Host ""

    $toDouble = { param($s) [double]::Parse($s, [System.Globalization.CultureInfo]::InvariantCulture) }
    $aucValuesD = @($aucValues | ForEach-Object { & $toDouble $_ })
    $recall0D = @($recallClass0Values | ForEach-Object { & $toDouble $_ })
    $recall1D = @($recallClass1Values | ForEach-Object { & $toDouble $_ })
    $runTimesD = @($runTimes | ForEach-Object { [double]$_ })

    $aucStats = Get-Stats $aucValuesD
    $recall0Stats = Get-Stats $recall0D
    $recall1Stats = Get-Stats $recall1D
    $timeStats = Get-Stats $runTimesD

    Write-Host "AUC ROC Statistics:"
    Write-Host ("  Mean: {0:F4}" -f $aucStats.Mean)
    Write-Host ("  Std:  {0:F4}" -f $aucStats.Std)
    Write-Host ("  Min:  {0:F4}" -f $aucStats.Min)
    Write-Host ("  Max:  {0:F4}" -f $aucStats.Max)

    Write-Host ""
    Write-Host "Recall (Class 0) Statistics:"
    Write-Host ("  Mean: {0:F4}" -f $recall0Stats.Mean)
    Write-Host ("  Std:  {0:F4}" -f $recall0Stats.Std)
    Write-Host ("  Min:  {0:F4}" -f $recall0Stats.Min)
    Write-Host ("  Max:  {0:F4}" -f $recall0Stats.Max)

    Write-Host ""
    Write-Host "Recall (Class 1) Statistics:"
    Write-Host ("  Mean: {0:F4}" -f $recall1Stats.Mean)
    Write-Host ("  Std:  {0:F4}" -f $recall1Stats.Std)
    Write-Host ("  Min:  {0:F4}" -f $recall1Stats.Min)
    Write-Host ("  Max:  {0:F4}" -f $recall1Stats.Max)

    Write-Host ""
    Write-Host ""
    Write-Host "Timing Statistics:"
    Write-Host ("  Average run time: {0:F0} seconds" -f $timeStats.Mean)
    Write-Host ("  Std deviation: {0:F0} seconds" -f $timeStats.Std)
    Write-Host ("  Fastest run: {0:F0} seconds" -f $timeStats.Min)
    Write-Host ("  Slowest run: {0:F0} seconds" -f $timeStats.Max)

    Write-Host ""
    Write-Host "Best Models Summary:"
    $bestModels | Group-Object | Sort-Object Count -Descending | ForEach-Object {
        "{0,3} {1}" -f $_.Count, $_.Name
    } | Write-Host

    # Save detailed results to summary file
    Add-ContentRetry -Path $SUMMARY_FILE -Value ""
    Add-ContentRetry -Path $SUMMARY_FILE -Value "=== FINAL STATISTICS ==="
    Add-ContentRetry -Path $SUMMARY_FILE -Value ("Total experiment time: {0}" -f (Format-Time $totalExperimentSeconds))
    Add-ContentRetry -Path $SUMMARY_FILE -Value ("Successful runs: {0}/{1}" -f $aucValues.Count, $NumRuns)
    Add-ContentRetry -Path $SUMMARY_FILE -Value ("AUC values: {0}" -f ($aucValues -join ' '))
    Add-ContentRetry -Path $SUMMARY_FILE -Value ("Recall (Class 0) values: {0}" -f ($recallClass0Values -join ' '))
    Add-ContentRetry -Path $SUMMARY_FILE -Value ("Recall (Class 1) values: {0}" -f ($recallClass1Values -join ' '))
    Add-ContentRetry -Path $SUMMARY_FILE -Value ("Run times (seconds): {0}" -f ($runTimes -join ' '))
    Add-ContentRetry -Path $SUMMARY_FILE -Value ("Best models: {0}" -f ($bestModels -join ' '))
} else {
    Write-Host "No successful runs to analyze!"
}

Write-Host ""
Write-Host ("All outputs saved in: {0}/" -f $RESULTS_DIR)
Write-Host ("Summary saved in: {0}" -f $SUMMARY_FILE)
Write-Host "=================================="


