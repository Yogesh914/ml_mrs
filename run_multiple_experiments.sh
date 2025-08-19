#!/bin/bash

# Script to run PTSD ML pipeline multiple times and collect statistics
# Usage: ./run_multiple_experiments.sh [number_of_runs]

# Default number of runs
NUM_RUNS=${1:-5}

# File names
PYTHON_SCRIPT="ml_pipelines_latest.py"
RESULTS_DIR="experiment_results"
SUMMARY_FILE="experiment_summary.txt"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Arrays to store metrics
declare -a auc_values
declare -a recall_class0_values  
declare -a recall_class1_values
declare -a best_models

# Arrays to store timing information
declare -a run_times

echo "=================================="
echo "Running PTSD ML Pipeline Experiments"
echo "Number of runs: $NUM_RUNS"
echo "=================================="

# Clear previous summary
> "$SUMMARY_FILE"

# Function to format seconds into human readable time
format_time() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    
    if [ $hours -gt 0 ]; then
        printf "%dh %dm %ds" $hours $minutes $secs
    elif [ $minutes -gt 0 ]; then
        printf "%dm %ds" $minutes $secs
    else
        printf "%ds" $secs
    fi
}

# Run experiments
experiment_start_time=$(date +%s)

for ((i=1; i<=NUM_RUNS; i++)); do
    echo ""
    progress_percent=$(((i-1) * 100 / NUM_RUNS))
    echo "--- Running Experiment $i/$NUM_RUNS [${progress_percent}%] ---"
    
    # Calculate and display ETA
    if [ $i -gt 1 ]; then
        # Calculate average time per run from completed runs
        total_elapsed=$(($(date +%s) - experiment_start_time))
        avg_time_per_run=$((total_elapsed / (i - 1)))
        remaining_runs=$((NUM_RUNS - i + 1))
        eta_seconds=$((avg_time_per_run * remaining_runs))
        
        echo "Progress: [$progress_percent%] | Avg time/run: $(format_time $avg_time_per_run)"
        echo "ETA: $(format_time $eta_seconds) (finishing around $(date -d "+${eta_seconds} seconds" +"%H:%M"))"
    fi
    
    echo "$(date): Starting run $i" | tee -a "$SUMMARY_FILE"
    
    # Output file for this run
    output_file="$RESULTS_DIR/run_${i}_output.txt"
    
    # Record start time for this run
    run_start_time=$(date +%s)
    
    # Run the Python script and capture output
    echo "Executing: python $PYTHON_SCRIPT"
    python "$PYTHON_SCRIPT" > "$output_file" 2>&1

    # Print the PID of the Python process
    python_pid=$!
    echo "Python process PID: $python_pid"
    
    # Check if script completed successfully
    run_end_time=$(date +%s)
    run_duration=$((run_end_time - run_start_time))
    run_times+=("$run_duration")
    
    if [ $? -eq 0 ]; then
        echo "✓ Run $i completed successfully in $(format_time $run_duration)"
        
        # Extract metrics from the last 100 lines
        tail -100 "$output_file" > "${RESULTS_DIR}/run_${i}_final.txt"
        final_output="${RESULTS_DIR}/run_${i}_final.txt"
        
        # Extract best model name
        best_model=$(grep -E "Best Model \(by Test AUC\):" "$final_output" | sed -E 's/.*Best Model \(by Test AUC\): (.+)/\1/' | sed 's/[[:space:]]*$//')
        
        # Extract AUC ROC
        auc=$(grep -E "Best Test AUC:" "$final_output" | sed -E 's/.*Best Test AUC: ([0-9.]+).*/\1/')
        
        # Extract Recall Class 0
        recall_0=$(grep -E "Best Test Recall \(Class 0\):" "$final_output" | sed -E 's/.*Best Test Recall \(Class 0\): ([0-9.]+).*/\1/')
        
        # Extract Recall Class 1  
        recall_1=$(grep -E "Best Test Recall \(Class 1\):" "$final_output" | sed -E 's/.*Best Test Recall \(Class 1\): ([0-9.]+).*/\1/')
        
        # Verify we extracted all values
        if [[ -n "$auc" && -n "$recall_0" && -n "$recall_1" && -n "$best_model" ]]; then
            # Store values
            auc_values+=("$auc")
            recall_class0_values+=("$recall_0")
            recall_class1_values+=("$recall_1")
            best_models+=("$best_model")
            
            echo "  Best Model: $best_model"
            echo "  AUC ROC: $auc"
            echo "  Recall (Class 0): $recall_0"
            echo "  Recall (Class 1): $recall_1"
            
            # Log to summary
            echo "Run $i: Model=$best_model, AUC=$auc, Recall0=$recall_0, Recall1=$recall_1" >> "$SUMMARY_FILE"
        else
            echo "⚠ Warning: Could not extract all metrics from run $i"
            echo "  AUC: '$auc', Recall0: '$recall_0', Recall1: '$recall_1', Model: '$best_model'"
            echo "Run $i: FAILED to extract metrics" >> "$SUMMARY_FILE"
        fi
    else
        echo "✗ Run $i failed with exit code $? after $(format_time $run_duration)"
        echo "Run $i: FAILED with exit code $?" >> "$SUMMARY_FILE"
    fi
    
    echo "$(date): Completed run $i" | tee -a "$SUMMARY_FILE"
done

# Calculate total experiment time
experiment_end_time=$(date +%s)
total_experiment_time=$((experiment_end_time - experiment_start_time))

echo ""
echo "=================================="
echo "EXPERIMENT SUMMARY"
echo "=================================="
echo "Total experiment time: $(format_time $total_experiment_time)"

# Calculate statistics if we have data
if [ ${#auc_values[@]} -gt 0 ]; then
    # Create temporary files for calculations
    printf '%s\n' "${auc_values[@]}" > /tmp/auc_vals.txt
    printf '%s\n' "${recall_class0_values[@]}" > /tmp/recall0_vals.txt  
    printf '%s\n' "${recall_class1_values[@]}" > /tmp/recall1_vals.txt
    
    echo "Successful runs: ${#auc_values[@]}/$NUM_RUNS"
    echo ""
    
    # Calculate statistics using awk
    echo "AUC ROC Statistics:"
    awk '{sum+=$1; sumsq+=$1*$1; count++} END {
        mean=sum/count; 
        var=(sumsq/count)-(mean*mean); 
        std=sqrt(var);
        printf "  Mean: %.4f\n  Std:  %.4f\n  Min:  %.4f\n  Max:  %.4f\n", mean, std, min, max
    } {if(NR==1){min=max=$1} if($1<min){min=$1} if($1>max){max=$1}}' /tmp/auc_vals.txt
    
    echo ""
    echo "Recall (Class 0) Statistics:"  
    awk '{sum+=$1; sumsq+=$1*$1; count++} END {
        mean=sum/count;
        var=(sumsq/count)-(mean*mean);
        std=sqrt(var);
        printf "  Mean: %.4f\n  Std:  %.4f\n  Min:  %.4f\n  Max:  %.4f\n", mean, std, min, max
    } {if(NR==1){min=max=$1} if($1<min){min=$1} if($1>max){max=$1}}' /tmp/recall0_vals.txt
    
    echo ""
    echo "Recall (Class 1) Statistics:"
    awk '{sum+=$1; sumsq+=$1*$1; count++} END {
        mean=sum/count;
        var=(sumsq/count)-(mean*mean); 
        std=sqrt(var);
        printf "  Mean: %.4f\n  Std:  %.4f\n  Min:  %.4f\n  Max:  %.4f\n", mean, std, min, max
    } {if(NR==1){min=max=$1} if($1<min){min=$1} if($1>max){max=$1}}' /tmp/recall1_vals.txt
    
    echo ""
    echo ""
    echo "Timing Statistics:"
    printf '%s\n' "${run_times[@]}" > /tmp/timing_vals.txt
    awk '{sum+=$1; sumsq+=$1*$1; count++} END {
        mean=sum/count;
        var=(sumsq/count)-(mean*mean);
        std=sqrt(var);
        printf "  Average run time: %.0f seconds\n  Std deviation: %.0f seconds\n  Fastest run: %.0f seconds\n  Slowest run: %.0f seconds\n", mean, std, min, max
    } {if(NR==1){min=max=$1} if($1<min){min=$1} if($1>max){max=$1}}' /tmp/timing_vals.txt
    
    echo ""
    echo "Best Models Summary:"
    printf '%s\n' "${best_models[@]}" | sort | uniq -c | sort -nr
    
    # Save detailed results to summary file
    echo "" >> "$SUMMARY_FILE"
    echo "=== FINAL STATISTICS ===" >> "$SUMMARY_FILE"
    echo "Total experiment time: $(format_time $total_experiment_time)" >> "$SUMMARY_FILE"
    echo "Successful runs: ${#auc_values[@]}/$NUM_RUNS" >> "$SUMMARY_FILE"
    echo "AUC values: ${auc_values[*]}" >> "$SUMMARY_FILE"
    echo "Recall (Class 0) values: ${recall_class0_values[*]}" >> "$SUMMARY_FILE"
    echo "Recall (Class 1) values: ${recall_class1_values[*]}" >> "$SUMMARY_FILE"
    echo "Run times (seconds): ${run_times[*]}" >> "$SUMMARY_FILE"
    echo "Best models: ${best_models[*]}" >> "$SUMMARY_FILE"
    
    # Clean up temp files
    rm -f /tmp/auc_vals.txt /tmp/recall0_vals.txt /tmp/recall1_vals.txt /tmp/timing_vals.txt
    
else
    echo "No successful runs to analyze!"
fi

echo ""
echo "All outputs saved in: $RESULTS_DIR/"
echo "Summary saved in: $SUMMARY_FILE"
echo "==================================" 