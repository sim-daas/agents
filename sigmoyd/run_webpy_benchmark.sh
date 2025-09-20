#!/bin/bash
# Run web.py multiple times and calculate average execution time

NUM_RUNS=15
SCRIPT_PATH="/home/ubuntu/githubrepos/agents/sigmoyd/web.py"

echo "Running $SCRIPT_PATH $NUM_RUNS times..."

total=0
for i in $(seq 1 $NUM_RUNS); do
    echo "Run $i:"
    # Run and capture output
    output=$(python3 "$SCRIPT_PATH" 2>&1)
    # Grep the execution time (in seconds)
    exec_time=$(echo "$output" | grep -i "Total execution time" | awk -F: '{print $2}' | awk '{print $1}')
    echo "Execution time: $exec_time seconds"
    # Add to total if found
    if [ ! -z "$exec_time" ]; then
        total=$(echo "$total + $exec_time" | bc)
    fi
    echo "----------------------"
done

# Calculate average
if [ "$total" != "0" ]; then
    avg=$(echo "scale=2; $total / $NUM_RUNS" | bc)
    echo "Average execution time over $NUM_RUNS runs: $avg seconds"
else
    echo "No execution times found. Check script output."
fi
