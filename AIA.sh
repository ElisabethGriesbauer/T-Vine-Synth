#!/bin/bash

SESSION="AIA"
CONDA_ENV="attack_env"

# Vector of window indices
window_indices=(1 5 10 15 20 26)

# Create a new tmux session
tmux new-session -d -s $SESSION

# Loop over the vector to create windows
for index in "${window_indices[@]}"; do

    # Create a new window
    tmux new-window -t $SESSION:$index -n "Window_$index"

    # Activate conda environment
    tmux send-keys -t $SESSION:$index "conda activate $CONDA_ENV" Enter

    # Execute the command with the running index substituted
    command="python ./synthetic_data_release/inference_cli.py -D ./synthetic_data_release/data/real_support2_small -RC ./synthetic_data_release/tests/inference/runconfig_totcst_50_0126_trunc${index}.json -O ./output/inference_realsupport2_small_totcstOutlier_trunc${index} -P 0"
    tmux send-keys -t $SESSION:$index "$command" Enter
done

# Attach to the tmux session
tmux attach-session -t $SESSION