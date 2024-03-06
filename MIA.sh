#!/bin/bash

SESSION="MIA"
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

    # Load the module
    tmux send-keys -t $SESSION:$index "module load $MODULE_NAME" Enter

    # Execute the command with the running index substituted
    command="python ./synthetic_data_release/linkage_cli.py -D ./synthetic_data_release/data/real_support2_small -RC ./synthetic_data_release/tests/linkage/runconfig_totcst_outliers_trunc${index}.json -O ./output/linkage_realsupport2_small_totcstOutlier_trunc${index}"
    tmux send-keys -t $SESSION:$index "$command" Enter
done

# Attach to the tmux session
tmux attach-session -t $SESSION