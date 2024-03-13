#!/bin/bash

SESSION="MIA"
CONDA_ENV="test2"

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
    command="python ./synthetic_data_release/linkage_cli.py -D ./data/real_support2_small -RC ./tests/linkage/runconfig_totcst_outliers_trunc${index}.json -O ./output/linkage_realsupport2_small_totcstOutlier_trunc${index}"
    
    # Run the command in the pane and wait for it to finish
    tmux send-keys -t $SESSION:$index "$command; tmux wait -S Window${index}_done" Enter

done

# Wait for all tmux windows to finish
for index in "${window_indices[@]}"; do
    tmux wait "Window${index}_done"
done

# Kill tmux session
tmux kill-session -t $SESSION

# Exit with success code
echo "MIA finished"
exit 0