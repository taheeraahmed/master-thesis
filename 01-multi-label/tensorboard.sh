#!/bin/bash

target_directory="/cluster/home/taheeraa/code/master-thesis/01-multi-label/output"
cd "$(dirname "$0")"

# Initialize
experiment_dirs=()
experiment_names=()
LOGDIR_SPEC=""

dir_count=0
for dir in "$target_directory"/*; do
    # Check if the item is a directory and starts with a number
    if [[ -d "$dir" && "$(basename "$dir")" =~ ^[0-9] ]]; then

        # Add the directory to the array
        experiment_dirs+=("$(basename "$dir")")
        ((dir_count++))

        # Extract name for run
        base_name="${dir##*/}"
        stripped=$(echo "$base_name" | sed -E 's/^[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}:[0-9]{2}:[0-9]{2}-//; s/-t[0-9]{2}:[0-9]{2}:[0-9]{2}$//')
        experiment_names+=("$stripped")

        LOGDIR_SPEC="${LOGDIR_SPEC}${dir_count}-${stripped}:${dir},"
    fi
done

# Print the directories found
echo "Opening these experiments in tensorboard: "
printf '%s\n' "${experiment_names[@]}"

# Remove the last comma
LOGDIR_SPEC=${LOGDIR_SPEC%,}

# Running tensorboard
tensorboard_command="tensorboard --logdir_spec=${LOGDIR_SPEC}"
eval "${tensorboard_command}"