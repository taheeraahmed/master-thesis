#!/bin/bash

# The parent directory you want to search in
PARENT_DIR=$(pwd)/output

# Initialize the logdir_spec argument for TensorBoard
LOGDIR_SPEC=""

# Counter to name each found log directory uniquely
COUNTER=1

# Find directories containing TensorBoard log files and construct logdir_spec
find "${PARENT_DIR}" -type f -name 'events.out.tfevents.*' | while read -r FILE; do
    DIR=$(dirname "${FILE}")
    # Use the counter or a meaningful name based on your directory structure
    NAME="run${COUNTER}"
    # Append this directory to the logdir_spec argument
    LOGDIR_SPEC="${LOGDIR_SPEC}${NAME}:${DIR},"
    ((COUNTER++))
done

# Remove the last comma
LOGDIR_SPEC=${LOGDIR_SPEC%,}

# Check if LOGDIR_SPEC is not empty
if [[ -n "${LOGDIR_SPEC}" ]]; then
    # Print the TensorBoard command
    echo "tensorboard --logdir_spec=${LOGDIR_SPEC}"
else
    echo "No TensorBoard log files found in ${PARENT_DIR}."
fi
