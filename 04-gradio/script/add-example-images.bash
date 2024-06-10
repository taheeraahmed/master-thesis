#!/bin/bash

# Source directory containing the images
SOURCE_DIR="/cluster/home/taheeraa/datasets/chestxray-14/images"

# Target directory to copy the images to
TARGET_DIR="/cluster/home/taheeraa/code/master-thesis/04-gradio/example_images"

# Array of image filenames to find and copy
declare -a IMAGES=("00010575_002.png" "00010828_039.png" "00011925_072.png"
                   "00018253_059.png" "00020482_032.png" "00026221_001.png")

# Loop over the array of image filenames
for img in "${IMAGES[@]}"; do
    # Check if the file exists in the source directory
    if [ -f "${SOURCE_DIR}/${img}" ]; then
        # Copy the image to the target directory
        cp "${SOURCE_DIR}/${img}" "${TARGET_DIR}/"
        echo "Copied ${img} to ${TARGET_DIR}"
    else
        echo "Image not found: ${img}"
    fi
done

echo "Operation completed."
