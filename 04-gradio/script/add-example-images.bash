#!/bin/bash

SOURCE_DIR="/cluster/home/taheeraa/datasets/chestxray-14/images"

TARGET_DIR="/cluster/home/taheeraa/code/master-thesis/04-gradio/example_images"

declare -a IMAGES=("00010575_002.png" "00010828_039.png" "00011925_072.png"
                   "00018253_059.png" "00020482_032.png" "00026221_001.png")

for img in "${IMAGES[@]}"; do
    if [ -f "${SOURCE_DIR}/${img}" ]; then
        cp "${SOURCE_DIR}/${img}" "${TARGET_DIR}/"
        echo "Copied ${img} to ${TARGET_DIR}"
    else
        echo "Image not found: ${img}"
    fi
done

echo "Operation completed."
