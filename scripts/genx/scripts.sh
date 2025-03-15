#!/bin/bash
NUM_PROCESSES=5  # set to the number of parallel processes to use
DATA_DIR=/mnt/ssd-4tb/gen4/

DT=(5 10 20 50 100)  # Different duration values

for dt in "${DT[@]}"; do
    DEST_DIR="/home/aten-22/dataset/gen4_preprocessed/dt_${dt}"  # Dynamic output directory
    CONFIG_DURATION="conf_preprocess/extraction/duration_${dt}.yaml"  # Dynamic YAML file

    echo "Processing with dt=${dt}, saving to ${DEST_DIR}, using config ${CONFIG_DURATION}"

    python3 preprocess_dataset.py "${DATA_DIR}" "${DEST_DIR}" \
        conf_preprocess/representation/stacked_hist.yaml \
        "${CONFIG_DURATION}" \
        conf_preprocess/filter_gen4.yaml \
        -ds gen4 -np "${NUM_PROCESSES}"
done
