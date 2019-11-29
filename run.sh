#!/bin/bash

export PYTHONPATH=$PWD:$PYTHONPATH

MODEL_DIR="./models"
DATA_DIR="./training_data"

UNTRAINED_MODEL="${MODEL_DIR}/untrained.h5"
TRAINED_MODEL="${MODEL_DIR}/trained.h5"
GRAPHDEF_FILE="${MODEL_DIR}/frozen"
TFLITE_FILE="${MODEL_DIR}/trained_model.tflite"

mkdir -p "${MODEL_DIR}"

# Build Model.
python scripts/build_and_save.py --out "${UNTRAINED_MODEL}"

python scripts/train_model.py --model "${UNTRAINED_MODEL}" --trained_model "${TRAINED_MODEL}" --data "${DATA_DIR}"

python scripts/freeze_model.py --model "${TRAINED_MODEL}" --out "${GRAPHDEF_FILE}"

./scripts/tflite_convert.sh "${GRAPHDEF_FILE}.pb" "${TFLITE_FILE}"

