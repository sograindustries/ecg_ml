#!/bin/bash

GRAPH_DEF_FILE=$1
TFLITE_FILE=$2

tflite_convert \
--graph_def_file="$GRAPH_DEF_FILE" \
--output_file="$TFLITE_FILE" \
--output_format=TFLITE \
--input_shapes=1,2560,1 \
--input_arrays='input' \
--output_arrays='output/Softmax'

tflite_convert \
--output_file="${TFLITE_FILE}.quant" \
--graph_def_file="$GRAPH_DEF_FILE" \
--inference_type=QUANTIZED_UINT8 \
--input_shapes=1,2560,1 \
--input_arrays='input' \
--output_arrays='output/Softmax' \
--mean_values=128 \
--std_dev_values=127 \
--default_ranges_min=0 \
--default_ranges_max=6 \
--output_format=TFLITE 


