#!/bin/bash

GRAPH_DEF_FILE=$1
TFLITE_FILE=$2

tflite_convert \
--graph_def_file="$GRAPH_DEF_FILE" \
--output_file="$TFLITE_FILE" \
--output_format=TFLITE \
--input_shapes=1,7424,1 \
--input_arrays='input' \
--output_arrays='output/Softmax'
