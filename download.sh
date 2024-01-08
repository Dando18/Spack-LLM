#!/bin/bash
# Download the data from the source
# usage: ./download.sh <?model>

POSSIBLE_MODELS="all 7B_16bit 7B_8bit 7B_6bit 7B_4bit 13B_16bit 13B_8bit 13B_6bit 13B_4bit"

if [ $# -eq 1 ]; then
    if [ $1 = "-h" ] || [ $1 = "--help" ]; then
        echo "usage: ./download.sh <?model>"
        echo "model: $POSSIBLE_MODELS"
        exit 0
    fi
    MODEL="$1"
else
    # get from stdin
    echo "Available models: $POSSIBLE_MODELS"
    printf "Enter model name:  "
    read MODEL
fi

# switch on model name
case "$MODEL" in
    "all")
        echo "Downloading all models..."
        ./download.sh 7B_16bit
        ./download.sh 7B_8bit
        ./download.sh 7B_6bit
        ./download.sh 7B_4bit
        ./download.sh 13B_16bit
        ./download.sh 13B_8bit
        ./download.sh 13B_6bit
        ./download.sh 13B_4bit
        ;;
    # any of the others
    "7B_16bit"|"7B_8bit"|"7B_6bit"|"7B_4bit"|"13B_16bit"|"13B_8bit"|"13B_6bit"|"13B_4bit")
        echo "Downloading $MODEL..."
        PARAMS=$(echo $MODEL | cut -d'_' -f1 | tr '[:upper:]' '[:lower:]')
        BITS=$(echo $MODEL | cut -d'_' -f2 | cut -d'b' -f1)
        ;;
    *)
        echo "Invalid model name: $MODEL"
        echo "Available models: $POSSIBLE_MODELS"
        exit 1
        ;;
esac

case "$BITS" in
    "16")
        BITS="f16"
        ;;
    "8"|"4")
        BITS="q${BITS}_0"
        ;;
    "6")
        BITS="q6_k"
        ;;
    *)
        echo "Invalid model name: $MODEL"
        exit 1
        ;;
esac

# download the data
# https://huggingface.co/daniellnichols/spack-llama-7b/resolve/main/ggml-model-f16.gguf?download=true
DST_ROOT="models/spack-llama-${PARAMS}"
mkdir -p $DST_ROOT
DST="${DST_ROOT}/ggml-model-${BITS}.gguf"
URL="https://huggingface.co/daniellnichols/spack-llama-${PARAMS}/resolve/main/ggml-model-${BITS}.gguf?download=true"
wget -O $DST $URL