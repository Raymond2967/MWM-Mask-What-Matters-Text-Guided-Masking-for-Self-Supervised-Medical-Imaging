#!/bin/bash

DATASET=$1
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results/${TIMESTAMP}"
mkdir -p ${OUTPUT_DIR}

echo "Run timestamp: $TIMESTAMP" > ${OUTPUT_DIR}/args.txt
echo "Dataset: $DATASET" >> ${OUTPUT_DIR}/args.txt

GENERATE_EXPANDED=true
EXPAND_RATIO=1
COARSE_EXPANDED_PATH=expanded_masks/coarse/${DATASET}/masks
SAM_EXPANDED_PATH=expanded_masks/sam/${DATASET}/masks

if [ "$GENERATE_EXPANDED" = "true" ]; then
    mkdir -p ${COARSE_EXPANDED_PATH}
    mkdir -p ${SAM_EXPANDED_PATH}
fi

python saliency_maps/generate_saliency_maps.py \
--input-path "${DATASET}/test_images" \
--output-path "saliency_map_outputs/${DATASET}/masks" \
--model-name BiomedCLIP \
--finetuned \
--hyper-opt \
--val-path ${DATASET}/val_images

mkdir -p "coarse_outputs/${DATASET}/masks"

python postprocessing/postprocess_saliency_maps.py \
--input-path "${DATASET}/test_images" \
--sal-path "saliency_map_outputs/${DATASET}/masks" \
--output-path "coarse_outputs/${DATASET}/masks" \
--postprocess thresholding \
--threshold 0.3 \
--filter \
--num-contours 2 \
--generate-expanded \
--expanded-output-path "${COARSE_EXPANDED_PATH}" \
--expand-ratio "${EXPAND_RATIO}"

mkdir -p "sam_outputs/${DATASET}/masks"

python segment-anything/prompt_sam.py \
--input "${DATASET}/test_images" \
--mask-input "coarse_outputs/${DATASET}/masks" \
--output "sam_outputs/${DATASET}/masks" \
--model-type vit_h \
--checkpoint segment-anything/sam_checkpoints/sam_vit_h_4b8939.pth \
--prompts boxes \
--multicontour

python postprocessing/postprocess_sam.py \
--sam-path "sam_outputs/${DATASET}/masks" \
--output-path "${SAM_EXPANDED_PATH}" \
--generate-expanded \
--expand-ratio "${EXPAND_RATIO}"

python evaluation/eval.py \
--gt_path "${DATASET}/test_masks" \
--seg_path "coarse_outputs/${DATASET}/masks" \
--eval-type coarse_output \
--output_dir ${OUTPUT_DIR}

if [ "$GENERATE_EXPANDED" = "true" ]; then
    python evaluation/eval.py \
    --gt_path "${DATASET}/test_masks" \
    --seg_path "coarse_outputs/${DATASET}/masks" \
    --expanded-path "${COARSE_EXPANDED_PATH}" \
    --eval-type coarse_expanded \
    --output_dir ${OUTPUT_DIR}
fi

python evaluation/eval.py \
--gt_path "${DATASET}/test_masks" \
--seg_path "sam_outputs/${DATASET}/masks" \
--eval-type sam_output \
--output_dir ${OUTPUT_DIR}

python evaluation/eval.py \
--gt_path "${DATASET}/test_masks" \
--seg_path "sam_outputs/${DATASET}/masks" \
--expanded-path "${SAM_EXPANDED_PATH}" \
--eval-type sam_expanded \
--output_dir ${OUTPUT_DIR}
