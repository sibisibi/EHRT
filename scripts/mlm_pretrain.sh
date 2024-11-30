PROJECT_DIR="/home/min99830/SiBeen/ehrt"
cd "${PROJECT_DIR}" 

TASK_NAME="mlm_pretrain_omop_fn"
FAKE_DATASET_PATH="${PROJECT_DIR}/data/dataset/fake/${TASK_NAME}.pkl"
REAL_DATASET_PATH="${PROJECT_DIR}/data/dataset/real/${TASK_NAME}.pkl"
TOKENIZER_PATH="${PROJECT_DIR}/data/tokenizer/feat_tokenizers.pkl"
OUTPUT_PATH="${PROJECT_DIR}/output/pretrain"

exp_name="SET_EXPERIMENT_NAME"
CUDA_VISIBLE_DEVICES=0 python src/ehrt/train/mlm_pretrain.py \
    --exp_name ${exp_name} \
    --fake_dataset_path ${FAKE_DATASET_PATH} --real_dataset_path ${REAL_DATASET_PATH} \
    --tokenizer_path ${TOKENIZER_PATH} --output_path ${OUTPUT_PATH}

#############################################################################################################################################