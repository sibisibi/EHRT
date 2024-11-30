PROJECT_DIR="/home/min99830/SiBeen/ehrt"
cd "${PROJECT_DIR}" 

TOKENIZER_PATH="${PROJECT_DIR}/data/tokenizer/feat_tokenizers.pkl"

#############################################################################################################################################

seed=9 # 9 29 45
EXP_NAME="seed_${seed}"

lr=1e-04 # 1e-04, 1e-04, 1e-03
task_name="mortality_prediction" # "mortality_prediction", "readmission_prediction", "length_of_stay_prediction"
TASK_FN="${task_name}_omop_fn"
DATASET_PATH="${PROJECT_DIR}/data/dataset/real/${TASK_FN}.pkl"
OUTPUT_PATH="${PROJECT_DIR}/output/train_from_scratch/${task_name}"

CUDA_VISIBLE_DEVICES=0 python src/ehrt/train/train_from_scratch.py \
    --seed $seed --exp_name ${EXP_NAME} --task_fn ${TASK_FN} --lr ${lr} \
    --dataset_path ${DATASET_PATH} --tokenizer_path ${TOKENIZER_PATH} \
    --output_path ${OUTPUT_PATH}

#############################################################################################################################################