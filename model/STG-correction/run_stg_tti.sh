#!/bin/bash
# Copyright 2022 The ZJU MMF Authors (Lvxiaowei Xu, Jianwang Wu, Jiawei Peng, Jiayu Fu and Ming Cai *).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Train and Test for STG-Indep+TTI
# Global Variable (!!! SHOULD ADAPT TO YOUR CONFIGURATION !!!)
CUDA_ID=0
SEED=2022
EPOCH=50
BATCH_SIZE=64
MAX_GENERATE=6 # MAX T
CHECKPOINT_DIR=checkpoints
# Roberta-base-chinese can be downloaded at https://github.com/ymcui/Chinese-BERT-wwm
PLM_PATH=/datadisk2/xlxw/Resources/pretrained_models/roberta-base-chinese/ # pretrained-model path
OUTPUT_PATH=stg_indep_tti_test.xlsx

# STEP 1 - PREPROCESS DATASET
DATA_BASE_DIR=dataset
DATA_OUT_DIR=stg_indep_tti
DATA_TRAIN_FILE=FCGEC_train.json
DATA_VALID_FILE=FCGEC_valid.json
DATA_TEST_FILE=FCGEC_test.json

python preprocess_data.py --mode tti --err_only True \
--data_dir ${DATA_BASE_DIR} --out_dir ${DATA_OUT_DIR} \
--train_file ${DATA_TRAIN_FILE} --valid_file ${DATA_VALID_FILE} --test_file ${DATA_TEST_FILE}

# STEP 2 - TRAIN STG-Indep+TTI MODEL
# -- Switch Module --
SWITCH_CHECK_DIR=switch_tti_module
python indep_tti_switch.py --mode train \
--gpu_id ${CUDA_ID} \
--seed ${SEED} \
--checkpoints ${CHECKPOINT_DIR} \
--checkp ${SWITCH_CHECK_DIR} \
--data_base_dir ${DATA_BASE_DIR}/${DATA_OUT_DIR} \
--lm_path ${PLM_PATH} \
--batch_size ${BATCH_SIZE} \
--epoch ${EPOCH}

# -- Tagger Module --
TAGGER_CHECK_DIR=tagger_tti_module
python indep_tti_tagger.py --mode train \
--gpu_id ${CUDA_ID} \
--seed ${SEED} \
--checkpoints ${CHECKPOINT_DIR} \
--checkp ${TAGGER_CHECK_DIR} \
--data_base_dir ${DATA_BASE_DIR}/${DATA_OUT_DIR} \
--max_generate ${MAX_GENERATE} \
--lm_path ${PLM_PATH} \
--batch_size ${BATCH_SIZE} \
--epoch ${EPOCH}

# -- Generator Module --
GEN_CHECK_DIR=generator_tti_module
python indep_generator.py --mode train \
--gpu_id ${CUDA_ID} \
--seed ${SEED} \
--checkpoints ${CHECKPOINT_DIR} \
--checkp ${GEN_CHECK_DIR} \
--data_base_dir ${DATA_BASE_DIR}/${DATA_OUT_DIR} \
--max_generate ${MAX_GENERATE} \
--lm_path ${PLM_PATH} \
--batch_size ${BATCH_SIZE} \
--epoch ${EPOCH}

# STEP 3 - Predict Test Output STG-Indep+TTI MODEL
python indep_evaluate.py --mode test --gpu_id ${CUDA_ID} --seed ${SEED} --is_tti True \
--checkpoints ${CHECKPOINT_DIR} --checkp_switch ${SWITCH_CHECK_DIR} --checkp_tagger ${TAGGER_CHECK_DIR} --checkp_gen ${GEN_CHECK_DIR} \
--export ${OUTPUT_PATH} \
--data_base_dir ${DATA_BASE_DIR}/${DATA_OUT_DIR} \
--max_generate ${MAX_GENERATE} \
--lm_path ${PLM_PATH} \
--batch_size ${BATCH_SIZE}