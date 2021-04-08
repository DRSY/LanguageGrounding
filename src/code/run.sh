###
# @Author: your name
# @Date: 2021-04-07 15:50:11
 # @LastEditTime: 2021-04-08 10:21:59
 # @LastEditors: Please set LastEditors
# @Description: In User Settings Edit

# @FilePath: /multiple-choice/run.sh
###

# export DATASET_NAME=hellaswag
# export DATASET_NAME=cosmosqa

# export DATASET_NAME=swag
# export DATASET_NAME=social_i_qa
export DATASET_NAME=piqa
# export DATASET_NAME=commonsense_qa

bert_name="bert-base-uncased"
roberta_name="roberta-base"
mpnet_name="microsoft/mpnet-base"
electra_name="google/electra-base-discriminator"

python run_swag_no_trainer.py \
  --gradient_accumulation_steps 2 \
  --model_name_or_path ${bert_name} \
  --max_length 120 \
  --per_device_train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$DATASET_NAME/ \
  --device_id 3 \
  --dataset_name $DATASET_NAME
# --train_file /home/roy/grounding/src/code/data/cosmosqa/train.csv \
# --validation_file /home/roy/grounding/src/code/data/cosmosqa/valid.csv
