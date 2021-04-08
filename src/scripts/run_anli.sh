###
 # @Author: your name
 # @Date: 2021-03-18 15:00:41
 # @LastEditTime: 2021-04-07 19:15:12
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /grounding/abductive-commonsense-reasoning/run.sh
### 
bli=8
tli=11
init_method=normal
log_file=anli.log

mlm="/home/roy/grounding/voken_models/bert"
mlm_vlm="/home/roy/grounding/voken_models/bert_vlm"
hg="bert-base-uncased"
seed=42
bert_name="bert-base-uncased"
roberta_name="roberta-base"
deberta_name="microsoft/deberta-base"
mpnet_name="microsoft/mpnet-base"
electra_name="google/electra-base-discriminator"

python -W ignore -u ../code/anli/run_anli.py --gpu_id 1 --bli $bli --tli $tli --init_method $init_method --task_name anli --model_name_or_path $bert_name --batch_size 8 --lr 1e-5 --epochs 4 --output_dir ../code/models/anli/bert-base-uncased-lr1e-5-batch8-epoch4-seed${seed}/ --data_dir ../code/data/anli/ --finetuning_model bert --max_seq_length 68 --tb_dir ../code/models/anli/bert-base-ft-lr1e-5-batch8-epoch4-seed${seed}/tb/ --warmup_proportion 0.2 --seed $seed --metrics_out_file ../code/models/anli/bert-base-uncased-lr1e-5-batch8-epoch4-seed${seed}/metrics.json --training_data_fraction 1.0 \
    --model_type bert \
    --model_name ${bert_name} \
    --device cuda:1 \
    --trans_nonlinearity tanh \
    --grounding_lr 3e-4 \
    --pretrain_grounding_bs 16 \
    --adapter_heads 4 \
    --adapter_size 128 \
    --lang_size 768 \
    --adapter_list "0,6,11" \
    --adapter_transformer_layers 2 \
    --do_test \
    --eval_step 2000 \
    --loss_type bilinear \
    --mode train
