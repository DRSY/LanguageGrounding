###
# @Author: your name
# @Date: 2021-03-14 00:02:26
 # @LastEditTime: 2021-04-07 19:09:28
 # @LastEditors: Please set LastEditors
# @Description: In User Settings Edit
# @FilePath: /grounding/src/scripts/pretrain_translator.sh
###
WORK_DIR="/home/roy/grounding/src/code"

# pretrained language models
bert_name="bert-base-uncased"
roberta_name="roberta-base"
deberta_name="microsoft/deberta-base"
mpnet_name="microsoft/mpnet-base"
electra_name="google/electra-base-discriminator"

python -Wignore -u $WORK_DIR/main.py \
    --model_type roberta \
    --model_name ${roberta_name} \
    --device cuda:1 \
    --seed 42 \
    --trans_nonlinearity tanh \
    --grounding_lr 3e-4 \
    --pretrain_grounding_bs 16 \
    --adapter_heads 4 \
    --adapter_size 128 \
    --lang_size 768 \
    --adapter_list "0,6,11" \
    --adapter_transformer_layers 2 \
    --eval_step 2000 \
    --loss_type bilinear \
    --do_test \
