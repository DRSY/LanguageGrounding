###
# @Author: your name
# @Date: 2021-03-14 00:02:26
 # @LastEditTime: 2021-04-04 23:18:16
 # @LastEditors: Please set LastEditors
# @Description: In User Settings Edit
# @FilePath: /grounding/src/scripts/pretrain_translator.sh
###
WORK_DIR="/home/roy/grounding/src/code"

bert_name="bert-base-uncased"
roberta_name="roberta-base"
mpnet_name="microsoft/mpnet-base"

python -Wignore -u $WORK_DIR/main.py --model_type mpnet --model_name ${mpnet_name} \
    --pretrain_trans_bs 16 \
    --pretrain_trans_epochs 1 \
    --pretrain_trans_lr 3e-4 \
    --device cuda:3 \
    --seed 42 \
    --trans_nonlinearity tanh \
    --grounding_lr 3e-4 \
    --pretrain_grounding_bs 16 \
    --adapter_heads 4 \
    --adapter_size 128 \
    --adapter_list "0,6,11" \
    --adapter_transformer_layers 2 \
    --do_train \
    --do_save \
    --eval_step 2000 \
    --loss_type bilinear
