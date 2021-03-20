###
# @Author: your name
# @Date: 2021-03-14 00:02:26
 # @LastEditTime: 2021-03-20 14:17:10
 # @LastEditors: Please set LastEditors
# @Description: In User Settings Edit
# @FilePath: /grounding/src/scripts/pretrain_translator.sh
###
export WORK_DIR="/home/roy/grounding/src/code"

python -Wignore -u $WORK_DIR/main.py --model_type roberta --model_name roberta-base \
    --pretrain_trans_bs 2 \
    --pretrain_trans_epochs 1 \
    --pretrain_trans_lr 1e-4 \
    --device cuda:1 \
    --seed 42 \
    --trans_nonlinearity gelu
wait