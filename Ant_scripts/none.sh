time CUDA_VISIBLE_DEVICES=$1 python main.py --cuda \
    --env-name Ant-v3 \
    --id $0 \
    --seed $2 \
    --switching none