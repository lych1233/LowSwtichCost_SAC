time CUDA_VISIBLE_DEVICES=$1 python main.py --cuda \
    --env-name Hopper-v3 \
    --id $0 \
    --seed $2 \
    --switching fix \
    --fix_interval 1000 \
    --record-feature-sim