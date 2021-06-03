time CUDA_VISIBLE_DEVICES=$1 python main.py --cuda \
    --env-name Walker2d-v3 \
    --id $0 \
    --seed $2 \
    --switching feature \
    --feature_sim 0.98