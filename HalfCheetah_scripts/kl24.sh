time CUDA_VISIBLE_DEVICES=$1 python main.py --cuda \
    --env-name HalfCheetah-v3 \
    --id $0 \
    --seed $2 \
    --switching kl \
    --policy_kl 0.24