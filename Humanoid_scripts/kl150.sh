time CUDA_VISIBLE_DEVICES=$1 python main.py --cuda \
    --env-name Humanoid-v3 --alpha 0.05 \
    --id $0 \
    --seed $2 \
    --switching kl \
    --policy_kl 1.5