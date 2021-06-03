time CUDA_VISIBLE_DEVICES=$1 python main.py --cuda \
    --env-name Swimmer-v3 --lr 0.0003 \
    --id $0 \
    --seed $2 \
    --switching info-det \
    --visit-eta 10