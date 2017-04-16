#!/bin/bash

COUNT=6
LR=0.00002
while :
do
    COUNTprev=$(( COUNT + 0 ))
    COUNT=$(( COUNT + 1 ))
    #LR=`echo "scale=100; ${LR}*0.8" | bc`
    python train.py --nworkers 4 --memory_limit 0.5 --learn_rate=${LR} --nmaxiter=100100 --save_folder=models/autoMax${COUNT} --reload models/autoMax${COUNTprev}/model.ckpt-100000 # from auto4
    #python train.py --nworkers 4 --memory_limit 0.5 --learn_rate=${LR} --nmaxiter=100100 --save_folder=models/auto${COUNT} --reload models/auto${COUNTprev}/model.ckpt-100000 # from auto4
    #echo train.py --nworkers 4 --memory_limit 0.5 --learn_rate=1e-4 --nmaxiter=100100 --save_folder=models/auto${COUNT} --reload models/auto${COUNTprev}/model.ckpt-100000
done

