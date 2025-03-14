model_name=SFNN
datapath=traffic
dataset=traffic

for rid in {0..9} ; do
    for sl in 168 336 672 1344 ; do
        for pl in 96 192 336 720 ; do
            python -u run.py \
              --root_path ./dataset/"$datapath"/ \
              --data_path "$dataset".csv \
              --model_id "$dataset"_"$sl"_"$pl"_"$rid" \
              --model $model_name \
              --data $dataset \
              --seq_len $sl \
              --pred_len $pl \
              --n_layers 4 \
              --batch_size 16 \
              --train_epochs 100 \
              --weight_decay 0.00005 \
              --dropout 0.1 \
              --loss_fn MSE \
              --learning_rate 0.0004
        done
    done
done
