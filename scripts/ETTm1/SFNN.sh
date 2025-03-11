model_name=SFNN
datapath=ETT-small
dataset=ETTm1

for rid in {0..9} ; do
    for sl in 96 192 384 672 1344 ; do
        for pl in 96 192 336 720 ; do
            python -u run.py \
              --root_path ./dataset/"$datapath"/ \
              --data_path "$dataset".csv \
              --model_id "$dataset"_"$sl"_"$pl"_"$rid" \
              --model $model_name \
              --data $dataset \
              --seq_len $sl \
              --pred_len $pl \
              --n_layers 2 \
              --mixer \
              --batch_size 256 \
              --train_epochs 100 \
              --weight_decay 0.0001 \
              --dropout 0.5 \
              --loss_fn MAE \
              --learning_rate 0.0005
        done
    done
done
