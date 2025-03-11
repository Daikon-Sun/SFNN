model_name=SFNN
datapath=weather
dataset=weather

for rid in {0..9} ; do
    for sl in 144 288 576 1008 ; do
        for pl in 96 192 336 720 ; do
            python -u run.py \
              --root_path ./dataset/"$datapath"/ \
              --data_path "$dataset".csv \
              --model_id "$dataset"_"$sl"_"$pl"_"$rid" \
              --model $model_name \
              --data $dataset \
              --seq_len $sl \
              --pred_len $pl \
              --n_layers 1 \
              --batch_size 256 \
              --train_epochs 50 \
              --weight_decay 0.0005 \
              --dropout 0.05 \
              --loss_fn MSE \
              --learning_rate 0.001
        done
    done
done
