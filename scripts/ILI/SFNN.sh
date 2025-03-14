model_name=SFNN
datapath=ILI
dataset=ILI

for rid in {0..9} ; do
    for sl in 52 104 208 ; do
        for pl in 24 36 48 60 ; do
            python -u run.py \
              --root_path ./dataset/"$datapath"/ \
              --data_path "$dataset".csv \
              --model_id "$dataset"_"$sl"_"$pl"_"$rid" \
              --model $model_name \
              --data $dataset \
              --seq_len $sl \
              --pred_len $pl \
              --n_layers 1 \
              --mixer \
              --batch_size 2 \
              --train_epochs 200 \
              --patience 100 \
              --weight_decay 0.0005 \
              --dropout 0.5 \
              --loss_fn MAE \
              --learning_rate 0.01 \
              --min_lr 2e-5 \
              --norm_len 0
        done
    done
done
