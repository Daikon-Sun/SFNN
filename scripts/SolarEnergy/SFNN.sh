model_name=SFNN
datapath=solar
dataset=solar

for rid in {0..9} ; do
    for sl in 144 288 576 1008 ; do
        for pl in 96 192 336 720 ; do
            if [ $pl -eq 96 ] ; then
                wd=0
                lf="MAE"
                dr=0.7
                lr=5e-4
                bs=256
            else
                wd=0.001
                lf="MSE"
                dr=0.1
                lr=5e-3
                bs=1024
            fi
            python -u run.py \
              --root_path ./dataset/"$datapath"/ \
              --data_path "$dataset".csv \
              --model_id "$dataset"_"$sl"_"$pl"_"$rid" \
              --model $model_name \
              --data $dataset \
              --seq_len $sl \
              --pred_len $pl \
              --n_layers 3 \
              --need_norm \
              --mixer \
              --batch_size $bs \
              --train_epochs 150 \
              --weight_decay $wd \
              --dropout $dr \
              --loss_fn $lf \
              --learning_rate $lr \
              --min_lr 5e-5
        done
    done
done
