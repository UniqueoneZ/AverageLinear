if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
model_name=AverageLinear

root_path_name=./dataset/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2


seq_len=96
for pred_len in 96
do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --d_model 256 \
      --dropout 0.2 \
      --train_epochs 10 \
      --patience 3 \
      --channel_id 1 \
      --loss mse \
      --revin 1 \
      --c_layers 0 \
      --itr 1 --batch_size 512 --learning_rate 0.0003 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done


seq_len=96
for pred_len in 192
do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --d_model 256 \
      --dropout 0.2 \
      --train_epochs 10 \
      --patience 3 \
      --channel_id 1 \
      --loss mse \
      --revin 1 \
      --c_layers 0 \
      --itr 1 --batch_size 512 --learning_rate 0.0003 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done


seq_len=96
for pred_len in 336
do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --d_model 256 \
      --dropout 0.2 \
      --train_epochs 10 \
      --patience 3 \
      --channel_id 1 \
      --loss mse \
      --revin 1 \
      --c_layers 0 \
      --itr 1 --batch_size 512 --learning_rate 0.0003 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done


seq_len=96
for pred_len in 720
do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --d_model 512 \
      --dropout 0.3 \
      --train_epochs 10 \
      --patience 3 \
      --channel_id 1 \
      --loss mse \
      --revin 1 \
      --c_layers 0 \
      --itr 1 --batch_size 512 --learning_rate 0.0003 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
