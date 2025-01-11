if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
model_name=LightAverageLinear_weather

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

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
      --seg_len 1 \
      --enc_in 7 \
      --d_model 1024 \
      --dropout 0.9 \
      --train_epochs 10 \
      --patience 3 \
      --rnn_type gru \
      --dec_way pmf \
      --channel_id 1 \
      --loss mse \
      --c_layers 0 \
      --revin 1 \
      --itr 1 --batch_size 2 --learning_rate 0.000012 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
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
      --seg_len 1 \
      --enc_in 7 \
      --d_model 1024 \
      --dropout 0.9 \
      --train_epochs 10 \
      --patience 3 \
      --rnn_type gru \
      --dec_way pmf \
      --channel_id 1 \
      --loss mse \
      --c_layers 0 \
      --revin 1 \
      --itr 1 --batch_size 4 --learning_rate 0.00002 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
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
      --seg_len 1 \
      --enc_in 7 \
      --d_model 256 \
      --dropout 0.9 \
      --train_epochs 10 \
      --patience 3 \
      --rnn_type gru \
      --dec_way pmf \
      --channel_id 1 \
      --loss mse \
      --c_layers 1 \
      --revin 1 \
      --itr 1 --batch_size 16 --learning_rate 0.0001 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
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
      --seg_len 1 \
      --enc_in 7 \
      --d_model 512 \
      --dropout 0.5 \
      --train_epochs 10 \
      --patience 3 \
      --rnn_type gru \
      --dec_way pmf \
      --channel_id 1 \
      --loss mse \
      --c_layers 0 \
      --revin 1 \
      --itr 1 --batch_size 16 --learning_rate 0.00005 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done



