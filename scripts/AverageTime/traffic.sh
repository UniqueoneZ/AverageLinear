if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
model_name=AverageTime

root_path_name=./dataset/
data_path_name=traffic.csv
model_id_name=traffic
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
      --enc_in 862 \
      --d_model 1024 \
      --dropout 0 \
      --train_epochs 30 \
      --patience 5 \
      --rnn_type gru \
      --dec_way pmf \
      --channel_id 0 \
      --loss mse \
      --revin 1 \
      --c_layers 2 \
      --num_layers 3 \
      --num_layers_trans 3 \
      --num_layers_linear 0 \
      --emb_dropout 0 \
      --itr 1 --batch_size 16 --learning_rate 0.0012 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
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
      --enc_in 862 \
      --d_model 1024 \
      --dropout 0 \
      --train_epochs 30 \
      --patience 5 \
      --rnn_type gru \
      --dec_way pmf \
      --channel_id 0 \
      --loss mse \
      --revin 1 \
      --c_layers 2 \
      --num_layers 3 \
      --num_layers_trans 3 \
      --num_layers_linear 0 \
      --emb_dropout 0 \
      --itr 1 --batch_size 16 --learning_rate 0.0012 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
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
      --enc_in 862 \
      --d_model 1024 \
      --dropout 0 \
      --train_epochs 30 \
      --patience 5 \
      --rnn_type gru \
      --dec_way pmf \
      --channel_id 0 \
      --loss mse \
      --revin 1 \
      --c_layers 2 \
      --num_layers 3 \
      --num_layers_trans 3 \
      --num_layers_linear 0 \
      --emb_dropout 0 \
      --itr 1 --batch_size 16 --learning_rate 0.0012 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
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
      --enc_in 862 \
      --d_model 1024 \
      --dropout 0 \
      --train_epochs 30 \
      --patience 5 \
      --rnn_type gru \
      --dec_way pmf \
      --channel_id 0 \
      --loss mse \
      --revin 1 \
      --c_layers 2 \
      --num_layers 3 \
      --num_layers_trans 3 \
      --num_layers_linear 0 \
      --emb_dropout 0 \
      --itr 1 --batch_size 16 --learning_rate 0.0012 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

