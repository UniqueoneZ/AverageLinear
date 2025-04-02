if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
model_name=AverageLinear

root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=Electricity
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
      --enc_in 321 \
      --d_model 256 \
      --dropout 0.01 \
      --train_epochs 10 \
      --patience 3 \
      --channel_id 1 \
      --loss mse \
      --c_layers 3 \
      --revin 1 \
      --itr 1 --batch_size 256 --learning_rate 0.001 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
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
      --enc_in 321 \
      --d_model 256 \
      --dropout 0 \
      --train_epochs 10 \
      --patience 3 \
      --channel_id 1 \
      --loss mse \
      --c_layers 4 \
      --revin 1 \
      --itr 1 --batch_size 256 --learning_rate 0.0011 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
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
      --enc_in 321 \
      --d_model 256 \
      --dropout 0 \
      --train_epochs 10 \
      --patience 3 \
      --channel_id 1 \
      --loss mse \
      --c_layers 3 \
      --revin 1 \
      --itr 1 --batch_size 256 --learning_rate 0.001 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
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
      --enc_in 321 \
      --d_model 256 \
      --dropout 0.02 \
      --train_epochs 10 \
      --patience 3 \
      --channel_id 1 \
      --loss mse \
      --c_layers 3 \
      --revin 1 \
      --itr 1 --batch_size 256 --learning_rate 0.001 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

