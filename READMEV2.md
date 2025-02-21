# Prepare data

Run the folowing 6 commands to generate data. (Under 6 different folders)
```shell
python prepare_data.py --data_name SGCC --attack_id 1
python prepare_data.py --data_name SGCC --attack_id 2
python prepare_data.py --data_name SGCC --attack_id 3
python prepare_data.py --data_name SGCC --attack_id 4
python prepare_data.py --data_name SGCC --attack_id 5
python prepare_data.py --data_name SGCC --attack_id 6
```

# Train

```shell
python train_t.py --data_name SGCC --attack_id 1 --val_percent 0.1
```

# Predict

```shell
python predict_t.py --data_name SGCC --attack_id 1 --val_percent 0.1
```