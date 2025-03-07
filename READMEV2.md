# SGCC

## Prepare data

Run the folowing 6 commands to generate data. (Under 6 different folders)
```shell
python prepare_data.py --data_name SGCC --attack_id 1
python prepare_data.py --data_name SGCC --attack_id 2
python prepare_data.py --data_name SGCC --attack_id 3
python prepare_data.py --data_name SGCC --attack_id 4
python prepare_data.py --data_name SGCC --attack_id 5
python prepare_data.py --data_name SGCC --attack_id 6
```

## Train

```shell
python train_t.py --data_name SGCC --attack_id 1 --val_percent 0.1
```

## Predict

```shell
python predict_t.py --data_name SGCC --attack_id 1 --val_percent 0.1
```

---

# DLC

## Prepare data

Run the folowing 6 commands to generate data. (Under 6 different folders)
```shell
python prepare_data_dlc.py --data_name DLC --attack_id 1
python prepare_data_dlc.py --data_name DLC --attack_id 2
python prepare_data_dlc.py --data_name DLC --attack_id 3
python prepare_data_dlc.py --data_name DLC --attack_id 4
python prepare_data_dlc.py --data_name DLC --attack_id 5
python prepare_data_dlc.py --data_name DLC --attack_id 6
```

## Train

```shell
python train_t.py --data_name DLC --attack_id 1 --val_percent 0.1
```

## Predict

```shell
python predict_t.py --data_name DLC --attack_id 1 --val_percent 0.1
```

---

---

# More tests

## `attack_id` & `model_name`

```bash
python train_t.py --data_name DLC --attack_id 1 --model_name UNet_1D_LL
python predict_t.py --data_name DLC --attack_id 1 --model_name UNet_1D_LL
```

| `--attack_id` | 1   | 2   | 3   | 4   | 5   | 6   |
| ----------- | --- | --- | --- | --- | --- | --- |

| `--model_name` | UNet_1D_LL | UNet_1D_L | UNet_1D | UNet_1D_N | UNet_1D_NN |
| ----------- | ---------- | --------- | ------- | --------- | ---------- |
