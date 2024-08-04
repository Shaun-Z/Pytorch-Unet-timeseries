# Data Preparation

The original raw data is stored in `./data/SGCC_data/data.csv`

## Manually attacked data generation

Run the script `./data_add_noise/attack_data_generate.py`, in which

```python
datapath = Path('data/SGCC_data')
df = pd.read_csv(datapath/"data.csv")
```

specifies the path to the raw data file, and

```python
group_normal_2016 = group_normal.loc[:, (group_normal.columns.year == 2016)].reset_index(drop=True).fillna(0)
```

selects the data in `2016`.

To generate attacked data and split data and labels, run the following command:

```cmd
python ./data_add_noise/attack_data_generate.py
```

then you will get the following two files to train the model.

- data: `./data/attack.csv`

- label: `./data/label.csv`

# Training

Run 

```cmd
python train_t.py
```

The model state dict will be stored in `./checkpoints`.

# Testing

The options and workflow are store in a notebook file `predict_t.ipynb`

Please check this file to generate the `result.csv` file.