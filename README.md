# SincKAN
## Requirements

To install requirements:

By anaconda:
```setup
conda install -r environment.yml
```

By docker:
```setupd

```

## Data

To generate data, you can use the current dataset in data.py or you can add new data.

## Training

To train the model(s) in the paper, change the directory to the specific directory,

for example, run command for approximation:

```train
cd ./approximation/
python approximation_1d.py --mode train
```

## Evaluation

To train the model(s) in the paper, change the directory to the specific directory,

for example, run command for approximation:

```train
cd ./approximation/
python approximation_1d.py --mode eval
```

## Results ($L^2$ Relative errors)
We demostrate partial results of our paper:

| Model name | MLP             | KAN             | SincKAN         | 
|------------|-----------------|-----------------|-----------------|
| pbl        | 2.89e-2 ± 3.09e-2 | 4.48e-3 ± 4.20e-3 | 1.88e-3 ± 8.55e-4 |
| bl_1000    | 9.87 ± 8.70 | 11.3 ± 8.79 | 5.48e-3 ± 3.45e-3 | 
