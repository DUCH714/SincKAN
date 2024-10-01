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

| Model name     | MLP             | KAN             | SincKAN         | 
| -------------- |-----------------|-----------------|-----------------|
| convection-diffusion | 1.79e-2±4.19e-3 | 1.14e-5±1.28e-5 | 1.35e-4±1.64e-4 |
| diffusion | 1.88e-2±1.44e-2 | 4.32e-4±1.6le-4 | 4.30e-4±2.38e-4 | 
