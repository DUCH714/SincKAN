# SincKAN
## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Data

To generate data, you can use the current dataset in data/ or you can generate them by yourself via the scripts in data/, for example:

```data
python generate_heat_2d.py
```

## Training

To train the model(s) in the paper, change the directory to the specific directory,

run this command for vanilla pinn:

```train
python train_vanilla.py --data <path_to_data> 
```

run this command for fspinn with si:

```train
python train_si.py --data <path_to_data> 
```

run this command for fspinn with wl:

```train
python train_wl.py --data <path_to_data> 
```

run this command for dealiasing pinn:

```train
python train_dealias.py --data <path_to_data> 
```

## Evaluation

To evaluate the model, change the directory to the specific directory, 

run this command for vanilla pinn:

```eval
python eval_vanilla.py --model  <path_to_model>  --data <path_to_data> 
```

run this command for fspinn with si:

```eval
python eval_si.py --model  <path_to_model>  --data <path_to_data> 
```

run this command for fspinn with wl:

```eval
python eval_wl.py --model  <path_to_model>  --data <path_to_data> 
```

run this command for dealiasing pinn:

```eval
python eval_delias.py --model  <path_to_model>  --data <path_to_data> 
```

## Results ($L^2$ Relative errors)
We demostrate partial results of our paper:

| Model name     | MLP             | KAN             | SincKAN         | 
| -------------- |-----------------|-----------------|-----------------|
| convection-diffusion | 1.79e-2±4.19e-3 | 1.14e-5±1.28e-5 | 1.35e-4±1.64e-4 |
| diffusion | 1.88e-2±1.44e-2 | 4.32e-4±1.6le-4 | 4.30e-4±2.38e-4 | 
