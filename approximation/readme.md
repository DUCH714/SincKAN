# Approximation

For approximation, hyperparmeters depends on different experiments:

## One-dimensional function

bl:
```train
python interpolation_1d.py --epochs 5000 --network sinckan --kanshape 16 --degree 100 --normalization 1 --datatype bl --len_h 6 --npoints 5000 --ntrain 3000 --ntest 10000 --seed 0 --device 0
```

double_exponential:
```train
python interpolation_1d.py --epochs 5000 --network sinckan --kanshape 16 --degree 100 --datatype double_exponential --interval="0.0,1.0" --len_h 6 --normalization 1 --npoints 5000 --ntrain 3000 --ntest 10000 --seed 0 --device 0
```

multi_sqrt:
```train
python interpolation_1d.py --epochs 5000 --network sinckan --kanshape 16 --degree 100 --datatype multi_sqrt --normalization 1 --interval="0.0,1.0" --len_h 6 --activation='none' --npoints 5000 --ntrain 3000 --ntest 10000 --seed 0 --device 0
```

piecewise:
```train
python interpolation_1d.py --epochs 5000 --network sinckan --kanshape 16 --degree 150 --datatype piecewise --normalization 1 --init_h 2 --len_h 6 --activation='none' --interval "0.0,2.0" --npoints 5000 --ntrain 3000 --ntest 10000 --seed 0 --device 0
```

sin_high:
```train
python interpolation_1d.py --epochs 5000 --network sinckan --kanshape 16 --degree 100 --datatype sin_high --normalization 1 --init_h 5 --len_h 6 --activation='none' --interval="-1.0,1.0" --npoints 5000 --ntrain 3000 --ntest 10000 --seed 0 --device 0
```

sin_low:
```train
python interpolation_1d.py --epochs 5000 --network sinckan --kanshape 16 --degree 80 --datatype sin_low --normalization 1 --init_h 2 --len_h 4 --interval="-1.0,1.0" --npoints 5000 --ntrain 3000 --ntest 10000 --seed 0 --device 0
```

spectral_bias:
```train
python interpolation_1d.py --epochs 5000 --network sinckan --kanshape 168 --degree 5 --datatype spectral_bias --normalization 1 --len_h 9 --init_h 2 --interval="-1.0,1.0" --activation='none' --npoints 5000 --ntrain 3000 --ntest 10000 --seed 6 --device 0
```

sqrt:
```train
python interpolation_1d.py --epochs 5000 --network sinckan --kanshape 320 --degree 6 --datatype sqrt --init_h 4 --len_h 4 --activation 'tanh' --interval="0.0,1.0" --npoints 5000 --ntrain 3000 --ntest 10000 --seed 0 --device 0
```

## Two-dimensional function

fractal-2d:
```train
python interpolation_hd.py --epochs 4000 --network sinckan --kanshape 32 --degree 100 --normalization 0 --datatype fractal --dim 2 --len_h 6 --noise 0 --activation None --interval='0,1' --skip 0 --ntrain 2000 --ntest 4000 --seed 0 --device 0
```

multimodal1-2d:
```train
python interpolation_hd.py --epochs 5000 --network sinckan --kanshape 32,32 --degree 50 --normalization 0 --datatype multimodal1 --dim 2 --init_h 2.0 --len_h 6 --noise 0 --activation None --initialization 'Xavier' --skip 0 --ntrain 2000 --ntest 4000 --seed 0 --device 0
```

multimodal2-2d:
```train
python interpolation_hd.py --epochs 4000 --network sinckan --kanshape 16 --degree 100 --normalization 0 --datatype multimodal2 --dim 2 --len_h 6 --noise 0 --activation 'tanh' --skip 0 --ntrain 2000 --ntest 4000 --seed 0 --device 0
```

## Four-dimensional function

exp-sin-4d:
```train
python interpolation_hd.py --epochs 10000 --network sinckan --kanshape 16,16 --degree 170 --normalization 0 --datatype 4D --dim 2 --init_h 2.0 --len_h 5 --noise 0 --activation None --initialization 'Xavier' --skip 0 --ntrain 2000 --ntest 4000 --seed 0 --device 0
```

## 100-dimensional function

exp-100d:
```train
python interpolation_hd.py --epochs 10000 --network sinckan --kanshape 8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8 --degree 3 --normalization 0 --datatype 100D --dim 100 --len_h 7 --noise 0 --activation 'tanh' --skip 0 --initialization 'Xavier' --ntrain 2000 --ntest 4000 --seed 0 --device 0
```

## Implicit function

ellipj:
```train
python approximation_hd_implicit.py  --datatype ellipj --network sinckan --len_h 4 --init_h 1.0 --epochs 10000
```

lpmv:
```train
python approximation_hd_implicit.py  --datatype lpmv --network sinckan --len_h 2 --init_h 4.0
```

sph_harm:
```train
python approximation_hd_implicit.py  --datatype sph_harm11 --network sinckan --len_h 4 --degree 80
```

## Update basis

```train
python approximation_1d_update_basis.py
```
