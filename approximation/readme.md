# Approximation

For approximation, hyperparmeters depends on different experiments:

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
