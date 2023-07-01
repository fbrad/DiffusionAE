## Train diffusion-based models on time series

### Train autoencoder
```
python DiffusionAE/train_transformer_val.py --model TransformerBasicBottleneckScaling --window_size 100 --lr 1e-4 --batch_size 128 --dataset pattern_seasonal
```

### Train diffusion model
```
python DiffusionAE/train_diffusion_val.py --dataset pattern_seasonal --denoise_steps 80 --batch_size 8 --training diffusion --lr 1e-3 --window_size 100 --noise_steps 100 
```

### Train DiffusionAE
```
python DiffusionAE/train_diffusion_val.py --dataset pattern_seasonal --diff_lambda 0.1 --denoise_steps 50 --batch_size 16 --anomaly_score diffusion --training both --model TransformerBasicBottleneckScaling --lr 1e-3 --window_size 100 --noise_steps 100
```

## Evaluate models


Set `CHECKPOINTS_FOLDER` inside `train_diffusion_val.py` and `train_transformer_val.py` to point to the trained models path.

### Test autoencoder
```
python DiffusionAE/train_transformer_val.py --model TransformerBasicBottleneckScaling --window_size 100 --lr 1e-4 --batch_size 128 --dataset pattern_seasonal --test_only True
```

### Test diffusion model
```
python DiffusionAE/train_diffusion_val.py --dataset pattern_seasonal --denoise_steps 80 --batch_size 8 --training diffusion --lr 1e-3 --window_size 100 --noise_steps 100 --test_only True
```

### Test DiffusionAE
```
python TranAD/train_diffusion_val.py --dataset pattern_seasonal --diff_lambda 0.1 --denoise_steps 50 --batch_size 16 --anomaly_score diffusion --training both --model TransformerBasicBottleneckScaling --lr 1e-3 --window_size 100 --noise_steps 100 --test_only True
```
