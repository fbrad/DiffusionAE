## Train diffusion-based models on time series

Synthetic datasets used in the main experiments can be found in ```DiffusionAE/processed```. For SWaT and WADI, please refer to https://itrust.sutd.edu.sg/itrust-labs_datasets/

### Train autoencoder
```
python DiffusionAE/train_transformer_val.py --dataset point_global --model TransformerBasicBottleneckScaling --window_size 100 --lr 1e-3 --batch_size 128 
python DiffusionAE/train_transformer_val.py --dataset point_contextual --model TransformerBasicBottleneckScaling --window_size 100 --lr 1e-3 --batch_size 32 
python DiffusionAE/train_transformer_val.py --dataset pattern_shapelet --model TransformerBasicBottleneckScaling --window_size 100 --lr 1e-3 --batch_size 8 
python DiffusionAE/train_transformer_val.py --dataset pattern_seasonal --model TransformerBasicBottleneckScaling --window_size 100 --lr 1e-4 --batch_size 128 
python DiffusionAE/train_transformer_val.py --dataset pattern_trendv2 --model TransformerBasicBottleneckScaling --window_size 100 --lr 1e-3 --batch_size 16 
```

### Train diffusion model
```
python DiffusionAE/train_diffusion_val.py --dataset point_global --denoise_steps 50 --batch_size 8 --training diffusion --lr 1e-3 --window_size 100 --noise_steps 100 
python DiffusionAE/train_diffusion_val.py --dataset point_contextual --denoise_steps 50 --batch_size 8 --training diffusion --lr 1e-3 --window_size 100 --noise_steps 100 
python DiffusionAE/train_diffusion_val.py --dataset pattern_shapelet --denoise_steps 80 --batch_size 8 --training diffusion --lr 1e-3 --window_size 100 --noise_steps 100 
python DiffusionAE/train_diffusion_val.py --dataset pattern_seasonal --denoise_steps 50 --batch_size 8 --training diffusion --lr 1e-3 --window_size 100 --noise_steps 100 
python DiffusionAE/train_diffusion_val.py --dataset pattern_trendv2 --denoise_steps 50 --batch_size 16 --training diffusion --lr 1e-3 --window_size 100 --noise_steps 100 
```

### Train DiffusionAE
```
python DiffusionAE/train_diffusion_val.py --dataset point_global --diff_lambda 0.1 --denoise_steps 20 --batch_size 32 --anomaly_score diffusion --training both --model TransformerBasicBottleneckScaling --lr 1e-3 --window_size 100 --noise_steps 100
python DiffusionAE/train_diffusion_val.py --dataset point_contextual --diff_lambda 0.1 --denoise_steps 80 --batch_size 16 --anomaly_score diffusion --training both --model TransformerBasicBottleneckScaling --lr 1e-3 --window_size 100 --noise_steps 100
python DiffusionAE/train_diffusion_val.py --dataset shapelet --diff_lambda 0.01 --denoise_steps 80 --batch_size 8 --anomaly_score diffusion --training both --model TransformerBasicBottleneckScaling --lr 1e-3 --window_size 100 --noise_steps 100
python DiffusionAE/train_diffusion_val.py --dataset pattern_seasonal --diff_lambda 0.1 --denoise_steps 50 --batch_size 16 --anomaly_score diffusion --training both --model TransformerBasicBottleneckScaling --lr 1e-3 --window_size 100 --noise_steps 100
python DiffusionAE/train_diffusion_val.py --dataset pattern_trendv2 --diff_lambda 0.01 --denoise_steps 80 --batch_size 16 --anomaly_score diffusion --training both --model TransformerBasicBottleneckScaling --lr 1e-3 --window_size 100 --noise_steps 100
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
python DiffusionAE/train_diffusion_val.py --dataset pattern_seasonal --diff_lambda 0.1 --denoise_steps 50 --batch_size 16 --anomaly_score diffusion --training both --model TransformerBasicBottleneckScaling --lr 1e-3 --window_size 100 --noise_steps 100 --test_only True
```
