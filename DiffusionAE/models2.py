import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer
from torch.nn import TransformerDecoder
from src.dlutils import *
from unet2 import Unet
from scipy import stats
from diffusion_module2 import p_losses, sample

device = 'cuda'
	

class Autoencoder_Diffusion(nn.Module):
	def __init__(self, feats, lr, window_size, batch_size, p1, p2):
		super().__init__()
		self.name = 'AutoencoderDiffusion'
		self.lr = lr
		self.batch = batch_size
		self.n_feats = feats
		self.n_window = window_size
		self.p1 = p1
		self.p2 = p2
		self.bottleneck_s1 = int(p1 * self.n_feats * self.n_window)
		self.bottleneck_s2 = int(p2 * self.n_feats * self.n_window)
		self.bottleneck = 32

		self.encoder = torch.nn.Sequential(
			torch.nn.Linear(self.n_feats * self.n_window, self.bottleneck_s1),
			torch.nn.ReLU(), 
			torch.nn.Linear(self.bottleneck_s1, self.bottleneck_s2),
		)
		
		#self.encoder = torch.nn.Linear(self.n_feats * self.n_window, self.bottleneck)

		self.decoder = torch.nn.Sequential(
			torch.nn.Linear(self.bottleneck_s2, self.bottleneck_s1),
			torch.nn.ReLU(),
			torch.nn.Linear(self.bottleneck_s1, self.n_feats * self.n_window),
		)

		self.activation_fn = torch.nn.ReLU()

		#self.decoder = torch.nn.Linear(self.bottleneck, self.n_feats * self.n_window)


	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded


class ConditionalDiffusionTrainingNetwork(nn.Module):
	def __init__(self,nr_feats, window_size, batch_size, noise_steps, denoise_steps, train=True):
		super().__init__()
		self.dim = min(nr_feats, 16)
		self.nr_feats = nr_feats
		self.window_size = window_size
		self.batch_size = batch_size

		self.training = train
		self.timesteps = noise_steps
		self.denoise_steps = denoise_steps

		self.denoise_fn = Unet(dim=self.dim, channels=1, resnet_block_groups=1, init_size=torch.Size([self.dim, self.window_size, self.nr_feats]))

	def forward(self, x):
		diffusion_loss = None
		x_recon = None

		x = x.reshape(-1,  1, self.window_size, self.nr_feats)		
		if self.training:
			t = torch.randint(0, self.timesteps, (x.shape[0],), device=device).long()
			diffusion_loss = p_losses(self.denoise_fn, x, t)
		else:
			x_recon = sample(self.denoise_fn, shape=(x.shape[0], 1, self.window_size, self.nr_feats), x_start=x, denoise_steps=self.denoise_steps)

		return diffusion_loss, x_recon



class TransformerBasic(nn.Module):
	def __init__(self, feats):
		super().__init__()
		self.name = 'TransformerBasic'
		self.lr = 0.1
		self.batch = 128
		self.n_feats = feats
		self.n_window = 10

		self.lin = nn.Linear(1, feats)
		self.out_lin = nn.Linear(feats, 1)
		self.pos_encoder = PositionalEncoding(feats, 0.1, feats*self.n_window)
		encoder_layers = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers = TransformerDecoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
		self.fcn = nn.Sigmoid()

	def forward(self, src, tgt):
		# bs x (ws x features) x features
		src = src * math.sqrt(self.n_feats)
		src = self.lin(src.unsqueeze(2))
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)

		tgt = tgt * math.sqrt(self.n_feats)
		tgt = self.lin(tgt.unsqueeze(2))
		tgt = self.pos_encoder(tgt)
		x = self.transformer_decoder(tgt, memory)
		x = self.out_lin(x)
		x = self.fcn(x)
		return x

class TransformerBasicv2(nn.Module):
	def __init__(self, feats, lr, window_size):
		super(TransformerBasicv2, self).__init__()
		self.name = 'TransformerBasicv2'
		self.lr = lr
		self.batch = 128
		self.n_feats = feats
		self.n_window = window_size
		self.scale = 16
		self.linear_layer = nn.Linear(feats, self.scale*feats)
		self.output_layer = nn.Linear(self.scale*feats, feats)
		self.pos_encoder = PositionalEncoding(self.scale*feats, 0.1, self.n_window, batch_first=True)
		encoder_layers = TransformerEncoderLayer(d_model=feats*self.scale, nhead=feats, batch_first=True, dim_feedforward=256, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers = TransformerDecoderLayer(d_model=feats*self.scale, nhead=feats, batch_first=True, dim_feedforward=256, dropout=0.1)
		self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
		self.fcn = nn.Sigmoid()

	def forward(self, src, tgt):
		src = src * math.sqrt(self.n_feats)
		src = self.linear_layer(src)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src) 

		tgt = tgt * math.sqrt(self.n_feats)
		tgt = self.linear_layer(tgt)

		x = self.transformer_decoder(tgt, memory)
		x = self.output_layer(x)
		x = self.fcn(x)
		return x

class TransformerBasicv2Scaling(nn.Module):
	def __init__(self, feats, lr, window_size):
		super(TransformerBasicv2Scaling, self).__init__()
		self.name = 'TransformerBasicv2Scaling'
		self.lr = lr
		self.batch = 128
		self.n_feats = feats
		self.n_window = window_size
		self.scale = 16
		self.linear_layer = nn.Linear(feats, self.scale*feats)
		self.output_layer = nn.Linear(self.scale*feats, feats)
		self.pos_encoder = PositionalEncoding(self.scale*feats, 0.1, self.n_window, batch_first=True)
		encoder_layers = TransformerEncoderLayer(d_model=feats*self.scale, nhead=feats, batch_first=True, dim_feedforward=256, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers = TransformerDecoderLayer(d_model=feats*self.scale, nhead=feats, batch_first=True, dim_feedforward=256, dropout=0.1)
		self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
		self.fcn = nn.Sigmoid()

	def forward(self, src, tgt):
		model_dim = self.scale * self.n_feats

		src = self.linear_layer(src)
		src = src * math.sqrt(model_dim)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src) 

		tgt = self.linear_layer(tgt)
		tgt = tgt * math.sqrt(model_dim)

		x = self.transformer_decoder(tgt, memory)
		x = self.output_layer(x)
		x = self.fcn(x)
		return x



class TransformerBasicBottleneck(nn.Module):
	def __init__(self, feats, lr, window_size):
		super(TransformerBasicBottleneck, self).__init__()
		self.name = 'TransformerBasicBottleneck'
		self.lr = lr
		self.batch = 16
		self.n_feats = feats
		self.n_window = window_size
		self.scale = 16
		self.linear_layer = nn.Linear(feats, self.scale*feats)
		self.output_layer = nn.Linear(self.scale*feats, feats)
		self.pos_encoder = PositionalEncoding(self.scale*feats, 0.1, self.n_window, batch_first=True)
		encoder_layers = TransformerEncoderLayer(d_model=feats*self.scale, nhead=feats, batch_first=True, dim_feedforward=256, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers = TransformerDecoderLayer(d_model=feats*self.scale, nhead=feats, batch_first=True, dim_feedforward=256, dropout=0.1)
		self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
		self.fcn = nn.Sigmoid()

	def forward(self, src, tgt):
		src = src * math.sqrt(self.n_feats)
		src = self.linear_layer(src)
		src = self.pos_encoder(src)
		# batch x t x d
		memory = self.transformer_encoder(src) 
		# batch x 1 x d
		z = torch.mean(memory, dim=1, keepdim=True)


		tgt = tgt * math.sqrt(self.n_feats)
		tgt = self.linear_layer(tgt)

		x = self.transformer_decoder(tgt, z)
		x = self.output_layer(x)
		x = self.fcn(x)
		return x

class TransformerBasicBottleneckScaling(nn.Module):
	def __init__(self, feats, lr, window_size, batch_size):
		super(TransformerBasicBottleneckScaling, self).__init__()
		self.name = 'TransformerBasicBottleneckScaling'
		self.lr = lr
		self.batch = batch_size
		self.n_feats = feats
		self.n_window = window_size
		self.scale = 16
		self.linear_layer = nn.Linear(feats, self.scale*feats)
		self.output_layer = nn.Linear(self.scale*feats, feats)
		self.pos_encoder = PositionalEncoding(self.scale*feats, 0.1, self.n_window, batch_first=True)
		encoder_layers = TransformerEncoderLayer(d_model=feats*self.scale, nhead=feats, batch_first=True, dim_feedforward=256, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers = TransformerDecoderLayer(d_model=feats*self.scale, nhead=feats, batch_first=True, dim_feedforward=256, dropout=0.1)
		self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
		self.fcn = nn.Sigmoid()

	def forward(self, src, tgt):
		model_dim = self.scale * self.n_feats

		src = self.linear_layer(src)
		src = src * math.sqrt(model_dim)
		src = self.pos_encoder(src)
		# batch x t x d
		memory = self.transformer_encoder(src) 
		# batch x 1 x d
		z = torch.mean(memory, dim=1, keepdim=True)

		tgt = self.linear_layer(tgt)
		tgt = tgt * math.sqrt(model_dim)

		x = self.transformer_decoder(tgt, z)
		x = self.output_layer(x)
		x = self.fcn(x)
		return x



