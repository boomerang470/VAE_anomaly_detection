from abc import abstractmethod, ABC

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
from torch.nn.functional import softplus
import pytorch_lightning as pl


class VAEAnomalyDetection(pl.LightningModule, ABC):

    def __init__(self, input_size: int, latent_size: int, L: int = 10, lr: float = 1e-3, log_steps: int = 1_000):
        super().__init__()
        self.L = L
        self.lr = lr
        self.input_size = input_size
        self.latent_size = latent_size
        self.encoder = self.make_encoder(input_size, latent_size)
        self.decoder = self.make_decoder(latent_size, input_size)
        self.prior = Normal(0, 1)
        self.log_steps = log_steps

    @abstractmethod
    def make_encoder(self, input_size: int, latent_size: int) -> nn.Module:
        
        pass

    @abstractmethod
    def make_decoder(self, latent_size: int, output_size: int) -> nn.Module:
        
        pass

    def forward(self, x: torch.Tensor) -> dict:
        
        pred_result = self.predict(x)
        x = x.unsqueeze(0)  # unsqueeze to broadcast input across sample dimension (L)
        log_lik = Normal(pred_result['recon_mu'], pred_result['recon_sigma']).log_prob(x).mean(
            dim=0)  # average over sample dimension
        log_lik = log_lik.mean(dim=0).sum()
        kl = kl_divergence(pred_result['latent_dist'], self.prior).mean(dim=0).sum()
        loss = kl - log_lik
        return dict(loss=loss, kl=kl, recon_loss=log_lik, **pred_result)

    def predict(self, x) -> dict:
        
        batch_size = len(x)
        latent_mu, latent_sigma = self.encoder(x).chunk(2, dim=1) #both with size [batch_size, latent_size]
        latent_sigma = softplus(latent_sigma)
        dist = Normal(latent_mu, latent_sigma)
        z = dist.rsample([self.L])  # shape: [L, batch_size, latent_size]
        z = z.view(self.L * batch_size, self.latent_size)
        recon_mu, recon_sigma = self.decoder(z).chunk(2, dim=1)
        recon_sigma = softplus(recon_sigma)
        recon_mu = recon_mu.view(self.L, *x.shape)
        recon_sigma = recon_sigma.view(self.L, *x.shape)
        return dict(latent_dist=dist, latent_mu=latent_mu,
                    latent_sigma=latent_sigma, recon_mu=recon_mu,
                    recon_sigma=recon_sigma, z=z)

    def is_anomaly(self, x: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
        
        p = self.reconstructed_probability(x)
        return p < alpha

    def reconstructed_probability(self, x: torch.Tensor) -> torch.Tensor:
        
        with torch.no_grad():
            pred = self.predict(x)
        recon_dist = Normal(pred['recon_mu'], pred['recon_sigma'])
        x = x.unsqueeze(0)
        p = recon_dist.log_prob(x).exp().mean(dim=0).mean(dim=-1)  # vector of shape [batch_size]
        return p

    def generate(self, batch_size: int = 1) -> torch.Tensor:
        
        z = self.prior.sample((batch_size, self.latent_size))
        recon_mu, recon_sigma = self.decoder(z).chunk(2, dim=1)
        recon_sigma = softplus(recon_sigma)
        return recon_mu + recon_sigma * torch.rand_like(recon_sigma)
    
    
    def training_step(self, batch, batch_idx):
        x = batch
        loss = self.forward(x)
        if self.training and self.global_step % self.log_steps == 0:
            self.log('train/loss', loss['loss'])
            self.log('train/loss_kl', loss['kl'], prog_bar=False)
            self.log('train/loss_recon', loss['recon_loss'], prog_bar=False)
            self._log_norm()

        return loss
    

    def validation_step(self, batch, batch_idx):
        x = batch
        loss = self.forward(x)
        self.log('val/loss_epoch', loss['loss'], on_epoch=True)
        self.log('val/loss_kl', loss['kl'], self.global_step)
        self.log('val/loss_recon', loss['recon_loss'], self.global_step)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


    def _log_norm(self):
        norm1 = sum(p.norm(1) for p in self.parameters())
        norm1_grad = sum(p.grad.norm(1) for p in self.parameters() if p.grad is not None)
        self.log('norm1_params', norm1)
        self.log('norm1_grad', norm1_grad)

class VAEAnomalyTabular(VAEAnomalyDetection):

    def make_encoder(self, input_size, latent_size):
       
        return nn.Sequential(
            nn.Linear(input_size, 500),
            nn.ReLU(),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, latent_size * 2)
            # times 2 because this is the concatenated vector of latent mean and variance
        )

    def make_decoder(self, latent_size, output_size):
       
        return nn.Sequential(
            nn.Linear(latent_size, 200),
            nn.ReLU(),
            nn.Linear(200, 500),
            nn.ReLU(),
            nn.Linear(500, output_size * 2)  # times 2 because this is the concatenated vector of reconstructed mean and variance
        )


