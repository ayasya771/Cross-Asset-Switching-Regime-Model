"""regimelab: cross-asset market regime detection via a differentiable
latent state-space model (VAE + neural SDE prior + differentiable Kalman filter).

Implements the architecture described in the blueprint
"Cross-Asset Market Regime Switching via Differentiable Latent State-Space Models":

- continuous-time latent dynamics  dz = f_theta(z) dt + g_theta(z) dW
- a continuous-discrete extended Kalman filter embedded in the computation
  graph (predict via SDE moment equations, masked measurement updates)
- an emission decoder with a Nelson-Siegel structural head for the yield curve
- an ELBO objective with KL annealing, trained end to end by backpropagating
  through the filter
- regime extraction by clustering the filtered latent trajectory
"""

__version__ = "1.0.0"
