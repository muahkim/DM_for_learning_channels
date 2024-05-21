# DM_for_learning_channels
This repository implements learning channel distributuons by using diffusion models and Wasserstein GANs for some channel models. 
This source code was used for implementing simulations in [our recent papers Our papers about this study](https://scholar.google.com/citations?view_op=view_citation&hl=ko&user=VOl55dwAAAAJ&citation_for_view=VOl55dwAAAAJ:IjCSPb-OGe4C).
Please refer to the paper for system formulation, gathered data, and the interpretation of those data.  
It is open to the public, so please feel free to play around or use it for your research, study, education, etc.


## Implementation Environment
Python >= 3.7

PyTorch >= 1.6

CUDA (I don't know exactly, but I used) 11.6

## Parameters and Options Available
The notebook files are made for different channel models and different generative models. 
For simple channel models, we use linear neural networks for the generative models and enabled the end-to-end (E2E) communication frameworks with the generated channels: 
they are named as "E2E_framework_{generative model}_{channel model}.ipynb" for AWGN channel, real Rayleigh fading channel, and a channel model with solid-state power amplifier (SSPA).
In these E2E frameworks, the generative models use all linear neural networks as the generative tasks are simple. 
The E2E framework's final goal is to learn a pair of a neural encoder and a neural decoder achieving a small symbol error rate.

To demonstrate for a more complicated setting, we included notebook files for the Clarke's model with correlated fading, which are named "Clarkes_model_{generative model}.ipynb".

The source codes are in the .ipynb format and are straightforward. You can easily find the parameters and simulation settings in the top cells and change them as you want. 
The adjustable parameters and available options are listed below.  

### Communication Channel
* Channel models: AWGN, Rayleigh, SSPA, Clarke's.
* Parameters: cardinality of the message set 'M', block length 'n', Training Eb/N0 'TRAINING_Eb/N0'.  

### Difusion-Denoising Models
* Prediction variable and loss: 'epsilon', 'v'
* Sampling algorithm: 'DDPM', 'DDIM'
* Diffusion noise (beta) scheduling: sigmoid, cosine, no scheduling (constant). 
* Parameters: # of diffusion steps 'num_steps', step size of skipped sampling 'skip'.

## Sources
[hojonathanho/diffusion](https://https://github.com/hojonathanho/diffusion): the original paper of the diffusion denoising probabilistic models. 

[karpathy/minGPT](https://github.com/karpathy/minGPT): we referred to this repository's structure to build up the skeleton code mainly for the autoencoder part.

[lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch?tab=readme-ov-file): we referred to this repository to implement the 1D U-Net-based diffusion models by PyTorch. 

## Acknowledgement
This repository is developed together with [Rick Fritschek](https://github.com/Fritschek/). 
