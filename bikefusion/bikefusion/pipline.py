import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

class InpaintingDenoisingPipeline:
    def __init__(self, model, scheduler):
        super().__init__()
        self.model = model
        self.scheduler = scheduler


    @torch.no_grad()
    def __call__(self, masked_images, num_inference_steps=50, guidance_function=None, guidance_scale=1.0, initial_noise=None, return_intermediate=False):
        # Ensure model and scheduler are in evaluation mode
        self.model.eval()

        # Initialize the noisy image as the input image
        device = self.model.device
        masked_images = masked_images.to(device)
        Bs = masked_images.size(0)
        
        if return_intermediate:
            out = torch.zeros((Bs, num_inference_steps+1, 3, masked_images.shape[2], masked_images.shape[3]), device=device)
        
        if initial_noise is not None:
            noise = torch.tensor(initial_noise).float().to(device)
        else:
            noise = torch.randn_like(masked_images, dtype=self.model.dtype).to(self.model.device)
        
        self.scheduler.set_timesteps(num_inference_steps)
        
        if return_intermediate:
            out[:, 0] = noise
            c = 1
            
        for t in tqdm(self.scheduler.timesteps):
            # Get the noise level for this timestep
            model_output = self.model(torch.cat([masked_images,noise],1), t).sample
            
            noise = self.scheduler.step(model_output, t, noise).prev_sample
            
            if guidance_function is not None:
                with torch.enable_grad():
                    noise.requires_grad = True
                    guidance_objective = guidance_function(noise, t)
                    guidance_objective.backward()
                    grads = noise.grad
                noise = noise - guidance_scale * grads
            
            if return_intermediate:
                out[:, c] = noise
                c += 1
        
        if return_intermediate:
            return out.clamp(-1,1).detach().cpu().numpy()
        else:
            return noise.clamp(-1,1).detach().cpu().numpy()