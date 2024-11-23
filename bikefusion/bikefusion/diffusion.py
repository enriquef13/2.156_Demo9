from diffusers import UNet2DModel
from diffusers import DDPMScheduler
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, random_split
from .pipline import InpaintingDenoisingPipeline as ConditionalDenoisingPipeline
import numpy as np
import matplotlib.pyplot as plt
from .visualizers import visualize_imagesets

class InpaintingDenoisingDiffusion:
    def __init__(self,
                train_dataset,
                validation_dataset,
                image_size = 128,  # the generated image resolution
                n_train_noise_timesteps = 1000,  # the number of timesteps for the noise scheduler
                train_batch_size = 16,
                eval_batch_size = 16,  # how many images to sample during evaluation
                num_epochs = 50,
                gradient_accumulation_steps = 1,
                learning_rate = 5e-5,
                lr_warmup_steps = 500,
                mixed_precision = "fp16",  # `no` for float32, `fp16` for automatic mixed precision
                masking_range = ((24, 104), (0, 128)),  # the range of the random masks applied in training
                minimum_mask_portion = 0.4,  # the minimum portion of the image to be masked
                maximum_mask_portion = 0.8,  # the maximum portion of the image to be masked
                full_image_probability = 0.5,  # the probability of applying a full image mask (no image)
                device = None,  # "cuda" or "cpu"
                model = None,
                ):
        
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.image_size = image_size
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.lr_warmup_steps = lr_warmup_steps
        self.mixed_precision = mixed_precision
        self.device = device
        self.masking_range = masking_range
        self.full_image_probability = full_image_probability
        self.minimum_mask_portion = minimum_mask_portion
        self.maximum_mask_portion = maximum_mask_portion
        self.n_train_noise_timesteps = n_train_noise_timesteps
        
        self.dataloader = DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True)
        self.val_loader = DataLoader(self.validation_dataset, batch_size=self.eval_batch_size, shuffle=False)
        
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if model is not None:
            self.unet = model
        else:
            self.unet =  UNet2DModel(
                    sample_size=self.image_size,  # the target image resolution
                    in_channels=6,  # the number of input channels, 3 for RGB masked images and 3 for RGB noise
                    out_channels=3,  # the number of output channels (RGB)
                    layers_per_block=2,  # how many ResNet layers to use per UNet block
                    block_out_channels=(128, 256, 512, 768),  # the number of output channels for each UNet block
                    down_block_types=(
                        "DownBlock2D", # a regular ResNet downsampling block
                        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                        "AttnDownBlock2D", 
                        "AttnDownBlock2D"
                    ),
                    up_block_types=(
                        "AttnUpBlock2D",# a ResNet upsampling block with spatial self-attention
                        "AttnUpBlock2D",
                        "AttnUpBlock2D",
                        "UpBlock2D"# a regular ResNet upsampling block
                    ),
                ).to(self.device)
        
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=n_train_noise_timesteps)
        self.optimizer = torch.optim.AdamW(self.unet.parameters(), lr=self.learning_rate)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.lr_warmup_steps,
            num_training_steps=(len(self.dataloader) * self.num_epochs),
        )
        self.current_epoch = 0
        
        self.denoiser = ConditionalDenoisingPipeline(self.unet, self.noise_scheduler)
    
    def checkpoints(self, path):
        torch.save({
            "model": self.unet.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.lr_scheduler.state_dict(),
            "epoch": self.current_epoch
        }, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        if "model" not in checkpoint:
            raise ValueError("Checkpoint does not contain a model state dict")
        else:
            self.unet.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            except:
                print("Could not load optimizer state dict")
        if "scheduler" in checkpoint:
            try:
                self.lr_scheduler.load_state_dict(checkpoint["scheduler"])
            except:
                print("Could not load scheduler state dict")
        if "epoch" in checkpoint:
            self.current_epoch = checkpoint["epoch"]
        else:
            self.current_epoch = 0
            print("Epoch information not found in checkpoint setting epoch to 0")
            self.lr_scheduler.last_epoch = 0
    
    def create_random_masks(self, n_masks):
        masks = torch.ones(n_masks, 1, self.image_size, self.image_size).to(self.device)
        height = self.masking_range[0][1] - self.masking_range[0][0]
        width = self.masking_range[1][1] - self.masking_range[1][0]
        mask_heights = np.random.randint(int(height * self.minimum_mask_portion), int(height * self.maximum_mask_portion), n_masks)
        mask_widths = np.random.randint(int(width * self.minimum_mask_portion), int(width * self.maximum_mask_portion), n_masks)
        
        top_positions = np.random.randint(0, height - mask_heights + 1, n_masks)
        left_positions = np.random.randint(0, width - mask_widths + 1, n_masks)
        
        top_positions += self.masking_range[0][0]
        left_positions += self.masking_range[1][0]
        for i in range(n_masks):
            if np.random.rand() < self.full_image_probability:
                masks[i] = 0
                continue
            top = top_positions[i]
            left = left_positions[i]
            mask_height = mask_heights[i]
            mask_width = mask_widths[i]
            
            bottom = top + mask_height
            right = left + mask_width
            
            masks[i, :, top:bottom, left:right] = 0
            
        return masks
    
    def get_sample_batch(self, n_samples, split='Validation'):
        if split == 'Validation':
            rnd_idx = np.random.choice(len(self.validation_dataset), n_samples).astype(int)
            images = torch.stack([self.validation_dataset[i][0] for i in rnd_idx]).to(self.device)
        else:
            rnd_idx = np.random.choice(len(self.train_dataset), n_samples).astype(int)
            images = torch.stack([self.train_dataset[i][0] for i in rnd_idx]).to(self.device)
        masks = self.create_random_masks(n_samples)
        masked_images = images * masks + (1-masks)
        masks = (1-masks)
        return masked_images.cpu().detach().numpy(), masks.cpu().detach().numpy(), images.cpu().detach().numpy()
    
    def get_sample_noising(self, n_samples, n_timesteps=5):
        rnd_idx = np.random.randint(0, len(self.train_dataset), n_samples)
        images = torch.stack([self.train_dataset[i][0] for i in rnd_idx])
        noise = torch.randn(images.shape, device=images.device)
        timesteps = torch.linspace(0, self.n_train_noise_timesteps-1, n_timesteps, device=images.device).long()
        
        noisy_images = torch.zeros((n_samples, n_timesteps, images.shape[1], images.shape[2], images.shape[3]), device=images.device)
        for i in range(n_timesteps):
            noisy_images[:, i] = self.noise_scheduler.add_noise(images, noise, timesteps[i])
        
        return noisy_images
        
    def reset_epoch(self):
        self.current_epoch = 0
        self.lr_scheduler.last_epoch = 0
        
    def train(self, n_epoch=None, display_interval=1, checkpoint_interval=10, checkpoint_path_prefix="checkpoints"):
        if n_epoch is None:
            n_epoch = self.num_epochs
        
        accelerator = Accelerator(
            mixed_precision=self.mixed_precision,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )
        
        
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            self.unet, self.optimizer, self.dataloader, self.lr_scheduler
        )
        
        for epoch in range(self.current_epoch, n_epoch):
            model.train()
            progress = tqdm(train_dataloader)
            for images in progress: 
                images = images[0]
                images = images.to(model.device)
                bs = images.size(0)
                # create random masks
                masks = self.create_random_masks(bs)
                masked_images = images * masks + (1-masks)
                
                noise = torch.randn(images.shape, device=images.device)
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=images.device,
                    dtype=torch.int64
                )
                
                noisy_images = self.noise_scheduler.add_noise(images, noise, timesteps)
                
                with accelerator.accumulate(model):
                    noise_pred = model(torch.cat([masked_images,noisy_images],1), timesteps, return_dict=False)[0]
                    
                    loss = F.mse_loss(noise_pred, noise)
                    accelerator.backward(loss)
                    
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                        
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                progress.set_postfix_str(f"Epoch: {epoch}/{self.num_epochs}, Loss: {loss.item():.7f}")
            
            if epoch % display_interval == 0:
                # unmasks one batch of validation images
                images = next(iter(self.val_loader))[0].to(model.device)[0:4]
                masks = self.create_random_masks(images.size(0))
                masked_images = images * masks + (1-masks)
                filled_images = self.generate(images=masked_images, num_inference_steps=100)
                visualize_imagesets(
                    (images.cpu().detach().numpy() + 1)/2,
                    masks.cpu().detach().numpy(),
                    (masked_images.cpu().detach().numpy() + 1)/2,
                    (filled_images + 1)/2,
                    titles=["Original", "Mask Maps", "Masked", "Filled"]
                )
                
            if epoch % checkpoint_interval == 0:
                self.checkpoints(f"{checkpoint_path_prefix}_{epoch}.pt")
                
            self.current_epoch += 1
    
    def generate(self, images=None, num_inference_steps=100, n_samples=10, noise_seed=None, return_intermediate=False, guidance_function=None, guidance_scale=1.0):
        if images is None and n_samples is None:
            raise ValueError("Either images or n_samples must be provided")
        elif images is None:
            images = torch.ones(n_samples, 3, self.image_size, self.image_size)
        
        initia_noise = None
        if noise_seed is not None:
            torch.manual_seed(noise_seed)
            initia_noise = torch.randn_like(images, device=images.device)
        
        self.unet.eval()
        return self.denoiser(images, num_inference_steps=num_inference_steps, initial_noise=initia_noise, return_intermediate=return_intermediate, guidance_function=guidance_function, guidance_scale=guidance_scale)