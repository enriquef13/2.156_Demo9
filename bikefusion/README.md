---
license: mit
---
This model has been trained to generate bike images. The dataset used is BIKED++ (found here in [GitHub](https://github.com/Lyleregenwetter/BIKED_multimodal/tree/main?utm_source=catalyzex.com))

This is a conditional diffusion model which is trained to both generate bike images and perform infill on partially masked bike images from the dataset above.

the baseline architecture was setup with diffusers UNet2DModel model for images:

```python
UNet2DModel(
            sample_size=128,  # the target image resolution is set to 128 here (128x128 images)
            in_channels=6,  # the number of input channels, 3 for RGB masked images(infill, feed all white for uncoditional) and 3 for RGB noise
            out_channels=3,  # the number of output channels (RGB)
            layers_per_block=2,  
            block_out_channels=(128, 256, 512, 768),  
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D", 
                "AttnDownBlock2D"
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D"
            ),
        )
```

The code for training and inference is included as well. There exists a postprocessing function which crops the white space above and below the bike images which the dataset was preprocessed to have such that the images become square shaped.

With only 10 denoising steps you can get uncoditional samples that mimic the dataset well.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/64d516ba80d47a6b76fc1015/VDqGqLee0XvqtiYYkzpia.png)

A pipeline with guidance is provided as well where you can feed your custom function for guidance.
