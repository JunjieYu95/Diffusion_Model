# 2024.1.20 - 2024.1.31
1. Develope seismic path inspired mask
2. test basic matrix completion algorithm

# 2024.2.2
Design potential algoirthm using diffusion model.

using DDPM + repaint

use the floowing nodebook for reference
https://github.com/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb

theory needed to be answered: how the Variational Lower Bound is constructed and how Unet is related to the reparametrization trick.

# 2024.2.3
Finish the unconditional diffusion model for 2D plume dataset.

# 2024.2.13
understand how repiant work
use hugginface diffusers to test repaint

# 2024.2.16
deal with repaint memory issue
resolved, turns out I use the document that refers to diffusers 0.9 but my current version is 0.26.0. To use the pipeline, I should use argument image= instead of original_image=.

# 2024.2.17
There may be error regarding the image scaling, check which scale of image is used for both training and repaint.

the network can be designed to predict the image itself ,but also resiudual image.

Good, following the documentation, I get correct results

# 2024.2.19
It turns out the repaint result has issue regarding the background, the background is not 0, check whether this is scale issue or the property of the algorithm.

1) for the unconditional generator, what's the scale, show the colormap with the same scale (0,1), using matplotlib instead of PIL to avoid any scale issue.

2) for the repaint, check the scale of the image.

# 2024.2.26
Try to solve the scale issue. 
It turns out the generated images also have this scale issue

note: the Norwegian team use latend diffsuion, essentially the mask is part of the input of a network and the hard data is not explicitly horned, that's why they introduce an extra loss to encourage the hard data honoring. Also the use path similar mask, but the state it's observed well log data.

I probably should quickly get some useful DDPM + repaint results especially for the SIS and Fluvial Example, I can put them on airxiv.

Try:
1) Transform all my data to RGB image to be consistent with the huggingface diffusers.
2) Change aritecture
   a. may be related to unit8 issue
   b. increase num_train_steps : 1000 -> 2000, not working
   c. decrease learning rate: 1e-4 -> 1e-5, not working
   c. increase num_res_blocks: 2 -> 4
3) Increase the dataset size


what scale the original examples are using?

```
for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(image, t).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
```
```
 prediction_type (`str`, defaults to `epsilon`, *optional*):
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
            Video](https://imagen.research.google/video/paper.pdf) paper).
```
idea:

I can test on MNIST dataset for testing purpose.
At least we know for example that looks like binar, the diffusion model should have the capability to generate correct background.

# 2024.2.27
It turns out my current training workflow can lead to unstable results (background value issue) even for the MNIST dataset. I need to look into the pipelin and also check my previous training from scratch code.

Thoughts: how to create a tomography example that make the problem more reasonable. 

diffsuion + tomography can be a good example.


# 2024.3.25
It takes a while for job seeking and understand how inverse with score-based model works.

Now let's try using the score-based model to do the reconstruction and check whether the background issue is resolved. 

The score-based model can also be used for the tomography example.

the score model pipline has some small bugs,
let's see based on DDPM, whether we can solve the issue by ajusting parameters.

TRY:
1) Use DDIM: the reverse process of DDIM is much faster than DDPM
2) OPTIMIZER?

CHECK what's the timesteps for training and inference, what's the relationship

when define DDPMpiple, we have num_inference_steps: int = 1000 -> The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

when define the noise schedule: 
num_train_timesteps: int = 1000 -> The number of diffusion steps to train the model.

INCREASE BOTH TO 5000 -> 


remember that the mnist dataset also have black backgound, maybe I can find successfult example for mnsit using diffusers.


# 2024.4.1
let's follow the setup in this 
https://github.com/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb
and test on MNIST, so I can get some insight about we I have the background issue.

let's learn how Git branch works



