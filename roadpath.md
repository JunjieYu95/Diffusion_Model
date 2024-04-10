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

let's learn how Git branch works -> done

following the note book I got good answer.

note:
scale issue: [-1,1], [0,1]
initial issue: rand vs randn
generation issue: follow the notebook or follow the default inference

explore:
why I can generate different dimension when I set the output dimension is (20,68)
e.g. I can generate 100*100 image like this ![flexible_100_100](roadpath_images/flexible_100_100.png)

for channel dataset. (64,64) -> (32,32)
![channel](roadpath_images/channel_100_100.png)

I think I found the reason, because the unet is all operated by CNN and there is no flatten operation, so the output dimension is not fixed, because all the learnable parameters are assigned within kernel.

Now the intereting question is, how the pattern is change when the output dimension is changed.

# 2024.4.2

Get results:

note:
1) the notebook is for motivation, for example:
   a. it use rand instead of randn
   b. it minimize the mismatch between image instead of noise
   c. it iteratively add noise using f for loop

The more stable version should be following the DDPM pipeline, but I get background scale issue.
Why this issue not happen in the notebook appraoch?

2) Need to check:
   1) what's the input output, we I use net.sample, what is predicted
   2) if I want to access the predicted noise, how to do it

# 2024.4.3
compare the package with the notebook,

# 2024.4.4
Now I understand the diffuser package better, it turns out the appraoch I use is correct, since I can train reasonable model on MNIST dataset.
![alt text](roadpath_images/MNIST_good.png)

which means there is no fundamental issue with my appraoch, I may need to perform severl experiments to find the best hyperparameters.

Question:
It seems turns out minimize mismatch between clean image and denoised image perform better than minimize mismatch between noise and predicted noise in my Plume case, is there any reason for this?

Is it possible because 

Damn: I found it, just epoch is not enough.

next step:
1) train model on geo dataset
2) test and get results of repainting on plume dataset

# 2024.4.7
Test the performance of repaint algorithm.

Observation: the performance of repaint
patch_based_mask > ray path > random pixel

test:
1) patch_based_mask
   a. different pacth size
   b. fifferent mask ratio

# 2024.4.9
sensitivity of the n_jump
20
![20](<DDPM/Geo_examples/repaint_patch_folder/jump_n_20_patch_size_(3, 3)_mask_ratio_0.02.png>)
10
![10](<DDPM/Geo_examples/repaint_patch_folder/jump_n_10_patch_size_(3, 3)_mask_ratio_0.02.png>)
5
![5](<DDPM/Geo_examples/repaint_patch_folder/jump_n_5_patch_size_(3, 3)_mask_ratio_0.02.png>)


