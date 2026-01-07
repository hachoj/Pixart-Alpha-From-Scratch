## PixArt-Alpha From Scratch

I originally did this entire project with JAX/Equinox/Grain, that repo can be found here: https://github.com/hachoj/PixelArt-Alpha-Equinox-Grain. I chose to switch to a PyTorch focus again mostly due to adoption.

I quite enjoyed the porcess of using JAX and Equinox especially and in my perfect world that's what I would keep doing. However, for practical reasons, I'm closing myself off from oportunities by specializing so early in my career. I will keep up with JAX so I can switch back if a oportunity needs it gladly.

##### Development Log
*Sumaary of past before I stareted log*: The atrcitecture and training of the stage 1 model was already in place from the jax implementation but I just redid it for practice.

I began trianing the stage2  odel with 14.5 million common canvas images however, I found the training was working poorly.
I then analyzed the dataset more closely, finding that a significant portion of the images, consisted of a random color. This significantly destabilized training, leading to me wanting to search for a new dataset.
To match pixart-$alpha$ I initially wanted to switch to the SAM dataset SA-1B however, I wasn't able to do this, as currently, the dataset is inaccesible.

So instead, I am working with the imagnet 21k dataset, which has a similar number of higher quality images. Further, the captioning I was doing with with the previous dataset was substandard, I changed the way I prompted for captioning with qwen3-2b-instruct and standardized two separate instruction sets such that the captioning of images, and recaptioning of user requests, produce similar outputs.

I also switched to vLLM to serve the local vLLMs for recaptioning and eventually serving the model, however at this point I'm not really sure how the boilerplate works, and had to use the assitence of LLMs for this portion of the code.

Jan 6 2026
I am now compressing the latents and recaptioning the dataset.I also plan on using the 
fact that I have access to B200 GPUs to use fp8 training

To switch to FP8 I have a few things that I need to do. 
1. Write new architecture.
2. Reparameterization to fp8, otherwise, retrain but there's no way that's what I have to do.
3. Autocast for the training.