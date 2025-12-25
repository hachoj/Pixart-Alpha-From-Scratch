First, I just read the pixart alpha paper.

Then I worked on implemnting the pixart alpha architecture with equinox. Concurrently I also compressed imagenet1k into latents with sd3 vae.

I then used Grain and Orbax as packages.

Troubles, First, memory usage was way to high but that was because I was loading the PyTorch VAE onto the GPU which apparently was very bad.
Even though it only took up ~2GB of VRAM on its own, when interacting with the Jax script, it was very bad.
I then got the training to work but there are issues. Convergence is not really happen9ing.

I checked and the mean and std even with the scling factor from the config are bad so now I'm trying to implement a new run with some of the 
SD3 flow matching optimizations. `
1. True mixed precision trianing.
2. New scaling and mean shift for image net
3. RMS norm for q and k in attention using jax.scaled_dot_product_attention
4. logit-normal time step sampling.
5. Adding EMA weighted model
   - In the future, I might have to put the EMA model on a separate GPU because it takes up a lot of space on the GPUs just for inference
   - And if the batch size was bigger, it wouldn't fit.


Note for the reparameterization trick, I will use the conditionless embedding as the base line

I completed the reparam trick and made the text cond dit model with t5gemma as the text encoder, I will also already have the training pipeline done, I've sharded the new 15m dataset for the most part with new captions, just need to fill some holes. I'm going to try to use 16 gpus accross 2 nodes for more practice.
