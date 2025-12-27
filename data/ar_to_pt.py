from array_record.python import array_record_module  #pyrefly:ignore
import pickle
import grain
from tqdm import tqdm
import os

import torch
print("importing diffusers")
from diffusers.models import AutoencoderKL
print("imported diffusers")


vae = AutoencoderKL.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    subfolder="vae",
    torch_dtype=torch.bfloat16,
)


base_dir = os.path.abspath("data/datasets")

records_per_shard: int = 50000
shard_number: int = 0

for path in tqdm(os.listdir(base_dir)):
    new_path = os.path.join(base_dir, path)
    try:
        array_record_data_source = grain.sources.ArrayRecordDataSource(new_path)
        for data in array_record_data_source:
            element = pickle.loads(data)
            latent = element['latent']
            label = element['label']
            latent = torch.from_numpy(latent).view(torch.bfloat16)
            latent = latent.unsqueeze(0)
            img = vae.decode(latent)[0]
            img = img.squeeze(0).permute(1, 2, 0).clip(0, 1).mult_(255.0).numpy()
            import matplotlib.pyplot as plt
            plt.imshow(img)
            plt.savefig("test.png")
            print(img.shape)
            print(latent.shape, latent.dtype)
            print(label.shape, label.dtype)
            break
            # writer.write(pickle.dumps(element))
            # current_record_count += 1
            # if current_record_count >= records_per_shard:
            #     writer.close()
            #     shard_number += 1                
            #     current_record_count = 0
            #     write_path = f"data/common_canvas_{shard_number}.array_record"
            #     writer = array_record_module.ArrayRecordWriter(write_path, "group_size:1")
    except:
        raise ValueError("invalid record")