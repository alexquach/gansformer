# Generate images using pretrained network pickle.
import os
import numpy as np
import PIL.Image
from tqdm import trange 
import argparse

import dnnlib
import torch
import loader
from tqdm import tqdm

import warnings
from training import misc
from training.misc import crop_max_rectangle as crop

# Generate images using pretrained network pickle.
def run(model, gpus, output_dir, images_num, truncation_psi, ratio, images_dir, vits_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus                             # Set GPUs
    device = torch.device("cuda")

    # Load dataset
    dataset_args = dnnlib.EasyDict(
        class_name     = "training.dataset.ImageFolderDataset", 
        path           = images_dir,
        max_items      = 500, 
        resolution     = 128,
        ratio          = 1,
    )
    dataset = dnnlib.util.construct_class_by_name(**dataset_args) # subclass of training.datasetDataset

    print("Loading networks...")
    G = loader.load_network(model, eval = True)["Gs"].to(device)          # Load pre-trained network

    print("Generate and save images...")
    os.makedirs(output_dir, exist_ok = True)                              # Make output directory

    if images_dir is not None:
        if vits_path is None:
            vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to(device)
        else:
            vits16 = torch.load(vits_path).to(device)
        for parameter in vits16.parameters():
            parameter.requires_grad = False

        images, labels = zip(*[dataset[i] for i in range(images_num)])
        images = torch.Tensor(np.stack(images)).to(device)

        images = misc.adjust_range(images, [0, 255], [-1,1])

        warnings.filterwarnings("ignore", category=UserWarning)
        latents = vits16(images).reshape(-1, 16, 24)
        warnings.filterwarnings("default", category=UserWarning)
    else:
        latents = np.random.randn(images_num, *G.input_shape[1:])  # Sample latent vectors
    generated_images = G(latents, truncation_psi = truncation_psi)[0]   # Generate images

    gen_pattern = "{}/sample_{{:06d}}.png".format(output_dir)   # Output images pattern
    src_pattern = "{}/sample_{{:06d}}_source.png".format(output_dir)       # Output images pattern
    for i, generated_image in tqdm(list(enumerate(generated_images))):              # Save images
        crop(misc.to_pil(generated_image.cpu().numpy()), ratio).save(gen_pattern.format(i))
    if images_dir is not None:
        for i, image in tqdm(list(enumerate(images))):
            crop(misc.to_pil(image.cpu().numpy()), ratio).save(src_pattern.format(i))

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description = "Generate images with the GANformer")
    parser.add_argument("--model",              help = "Filename for a snapshot to resume", type = str)
    parser.add_argument("--gpus",               help = "Comma-separated list of GPUs to be used (default: %(default)s)", default = "0", type = str)
    parser.add_argument("--output-dir",         help = "Root directory for experiments (default: %(default)s)", default = "images", metavar = "DIR")
    parser.add_argument("--images-num",         help = "Number of images to generate (default: %(default)s)", default = 32, type = int)
    parser.add_argument("--truncation-psi",     help = "Truncation Psi to be used in producing sample images (default: %(default)s)", default = 0.7, type = float)
    parser.add_argument("--ratio",              help = "Crop ratio for output images (default: %(default)s)", default = 1.0, type = float)
    # Pretrained models' ratios: CLEVR (0.75), Bedrooms (188/256), Cityscapes (0.5), FFHQ (1.0)

    # Options for mode with image inputs
    parser.add_argument("--images-dir",         help = "Root directory for input images (default: %(default)s)", default = None, type = str)
    parser.add_argument("--vits-path",          help = "Path to the VITS dataset (default: %(default)s)", default = None, type = str)
    args, _ = parser.parse_known_args()
    run(**vars(args))

if __name__ == "__main__":
    main()
