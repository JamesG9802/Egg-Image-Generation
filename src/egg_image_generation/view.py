"""Visualizes images from the model.
"""
import logging
from typing import Final, List, Tuple

from torchvision.utils import make_grid
from matplotlib import pyplot as plt
import torch
from train import ConvGenerator, Generator
from view_get_args import get_args

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%I:%M:%S %p')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

def main():
    args = get_args()
    model_path: str = args.model
    z_size: int = args.z
    image_sizes: Tuple[int, int] = args.image_size
    s: int|None = args.seed
    class_count: int = 2
    torch.manual_seed(s if s != None else 0)

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    generator_layer_size: List[int] = [256, 512, 1024]
    generator: Generator = Generator(generator_layer_size, z_size, (3, *image_sizes), class_count).to(device)
    
    # generator_layer_size: List[int] = [64, 32]
    # generator: ConvGenerator = ConvGenerator(generator_layer_size, z_size, (3, *image_sizes), class_count).to(device)

    
    checkpoint = torch.load(model_path)
    generator.load_state_dict(checkpoint['G'])

    with torch.no_grad():
        z = torch.randn(class_count, z_size, device=device)
        labels = torch.arange(class_count, device=device)
        sample_images = generator(z, labels)
        grid = make_grid(sample_images, nrow=class_count//2, normalize=True)
        plt.imshow(grid.permute(1,2,0).cpu()); plt.axis('off'); plt.show()

if __name__ == "__main__":
    exit(main())