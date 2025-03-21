from translator import MyOptim
from bicubic import BicubicDownSample
import os.path
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from pathlib import Path
import torch
from PIL import Image
import torchvision
from math import log10, ceil
import argparse
# import wandb
# torch.autograd.set_detect_anomaly(True)

class Images(Dataset):
    def __init__(self, root_dir, duplicates):
        self.root_path = Path(root_dir)
        self.image_list = list(self.root_path.glob("*.jpg"))
        self.duplicates = duplicates  # Number of times to duplicate the image in the dataset to produce multiple HR images

    def __len__(self):
        return self.duplicates * len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx // self.duplicates]
        image = torchvision.transforms.ToTensor()(Image.open(img_path))
        image = torchvision.transforms.Resize((512, 512))(image)
        if (self.duplicates == 1):
            return image, img_path.stem
        else:
            return image, img_path.stem + f"_{(idx % self.duplicates) + 1}"


parser = argparse.ArgumentParser(description='MyOptim')

# I/O arguments
parser.add_argument('-input_dir', type=str, default='input', help='input data directory')
parser.add_argument('-output_dir', type=str, default='runs', help='output data directory')
parser.add_argument('-cache_dir', type=str, default='cache', help='cache directory for model weights')
parser.add_argument('-duplicates', type=int, default=1,
                    help='How many HR images to produce for every image in the input directory')
parser.add_argument('-batch_size', type=int, default=1, help='Batch size to use during optimization')
parser.add_argument('-seed', type=int, help='manual seed to use')
parser.add_argument('-loss_str', type=str, default="100*L2+0.05*GEOCROSS+1*L1_CLIP",
                    help='Loss function to search in sphere')
parser.add_argument('-loss_str2', type=str, default="1*L1_MLP_CLIP", help='Loss function to train FC')
parser.add_argument('-eps', type=float, default=2e-3, help='Target for downscaling loss (L2)')
parser.add_argument('-noise_type', type=str, default='trainable', help='zero, fixed, or trainable')
parser.add_argument('-num_trainable_noise_layers', type=int, default=5, help='Number of noise layers to optimize')
parser.add_argument('-tile_latent', action='store_true', help='Whether to forcibly tile the same latent 18 times')
parser.add_argument('-bad_noise_layers', type=str, default="17",
                    help='List of noise layers to zero out to improve image quality')
parser.add_argument('-opt_name', type=str, default='adam', help='Optimizer to use in projected gradient descent')
parser.add_argument('-learning_rate', type=float, default=0.4, help='Learning rate to use during optimization')
parser.add_argument('-steps', type=int, default=100, help='Number of optimization steps')
parser.add_argument('-fc_every', type=int, default=1, help='Number of optimization steps')
parser.add_argument('-lr_schedule', type=str, default='linear1cycledrop', help='fixed, linear1cycledrop, linear1cycle')
parser.add_argument('-save_intermediate', action='store_true',
                    help='Whether to store and save intermediate HR and LR images during optimization')

kwargs = vars(parser.parse_args())
# torch.autograd.set_detect_anomaly = True
dataset = Images(kwargs["input_dir"], duplicates=kwargs["duplicates"])
out_path = Path(kwargs["output_dir"])
out_path.mkdir(parents=True, exist_ok=True)

sig_path = Path(out_path / "single")
sig_path.mkdir(parents=True, exist_ok=True)
# _, exp = os.path.split(kwargs["output_dir"])
# wandb.init(project=exp)

dataloader = DataLoader(dataset, batch_size=kwargs["batch_size"])

model = MyOptim(cache_dir=kwargs["cache_dir"]).cuda()
toPIL = torchvision.transforms.ToPILImage()
D_4 = BicubicDownSample(factor=4)
D_2 = BicubicDownSample(factor=2)
num = 0


for ref_im, ref_im_name in dataloader:

    ref_im = ref_im.cuda()
    if (kwargs["save_intermediate"]):
        padding = ceil(log10(100))
        for i in range(kwargs["batch_size"]):
            int_path_HR = Path(out_path / "intermediate")
            int_path_HR.mkdir(parents=True, exist_ok=True)
            int_path_HR_single = Path(int_path_HR / "single")
            int_path_HR_single.mkdir(parents=True, exist_ok=True)

        # for j, (HR, LR) in enumerate(model(ref_im,ref_im_name, **kwargs)):
        for j, (HR,) in enumerate(model(ref_im,ref_im_name, **kwargs)):
            for i in range(kwargs["batch_size"]):
                with torch.no_grad():
                    sample = torch.cat([
                                        # LR[i].unsqueeze(0).cpu().detach(),
                                        HR[i].unsqueeze(0).cpu().detach(),
                                        ref_im[i].unsqueeze(0).cpu().detach()],
                                       dim=0)
                    torchvision.utils.save_image(
                        sample,
                        f"%s/{ref_im_name[i]}_{j:0{padding}}.png" % (int_path_HR),
                        range=(-1, 1),
                    )

                    sample1 = HR[i].unsqueeze(0)#.cpu().detach(),
                    torchvision.utils.save_image(
                        sample1,
                        f"%s/single/{ref_im_name[i]}_{j:0{padding}}.png" % (int_path_HR),
                    )
    else:
        print(kwargs['seed'])
        # for j, (HR, LR) in enumerate(model(ref_im,ref_im_name, **kwargs)):
        for j, (HR,) in enumerate(model(ref_im,ref_im_name, **kwargs)):
            for i in range(kwargs["batch_size"]):
                with torch.no_grad():
                    sample = torch.cat([
                                        # [LR[i].unsqueeze(0).cpu().detach(),
                                        HR[i].unsqueeze(0).cpu().detach(),
                                        ref_im[i].unsqueeze(0).cpu().detach()],
                                       dim=0)
                    torchvision.utils.save_image(
                        sample,
                        f"%s/{ref_im_name[i]}.png" % (out_path),
                        range=(-1, 1),
                    )
                    sample1 = HR[i].unsqueeze(0)#.cpu().detach(),
                    torchvision.utils.save_image(
                        sample1,
                        f"%s/single/{ref_im_name[i]}.png" % (out_path),
                    )
    num += 1
