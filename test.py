import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import numpy as np
from LGFFM.Model import LGFFM
from dataset import FullDataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default='output/BUSI/LGFFM-10.pth',
                help="path to the checkpoint of LGFFM")
parser.add_argument("--test_image_path", type=str,
                    default='data/BUSI/all/img.yaml',
                    # default='data/thyroid/DDTI/img.yaml',
                    # default='data/Heart/HMC-QU/img.yaml',
                    # default='data/Abdomen/HC18/img.yaml',
                    help="path to the image files for testing")
parser.add_argument("--test_gt_path", type=str,
                    default='data/BUSI/all/ann.yaml',
                    # default='data/thyroid/DDTI/ann.yaml',
                    # default='data/Heart/HMC-QU/ann.yaml',
                    # default='data/Abdomen/HC18/ann.yaml',
                    help="path to the mask files for testing")
parser.add_argument("--save_path", type=str, default='output/',
                    help="path to save the predicted masks")
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = FullDataset(args.test_image_path, args.test_gt_path, 256, mode='train')
test_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=8)

model = LGFFM().to(device)
model.load_state_dict(torch.load(args.checkpoint), strict=True)
model.eval()
model.cuda()
os.makedirs(args.save_path, exist_ok=True)

for i, batch in enumerate(test_loader):
    with torch.no_grad():
        image = batch['image']
        image = image.to(device)
        res = model(image)
        res = res.sigmoid().data.cpu()
        res = res.numpy().squeeze(1)
        binary_pred = (res > 0.5).astype(np.int64)
        res = (res * 255).astype(np.uint8)
        # print("Saving " + name)
        # imageio.imsave(os.path.join(args.save_path, name[:-4] + ".png"), res)

