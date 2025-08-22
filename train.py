import os
import argparse
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset
from LGFFM.Model import LGFFM

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser("SAM2-UNet")
parser.add_argument("--hiera_path", type=str, default='./checkpoints/sam2_hiera_large.pt',
                    help="path to the sam2 pretrained hiera")
parser.add_argument("--train_image_path", type=str, default='data/BUSI/all/img.yaml',
                    help="path to the image that used to train the model")
parser.add_argument("--train_mask_path", type=str, default='data/BUSI/all/ann.yaml',
                    help="path to the mask file for training")
parser.add_argument('--save_path', type=str, default='output/BUSI',
                    help="path to store the checkpoint")
parser.add_argument("--epoch", type=int, default=70,
                    help="training epochs")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--batch_size", default=12, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
args = parser.parse_args()


def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def main(args):    
    dataset = FullDataset(args.train_image_path, args.train_mask_path, 256, mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    device = torch.device("cuda")
    model = LGFFM(args.hiera_path)
    model.to(device)
    optim = opt.AdamW([{"params":model.parameters(), "initia_lr": args.lr}], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)
    os.makedirs(args.save_path, exist_ok=True)
    for epoch in range(args.epoch):
        for i, batch in enumerate(dataloader):
            x = batch['image']
            target = batch['label']*255
            x = x.to(device)
            target = target.to(device)
            optim.zero_grad()
            pred = model(x)
            loss = structure_loss(pred, target)
            loss.backward()
            optim.step()
            if i % 20 == 0:
                print("epoch:{}-{}: loss:{}".format(epoch + 1, i + 1, loss.item()))
                
        scheduler.step()
        if (epoch+1) % 20 == 0 or (epoch+1) == args.epoch:
            torch.save(model.state_dict(), os.path.join(args.save_path, 'LGFFM-%d.pth' % (epoch + 1)))
            print('[Saving Snapshot:]', os.path.join(args.save_path, 'LGFFM-%d.pth'% (epoch + 1)))


if __name__ == "__main__":

    main(args)