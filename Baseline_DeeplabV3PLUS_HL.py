import segmentation_models_pytorch as smp
import numpy as np
import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from util.read_data_v3 import SegmentationDataset # read_data_v3 not read_data
import argparse
import random
import os
import monai

import contextual_loss as cl


'''
Using Hybrid Loss (Dice + CL)
'''

torch.manual_seed(777)
np.random.seed(777)
random.seed(777)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args, dataset, model, optimizer, loss, val_metric):

    # split train and val dataset
    length =  dataset.num_of_samples()
    train_size = int(0.8 * length) 
    train_set, validate_set = torch.utils.data.random_split(dataset,[train_size,(length-train_size)]) # manual_seed fixed

    dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    dataloader_val = DataLoader(validate_set, batch_size=args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    # define tensorboard writer
    writer = SummaryWriter()
    batch_num = 0 

    args_dict = args.__dict__
    writer.add_hparams(args_dict, {})

    Context_crit = cl.ContextualLoss(use_vgg=True, vgg_layer='relu5_4',band_width=0.8).to(device) 

    best_metric = -100 # best metric for all trials
    best_metric_batch = -1 # best metric 

    for epoch in range(args.epoch):

        for i_batch, sample_batched in enumerate(dataloader_train):

            model.train()
            batch_num += 1 
            
            # load data
            img, mask = sample_batched['image'], sample_batched['mask']
            
            mask = mask.to(device).float()
            img = img.to(device) 

            optimizer.zero_grad()
            output = model(img) 
            prediction = torch.sigmoid(output)

            ##############################################

            loss_seg_ = loss(input=prediction, target=mask) # focal will use sigmoid in loss function...

            pred_3C = torch.cat((prediction, prediction, prediction), dim=1)
            mask_3C = torch.cat((mask, mask, mask), dim=1)
            
            loss_con = Context_crit(pred_3C, mask_3C)

            loss_seg = loss_seg_ + 0.005 * loss_con
            
            print("loss_con", loss_con)
            print("loss_seg_", loss_seg_)

            ##############################################  

            loss_seg.backward()
            optimizer.step()
    
            writer.add_scalar('loss', loss_seg_.item(), epoch * len(dataloader_train) + i_batch)

            # validation
            if batch_num % (args.val_batch) == 0: 
                model.eval()
                val_scores = []  
                with torch.no_grad():
                    # metric_sum = 0.0
                    # metric_count = 0
                    for i_val, val_data in enumerate(dataloader_val):
                        val_images, val_labels = val_data['image'], val_data['mask']
                        val_images = val_images.to(device)
                        val_labels = val_labels.to(device)

                        val_outputs = model(val_images)
                        # val_outputs = torch.sigmoid(val_outputs) # val_metric close sigmoid

                        value,_ = val_metric(y_pred=val_outputs, y=val_labels)

                        val_scores.append(value.cpu().numpy())
                        
                        # metric_count += len(value)
                        # metric_sum += sum(value)
                        
                    # metric = metric_sum / metric_count

                    print("val_scores", val_scores)
                    metric = np.mean(val_scores)

                    writer.add_scalar('val_metric', metric, batch_num)
                    print('metric:', metric)

                    # save best metric model
                    if metric > best_metric:
                        best_metric = metric
                        best_metric_batch = batch_num
                        torch.save(model.state_dict(), './save_model/best_metric_model_DeepLabV3Plus' + str(round(metric, 2)) +'.pth')
                        print('saved new best metric model')
                    else:
                        print('not saved new best metric model')

                img_grid = torchvision.utils.make_grid(img, nrow=3, padding=2, normalize=False, value_range=None, scale_each=False, pad_value=0)
                writer.add_images('input', img_grid, epoch * len(dataloader_train) + i_batch, dataformats='CHW')
                mask_grid = torchvision.utils.make_grid(mask, nrow=3, padding=2, normalize=False, value_range=None, scale_each=False, pad_value=0)
                writer.add_images('mask', mask_grid, epoch * len(dataloader_train) + i_batch, dataformats='CHW')
                g_output_grid = torchvision.utils.make_grid(prediction, nrow=3, padding=2, normalize=False, value_range=None, scale_each=False, pad_value=0)
                writer.add_images('output', g_output_grid, epoch * len(dataloader_train) + i_batch, dataformats='CHW')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='./data/Basic_Pork/imgs', help='input RGB or Gray image path')
    parser.add_argument('--mask_dir', type=str, default='./data/Basic_Pork/masks', help='input mask path')
    parser.add_argument('--split_ratio', type=float, default='0.8', help='train and val split ratio')

    parser.add_argument('--lr', type=float, default='1e-4', help='learning rate')
    parser.add_argument('--optimizer', type=str, default='Adam', help='RMSprop/Adam/SGD')
    parser.add_argument('--batch_size', type=int, default='8', help='batch_size in training')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--epoch", type=int, default=100, help="epoch in training")

    parser.add_argument("--val_batch", type=int, default=100, help="Every val_batch, do validation")
    parser.add_argument("--save_batch", type=int, default=500, help="Every val_batch, do saving model")

    args = parser.parse_args()

    os.makedirs('./save_model', exist_ok=True)

    model = smp.DeepLabV3Plus(    
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,)                     # a number of channels of output mask
    model = model.to(device)

    dataset = SegmentationDataset(args.image_dir, args.mask_dir, resolution=512)  # 512*512 discriminator also need this size

    # define optimizer
    if args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr = args.lr)
    elif args.optimizer == "Adam": 
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # loss = smp.losses.DiceLoss(mode='binary', log_loss=False, from_logits=False)
    # loss = torch.nn.BCELoss().to(device) # mean
    loss = monai.losses.Dice(sigmoid=False).to(device)

    val_metric = monai.metrics.DiceHelper(sigmoid=True)  # sigmoid + 0.5 threshold

    train(args, dataset, model, optimizer, loss, val_metric)
