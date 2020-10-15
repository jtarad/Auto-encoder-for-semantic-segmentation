import argparse
import os
import torch
from torch.utils import data
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from collections import OrderedDict

from metrics import _confusion_matrix, _acc, _cohen_kappa_score

import autoencoder as ae
from sklearn import metrics
import sen2
from torchsummary import summary
from PIL import Image

CHECKPOINT_DIR = './checkpoints/'
NUM_STEPS = 40000
BATCH_SIZE = 8
NUM_CLASSES = 15
INPUT_SIZE = '64,64'
LAMBDA_CE = 1
SHUFFLE = True
VAL_SPLIT = 0.2
RESTORE_FROM = './checkpoints50/'
DATASET_PATH = '/work/stages/taradel/data/S2_10_bandes_11_mois_avec_annotations/T31TDJ/smallSen2_64'

def get_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
  parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of iterations.")
  parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
  parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR,
                        help="Where to save checkpoints of the model.")
  parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
  parser.add_argument("--input-size", type=str, default=INPUT_SIZE)
  parser.add_argument("--lambda-ce", type=float, default=LAMBDA_CE)
  parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                    help="Where restore model parameters from.")
  parser.add_argument("--shuffle", type=bool, default=SHUFFLE)
  parser.add_argument("--validation-split", type=float, default=VAL_SPLIT)
  parser.add_argument("--dataset-path", type=str, default=DATASET_PATH)
  return parser.parse_args()

args = get_arguments()

def get_dataloader(data_loader, data_path, input_size, partitions, purpose):
    imgs_list = []
    for partition in partitions:
        imgs_curr = data_loader(
            **{"dataset_root": data_path,
               "input_sz": input_size[0],
               "gt_k": 15,
               "split": partition,
               "purpose": purpose}  # return testing tuples, image and label
        )
        imgs_list.append(imgs_curr)

    return ConcatDataset(imgs_list)

def main():

  if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

  h, w = map(int, args.input_size.split(','))
  input_size = (h, w)

  #model = ae.autoencoder
  model = nn.Sequential(OrderedDict([
          ('autoencoder', ae.autoencoder),
          ('out_conv', nn.Conv2d(32, 110, 1))
          ])).cuda()

  
  #enc_path = os.path.join(args.restore_from, 'latest_encoder.pth')
  #dec_path = os.path.join(args.restore_from, 'latest_decoder.pth')
  # load pretrained parameters
  #saved_state_dict_enc = torch.load(enc_path)
  #saved_state_dict_dec = torch.load(dec_path)

  #model.encoder.load_state_dict(saved_state_dict_enc)
  #model.decoder.load_state_dict(saved_state_dict_dec)

  model.train()
  
  summary(model, (110,64,64))

  criterion = nn.MSELoss()

  fig, axarr = plt.subplots(1, sharex=False, figsize=(20, 10))

  data_loader = sen2.__dict__["Sen2"]
  data_path = args.dataset_path
  dataset = get_dataloader(data_loader, data_path, input_size, ["labelled_train"], "train")
  #test_dataset = get_dataloader(data_loader, data_path, input_size, ["labelled_test"], "test")

  #test_dataset_size = len(test_dataset)
  #num_batches_test = int(test_dataset_size / args.batch_size) + 1

  dataset_size = len(dataset)
  num_batches_train = int((1-args.validation_split) * dataset_size / args.batch_size) + 1
  num_batches_test = int(args.validation_split * dataset_size / args.batch_size) + 1
  last_batch_sz = (args.validation_split * dataset_size) % args.batch_size

  indices = list(range(dataset_size))
  split = int(np.floor(args.validation_split * dataset_size))
  if args.shuffle :
      #np.random.seed(random_seed)
      np.random.shuffle(indices)
  train_indices, val_indices = indices[split:], indices[:split]

  print('num batches val : ', num_batches_test)
  print('num batches train : ', num_batches_train)
  print('last batch size : ', last_batch_sz)

  # Creating PT data samplers and loaders:
  train_sampler = SubsetRandomSampler(train_indices)
  test_sampler = SubsetRandomSampler(val_indices)

  trainloader = data.DataLoader(dataset,
                  batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
  testloader = data.DataLoader(dataset,
                  batch_size=args.batch_size, sampler=test_sampler, num_workers=4, pin_memory=True)
  #trainloader = data.DataLoader(train_dataset,
  #                batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

  #testloader = data.DataLoader(test_dataset,
  #                batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

  trainloader_iter = iter(trainloader)
  testloader_iter = iter(testloader)

  optimizer = ae.optimizer_ae
  optimizer.zero_grad()

  losses = []
  losses_val = []

  loss_value = 0
  loss_val_value = 0
  e_i = 0
  for i_iter in range(args.num_steps):

      try:
          batch = next(trainloader_iter)
      except:
          print("end epoch %s" % e_i)
          trainloader_iter = iter(trainloader)
          batch = next(trainloader_iter)

      images = batch
      images = images.cuda()
      # ===================forward=====================
      out_ae = model(images)
      loss = criterion(out_ae, images)
      # ===================backward====================
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      loss_value += loss.item()

      # ==================validation===================
      try:
          batch = next(testloader_iter)
      except:
          print("end epoch %s" % e_i)
          testloader_iter = iter(testloader)
          batch = next(testloader_iter)
      img_val = batch
      img_val = img_val.cuda()
      with torch.no_grad():
        out = model(img_val)
      loss_val = criterion(out, img_val)
      loss_val_value += loss_val.item()

      print('iter = {0:8d}/{1:8d}, loss = {2:.3f}'.format(i_iter, args.num_steps, loss_value))

      # ==================save model===================

      if (i_iter % (num_batches_train-1) == 0) and (i_iter > 0) or (i_iter == args.num_steps-1):
          e_i += 1

          avg_loss = loss_value / num_batches_train
          avg_loss_val = loss_val_value / num_batches_test

          loss_value = 0

          losses.append(avg_loss)
          losses_val.append(avg_loss_val)

          axarr.clear()
          axarr.plot(losses, 'b')
          axarr.plot(losses_val, 'm')
          axarr.set_title("MSE loss")      

          fig.canvas.draw_idle()
          fig.savefig(os.path.join(args.checkpoint_dir, "plots.png"))

          if (e_i % 10 == 0) or (i_iter == args.num_steps-1):
            print ('save model ...')
            torch.save(model[0].encoder.state_dict(),os.path.join(args.checkpoint_dir, 'latest_encoder.pth'))
            torch.save(model[0].decoder.state_dict(),os.path.join(args.checkpoint_dir, 'latest_decoder.pth'))
            #break


if __name__ == '__main__':
    main()
