import argparse
import os
import torch
from torch.utils import data
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

from loss import CrossEntropy2d
from metric import _confusion_matrix, _acc, _cohen_kappa_score

import autoencoder as ae
from sklearn import metrics
import sen2
from torchsummary import summary
from PIL import Image

CHECKPOINT_DIR = './checkpoints/'
NUM_STEPS = 40000
BATCH_SIZE = 100
NUM_CLASSES = 15
INPUT_SIZE = '64,64'
IGNORE_LABEL = -1
LAMBDA_CE = 1
RESTORE_FROM = './checkpoints50/'
DATASET_PATH = '/work/stages/taradel/data/S2_10_bandes_11_mois_avec_annotations/T31TDJ/dataSentinel2_64'

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
  parser.add_argument("--ignore-label", type=float, default=IGNORE_LABEL,
                    help="label value to ignored for loss calculation")
  parser.add_argument("--lambda-ce", type=float, default=LAMBDA_CE)
  parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                    help="Where restore model parameters from.")
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

def _eval(model, dataset, batch_size, input_size, num_batches, gpu, rend):

    torch.cuda.empty_cache()
    model.eval()
    print("EVAL")

    dir_rend = os.path.join(args.checkpoint_dir,'rend')
    if not os.path.exists(dir_rend):
      os.makedirs(dir_rend)

    samples_per_batch = batch_size * input_size[0] * input_size[1]
    flat_predss_all = torch.zeros((num_batches * samples_per_batch),
                                 dtype=torch.uint8).cuda()
    flat_labels_all = torch.zeros((num_batches * samples_per_batch),
                                 dtype=torch.uint8).cuda()
    num_samples = 0
    colour_map = [(np.random.rand(3) * 255.).astype(np.uint8)
                  for _ in range(15)]
    for b_i, batch in enumerate(dataset):
        # print("Loading data batch %s" % b_i)
        imgs, labels = batch
        imgs = imgs.cuda()

        with torch.no_grad():
          out_dec, out_seg = model(imgs)
        n, c, w, h = out_seg.shape

        actual_samples_curr = n * w * h
        num_samples += actual_samples_curr
        start_i = b_i * samples_per_batch
        flat_preds_curr = torch.argmax(out_seg, dim=1)

        flat_predss_all[start_i:(start_i + actual_samples_curr)] = flat_preds_curr.view(-1)
        flat_labels_all[start_i:(start_i + actual_samples_curr)] = labels.view(-1)

        if rend and b_i < 5:
          seg_preds = np.array(out_seg.cpu(), dtype=np.uint8)
          preds = np.array(((out_dec[:,1:4,:,:])*3000.).permute(0,2,3,1).cpu(), dtype=np.uint8)
          ims = np.array(((imgs[:,1:4,:,:])*3000.).permute(0,2,3,1).cpu(), dtype=np.uint8)
          pred_curr = np.zeros((64,64,3),dtype=np.uint8)
          im_curr = np.zeros((64,64,3),dtype=np.uint8)
          for i in range(n):
            pred_curr = preds[i,:,:,:][...,::-1]
            im_curr = ims[i,:,:,:][...,::-1]
            Image.fromarray(pred_curr).save(os.path.join(dir_rend,'pred_'+str(b_i)+'_'+str(i)+'.png'))
            Image.fromarray(im_curr).save(os.path.join(dir_rend,'img_'+str(b_i)+'_'+str(i)+'.png'))

            #img_rend = np.zeros((w,h,3), dtype=np.uint8)
            #lbl_rend = np.zeros((w,h,3), dtype=np.uint8)
            #for ch in range(15):
            #  img_rend[flat_preds_curr[i,:,:] == ch, :] = colour_map[ch]
            #  lbl_rend[labels[i,:,:] == ch, :] = colour_map[ch]
            #Image.fromarray(img_rend).save(os.path.join(dir_rend, 'seg_'+str(b_i)+'_'+str(i)+'.png'))
            #Image.fromarray(lbl_rend).save(os.path.join(dir_rend, 'gt_'+str(b_i)+'_'+str(i)+'.png'))

    flat_predss_all = flat_predss_all[:num_samples]
    flat_labels_all = flat_labels_all[:num_samples]

    acc = _acc(flat_predss_all, flat_labels_all, c)
    kappa_score = _cohen_kappa_score(flat_predss_all, flat_labels_all)
    cm = _confusion_matrix(flat_predss_all, flat_labels_all, c)
    
    model.train()
    torch.cuda.empty_cache()
    return acc, kappa_score, cm

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.PuRd):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def criterion_seg(pred, label, weights, gpu):
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d(ignore_label=args.ignore_label).cuda(gpu)  # Ignore label ??
    return criterion(pred, label, weights)

def main():

  if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

  h, w = map(int, args.input_size.split(','))
  input_size = (h, w)

  #model = ae.autoencoder
  model = ae.autoencoder2heads.cuda()
  
  enc_path = os.path.join(args.restore_from, 'latest_encoder.pth')
  dec_path = os.path.join(args.restore_from, 'latest_decoder.pth')
  # load pretrained parameters
  saved_state_dict_enc = torch.load(enc_path)
  saved_state_dict_dec = torch.load(dec_path)

  model.encoder.load_state_dict(saved_state_dict_enc)
  model.decoder.load_state_dict(saved_state_dict_dec)

  model.train()
  
  summary(model, (110,64,64))

  criterion_dec = nn.MSELoss()

  fig, axarr = plt.subplots(5, sharex=False, figsize=(20, 20))
  
  p_class = [7.25703402e-02, 1.57180553e-01, 1.81395714e-01, 2.15331438e-01,
  8.59744781e-02, 6.45834114e-02, 2.08535688e-03, 2.95754679e-02,
  2.30909954e-02, 1.18364523e-03, 5.40670110e-04, 4.34120229e-02,
  1.03664125e-01, 2.07385748e-03, 1.73379244e-02]
 
  weights = (1/torch.log(1.02 + torch.tensor(p_class))).cuda()

  data_loader = sen2.__dict__["Sen2"]
  data_path = args.dataset_path
  train_dataset = get_dataloader(data_loader, data_path, input_size, ["labelled_train"], "train_sup")
  test_dataset = get_dataloader(data_loader, data_path, input_size, ["labelled_test"], "test")

  test_dataset_size = len(test_dataset)
  num_batches_test = int(test_dataset_size / args.batch_size) + 1

  train_dataset_size = len(train_dataset)
  num_batches_train = int(train_dataset_size / args.batch_size) + 1
  last_batch_sz = train_dataset_size % args.batch_size
  
  print('num batches test : ', num_batches_test)
  print('num batches train : ', num_batches_train)
  print('last batch size : ', last_batch_sz)
  
  trainloader = data.DataLoader(train_dataset,
                  batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

  testloader = data.DataLoader(test_dataset,
                  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

  trainloader_iter = iter(trainloader)

  optimizer = ae.optimizer_s2h
  optimizer.zero_grad()

  train_accs = []
  test_accs = []
  train_kscores = []
  test_kscores = []
  losses = []
  losses_ce = []
  losses_mse = []

  loss_value = 0
  loss_ce_value = 0
  loss_mse_value = 0
  e_i = 0
  for i_iter in range(args.num_steps):

      try:
          batch = next(trainloader_iter)
      except:
          print("end epoch %s" % e_i)
          trainloader_iter = iter(trainloader)
          batch = next(trainloader_iter)

      images, labels = batch
      images = images.cuda()
      # ===================forward=====================
      out_ae, out_seg = model(images)
      loss_ce = criterion_seg(out_seg, labels, weights, args.gpu)
      loss_mse = criterion_dec(out_ae, images)
      loss = loss_mse + args.lambda_ce*loss_ce
      # ===================backward====================
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      loss_value += loss.item()
      loss_mse_value += loss_mse.item()
      loss_ce_value += loss_ce.item()

      print('iter = {0:8d}/{1:8d}, loss = {2:.3f}'.format(i_iter, args.num_steps, loss_value))

      # ==================save model===================

      if (i_iter % (num_batches_train-1) == 0) and (i_iter > 0) or (i_iter == args.num_steps-1):
          e_i += 1

          rend = (e_i % 10 == 0)
          
          train_acc, train_kappa_score, train_cm = _eval(model, trainloader, args.batch_size, input_size, num_batches_train, args.gpu, rend)
          test_acc, test_kappa_score, test_cm = _eval(model, testloader, args.batch_size, input_size, num_batches_test, args.gpu, rend)

          print('train_acc = ',train_acc)
          print('test_acc = ',test_acc)

          x = e_i + 10
          cm_name_train = "confusion_matrix_%d.png" % x
          plot_confusion_matrix(cm=train_cm,classes=(range(args.num_classes)), normalize=True)   
          plt.savefig(os.path.join(args.checkpoint_dir, cm_name_train))

          avg_loss = loss_value / num_batches_train
          avg_loss_ce = loss_ce_value / num_batches_train
          avg_loss_mse = loss_mse_value / num_batches_train

          loss_value = 0
          loss_ce_value = 0
          loss_mse_value = 0

          train_accs.append(train_acc)
          test_accs.append(test_acc)
          train_kscores.append(train_kappa_score)
          test_kscores.append(test_kappa_score)
          losses.append(avg_loss)
          losses_ce.append(avg_loss_ce)
          losses_mse.append(avg_loss_mse)
          
          axarr[0].clear()
          axarr[0].plot(train_accs, 'g')
          axarr[0].plot(test_accs, 'r')
          axarr[0].set_title("acc (best), top train : %f" % max(train_accs))

          axarr[1].clear()
          axarr[1].plot(train_kscores, 'g')
          axarr[1].plot(test_kscores, 'r')
          axarr[1].set_title("Cohen's kappa score (best) : %f" % max(train_kscores))

          axarr[2].clear()
          axarr[2].plot(losses)
          axarr[2].set_title("Total loss")

          axarr[3].clear()
          axarr[3].plot(losses_mse)
          axarr[3].set_title("MSE loss")
          
          axarr[4].clear()
          axarr[4].plot(losses_ce)
          axarr[4].set_title("CE loss")          

          fig.canvas.draw_idle()
          fig.savefig(os.path.join(args.checkpoint_dir, "plots_suite.png"))

          #if i_iter == args.num_steps-1:
          if e_i % 10 == 0:
            print ('save model ...')
            torch.save(model.encoder.state_dict(),os.path.join(args.checkpoint_dir, 'latest_encoder.pth'))
            torch.save(model.decoder.state_dict(),os.path.join(args.checkpoint_dir, 'latest_decoder.pth'))
            #break


if __name__ == '__main__':
    main()
