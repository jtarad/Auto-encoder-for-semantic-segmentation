import numpy as np
import torch

from sklearn import metrics

def _confusion_matrix(preds, targets, num_k):
  target_mask = (targets != -1)
  targets_filtered = targets[target_mask].cpu()
  preds_filtered = preds[target_mask].cpu()
  return metrics.confusion_matrix(targets_filtered, preds_filtered, labels=(range(num_k)))

def _acc(preds, targets, num_k):
  assert (isinstance(preds, torch.Tensor) and
          isinstance(targets, torch.Tensor) and
          preds.is_cuda and targets.is_cuda)
  assert (preds.shape == targets.shape)
  target_mask = (targets != -1)

  targets = targets[target_mask]
  preds = preds[target_mask]
  assert (preds.shape == targets.shape)
  acc = int((preds == targets).sum()) / float(preds.shape[0])

  return acc

def _cohen_kappa_score(preds, targets):
  target_mask = (targets != -1)
  targets = targets[target_mask]
  preds = preds[target_mask]
  return metrics.cohen_kappa_score(preds.cpu(), targets.cpu())
