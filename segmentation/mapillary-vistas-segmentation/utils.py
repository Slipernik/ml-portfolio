from training import SegLightning, trainer_work, load_ckpt

import os
import time

import torch
from tqdm import tqdm

import segmentation_models_pytorch as smp


def create_class_weight(mask_dl, num_classes):
  weights_path = "class_weight.pt"
  if os.path.exists(weights_path):
    class_weight = torch.load(weights_path, map_location="cuda")
  else:
    pixel_per_class = torch.zeros(num_classes, device="cuda")
    for _, mask in tqdm(mask_dl):
        mask = mask.to("cuda", non_blocking=True)
        if mask.ndim == 4 and mask.shape[1] == 1:
            mask = mask[:, 0]

        counts = torch.bincount(mask.reshape(-1), minlength=num_classes).float()
        pixel_per_class += counts

    freq = pixel_per_class / pixel_per_class.sum().clamp_min(1.0)
    class_weight = 1.0 / (freq + 1e-6)
    class_weight = class_weight / class_weight.mean()
    class_weight = torch.clamp(class_weight, 0.5, 5.0)
    class_weight[14] = 0.0

    torch.save(class_weight.detach().cpu(), weights_path)

  return class_weight

def create_sample_weights(mask_dt, class_weight, num_classes, ignore_index):
  sample_weights = []
  weights_path = "sampler_weight.pt"

  if os.path.exists(weights_path):
    sample_weights = torch.load(weights_path, map_location="cuda")
  else:
    for i in tqdm(range(len(mask_dt))):
      _, mask = mask_dt[i]
      present = torch.bincount(mask.reshape(-1), minlength=num_classes) > 0
      present[ignore_index] = False

      if present.any():
        w = class_weight[present].mean().item()
      else:
        w = 0.0

      sample_weights.append(w)

    sample_weights = torch.tensor(sample_weights, dtype=torch.double)
    torch.save(sample_weights.detach().cpu(), weights_path)

  return sample_weights

def get_metric(
    all_metrics,
    model_name,
    table_name,
    loss_fn,
    train_dl,
    valid_dl,
    num_real_classes,
    num_epochs=50,    
    pivot=None
  ):
  def build_model(model_name, num_real_classes):
    if 'unet' in model_name:
      return smp.Unet(
          encoder_name="efficientnet-b3",
          encoder_weights="imagenet",
          in_channels=3,
          classes=num_real_classes,
      )
    elif 'segformer' in model_name:
      return smp.Segformer(
          encoder_name="mit_b2",
          encoder_weights="imagenet",
          in_channels=3,
          classes=num_real_classes,
      )

  model = build_model(model_name, num_real_classes)
  model = load_ckpt(model_name, model, loss_fn, num_epochs)

  pl_model = SegLightning(model, loss_fn, num_epochs)
  start_time = time.perf_counter()
  metrics = trainer_work(pl_model, model_name,
                         train_dl, valid_dl,
                         num_epochs, train=False)[0]

  metrics['valid_time'] = time.perf_counter() - start_time
  if pivot is not None:
    all_metrics.setdefault(table_name, {})
    all_metrics[table_name][pivot] = metrics
  else:
    all_metrics[table_name] = metrics
  return all_metrics

def tversky_ignore(logits, y_true, tv_loss_fn, num_real_classes, ignore_index):
  valid = (y_true != ignore_index)

  logits_v = logits.permute(0, 2, 3, 1)[valid]
  y_v = y_true[valid]

  return tv_loss_fn(logits_v.view(-1, num_real_classes), y_v.view(-1))
