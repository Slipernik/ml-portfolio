import pathlib
from pathlib import Path

import torch
import torch.nn as nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from pytorch_lightning.loggers import TensorBoardLogger


class SelfMadeUNet(nn.Module):
  """
  U-Net декодер поверх CNN-backbone (backbone.features).
  Скипы берутся из encoder-стадий backbone, апсемплинг — ConvTranspose2d.
  """

  def __init__(self, out_ch, backbone, layers=[6, 13, 26, 39]):
    """
    Args:
        out_ch (int): число выходных каналов (классов)
        backbone (nn.Module): модель с backbone.features
        layers (tuple[int,int,int,int]): индексы границ encoder-блоков/пулинга
        в features
    """
    super().__init__()
    self.backbone = backbone

    # Encoder (разрезаем backbone.features на блоки + pool-слои между ними)
    self.enc1 = self.backbone.features[:layers[0]]
    self.pool1 = self.backbone.features[layers[0]]
    self.enc2 = self.backbone.features[layers[0]+1:layers[1]]
    self.pool2 = self.backbone.features[layers[1]]
    self.enc3 = self.backbone.features[layers[1]+1:layers[2]]
    self.pool3 = self.backbone.features[layers[2]]
    self.enc4 = self.backbone.features[layers[2]+1:layers[3]]
    self.pool4 = self.backbone.features[layers[3]]

    self.bottleneck = self.backbone.features[40:52]

    # Decoder (апсемплинг + concat skip + conv_block)
    self.up_conv1 = nn.ConvTranspose2d(512, 512, 2, stride=2)
    self.conv1 = self.conv_block(1024, 512)
    self.up_conv2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
    self.conv2 = self.conv_block(512, 256)
    self.up_conv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
    self.conv3 = self.conv_block(256, 128)
    self.up_conv4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
    self.conv4 = self.conv_block(128, 64)
    self.conv5 = nn.Conv2d(64, out_ch, 1)

  def forward(self, x):
    # Encoder + skip features
    x1 = self.enc1(x)
    out = self.pool1(x1)

    x2 = self.enc2(out)
    out = self.pool2(x2)

    x3 = self.enc3(out)
    out = self.pool3(x3)

    x4 = self.enc4(out)
    out = self.pool4(x4)

    bn = self.bottleneck(out)

    # Decoder: up -> concat(skip) -> conv
    out = self.up_conv1(bn)
    x4 = torch.concat((x4, out), dim=1)
    out = self.conv1(x4)

    out = self.up_conv2(out)
    x3 = torch.concat((x3, out), dim=1)
    out = self.conv2(x3)

    out = self.up_conv3(out)
    x2 = torch.concat((x2, out), dim=1)
    out = self.conv3(x2)

    out = self.up_conv4(out)
    x1 = torch.concat((x1, out), dim=1)

    out = self.conv4(x1)
    out = self.conv5(out)

    return out

  def conv_block(self, in_ch, out_ch):
    """(Conv-BN-ReLU) x2"""
    return nn.Sequential(
      nn.Conv2d(in_ch, out_ch, 3, padding=1),
      nn.BatchNorm2d(out_ch),
      nn.ReLU(),
      nn.Conv2d(out_ch, out_ch, 3, padding=1),
      nn.BatchNorm2d(out_ch),
      nn.ReLU(),
    )

class SegLightning(pl.LightningModule):
  """
  LightningModule для мультиклассовой сегментации + метрики (acc/iou/dice).
  """

  def __init__(self, model, loss_fn, num_epochs, lr=1e-4,
               num_classes=None, ignore_index=None):
    super().__init__()
    self.model = model
    self.loss_fn = loss_fn
    self.num_epochs = num_epochs
    self.lr = lr
    self.num_classes = num_classes
    self.ignore_index = getattr(loss_fn, "ignore_index", ignore_index)

    # smp.metrics ожидает tp/fp/fn/tn (по классам),
    # ниже суммируем по batch и усредняем по классам
    self.accuracy = lambda tp, fp, fn, tn:\
                    smp.metrics.accuracy(tp, fp, fn, tn)
    self.iou_score = lambda tp, fp, fn, tn:\
                     smp.metrics.iou_score(tp, fp, fn, tn)
    self.dice_score = lambda tp, fp, fn, tn:\
                      smp.metrics.f1_score(tp, fp, fn, tn)

  def forward(self, x):
    return self.model(x)

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=self.num_epochs)
    return [optimizer], [{'scheduler': scheduler}]

  def training_step(self, batch, batch_idx):
    x, y_true = batch
    logits = self.model(x)

    # Если модель возвращает dict (deeplabv3 -> out + aux)
    if isinstance(logits, dict):
      loss_main = self.loss_fn(logits['out'], y_true)
      loss_aux = self.loss_fn(logits['aux'], y_true)

      loss = loss_main + 0.4 * loss_aux
      logits = logits['out']
    else:
      loss = self.loss_fn(logits, y_true)

    y_pred = logits.argmax(dim=1)
    num_classes = self.num_classes or logits.shape[1]

    tp, fp, fn, tn = smp.metrics.get_stats(y_pred, y_true,
                                           mode="multiclass",
                                           num_classes=num_classes,
                                           ignore_index=self.ignore_index)

    acc = self.accuracy(tp.sum(0), fp.sum(0), fn.sum(0), tn.sum(0)).mean()
    iou = self.iou_score(tp.sum(0), fp.sum(0), fn.sum(0), tn.sum(0)).mean()
    dice = self.dice_score(tp.sum(0), fp.sum(0), fn.sum(0), tn.sum(0)).mean()

    self.log('train_loss', loss, on_epoch=True, prog_bar=True)
    self.log('train_acc', acc, on_epoch=True, prog_bar=True)
    self.log('train_iou', iou, on_epoch=True, prog_bar=True)
    self.log('train_dice', dice, on_epoch=True, prog_bar=True)

    return loss

  def validation_step(self, batch, batch_idx):
    x, y_true = batch
    logits = self.model(x)

    if isinstance(logits, dict):
      loss_main = self.loss_fn(logits['out'], y_true)
      loss_aux = self.loss_fn(logits['aux'], y_true)

      loss = loss_main + 0.4 * loss_aux
      logits = logits['out']
    else:
      loss = self.loss_fn(logits, y_true)

    y_pred = logits.argmax(dim=1)
    num_classes = self.num_classes or logits.shape[1]

    tp, fp, fn, tn = smp.metrics.get_stats(y_pred, y_true,
                                           mode="multiclass",
                                           num_classes=num_classes,
                                           ignore_index=self.ignore_index)

    acc = self.accuracy(tp.sum(0), fp.sum(0), fn.sum(0), tn.sum(0)).mean()
    iou_pc = self.iou_score(tp.sum(0), fp.sum(0), fn.sum(0), tn.sum(0))
    for c, v in enumerate(iou_pc):
        self.log(f"valid_iou_{c}", v, on_epoch=True, prog_bar=False)
    iou_without_animal = iou_pc[1:].mean()
    iou = iou_pc.mean()
    dice = self.dice_score(tp.sum(0), fp.sum(0), fn.sum(0), tn.sum(0)).mean()

    self.log('valid_loss', loss, on_epoch=True, prog_bar=True)
    self.log('valid_acc', acc, on_epoch=True, prog_bar=True)
    self.log('valid_miou', iou, on_epoch=True, prog_bar=True)
    self.log('valid_miou_without_animal', iou_without_animal,
             on_epoch=True, prog_bar=True)
    self.log('valid_dice', dice, on_epoch=True, prog_bar=True)

    return loss

def trainer_work(
    pl_model,
    model_name,
    train_dl,
    valid_dl,
    num_epochs=30,
    ignore_index=14,
    ckpt=None,
    train=True
  ):
  """
  Запуск обучения Lightning: logger + early stop + checkpoint по valid_miou.
  """

  pl.seed_everything(42, workers=True)

  logger = TensorBoardLogger(
      save_dir="lightning_logs",
      name=model_name,
      version=0
  )

  early_stop = pl.callbacks.EarlyStopping(
      monitor='valid_miou',
      mode='max',
      patience=5,
      verbose=True
  )

  dirpath = f'lightning_logs/{model_name}/checkpoints/'
  checkpoint = pl.callbacks.ModelCheckpoint(
      dirpath=dirpath,
      monitor='valid_miou',
      mode='max',
      save_top_k=1,
      save_last=True,
      filename=f'{model_name}' + '-{epoch}'
  )

  trainer = pl.Trainer(
      logger=logger,
      max_epochs=num_epochs,
      accelerator='gpu',
      devices=1,
      precision='16-mixed',
      callbacks=[early_stop, checkpoint],
      benchmark=True,
      enable_progress_bar=True,
      log_every_n_steps=10
      )
  pl_model.ignore_index = ignore_index
  if train:
    trainer.fit(pl_model, train_dl, valid_dl,
                ckpt_path=f"{dirpath}{ckpt}" if ckpt else None)
  else:
    return trainer.validate(pl_model, valid_dl)

def check_mask_board(idx, model, valid_dt, to_cls, num_classes):
  """
  Загружает последний ckpt (по epoch) и рисует сравнение масок на valid_dt[idx].
  """

  x, _ = valid_dt[idx] # 869, 873, 92
  x = x.to('cuda').unsqueeze(0)

  y_pred = model(x).argmax(dim=1).to('cpu')

  C = len(set(valid_dt.new_classes.values())) - 1
  mask_pred = torch.nn.functional.one_hot(
      y_pred.squeeze(0).long(), num_classes=num_classes
  ).permute(2, 0, 1).float()

  valid_dt.mask_board(to_cls, mask_pred);

def load_ckpt(model_name, model, loss_fn, num_epochs):
  dir_path = Path(f'lightning_logs/{model_name}/checkpoints')
  ckpt = [
      p for p in dir_path.glob('*.ckpt')
      if 'epoch=' in p.name
  ]
  ckpt = sorted(
      ckpt,
      key=lambda p: int(p.name.partition('epoch=')[2].partition('.')[0]),
      reverse=True
  )[0]
  model = SegLightning.load_from_checkpoint(
      f'{dir_path}/{pathlib.PurePath(ckpt).name}',
      model=model,
      loss_fn=loss_fn,
      num_epochs=num_epochs,
      strict=False,
      )
  model.eval()
  return model
