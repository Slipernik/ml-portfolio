import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
import timm


class CustomCNNLightning(pl.LightningModule):
  def __init__(self, model, weights, num_classes):
    super().__init__()
    self.model = model
    self.loss = nn.CrossEntropyLoss(weight=weights)
    self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
    self.valid_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

  def forward(self, x):
    return self.model(x)

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=1e-3
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    return [optimizer], [{"scheduler": scheduler}]

  def training_step(self, train_batch, batch_idx):
    x, y = train_batch
    logits = self.model(x)
    loss = self.loss(logits, y)

    pred = torch.argmax(logits, dim=1)
    acc = self.train_acc(pred, y)

    self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
    self.log('train_acc', acc, on_epoch=True, prog_bar=True)

    return loss

  def validation_step(self, valid_batch, batch_idx):
    x, y = valid_batch
    logits = self.model(x)
    loss = self.loss(logits, y)

    pred = torch.argmax(logits, dim=1)
    acc = self.valid_acc(pred, y)

    self.log('valid_loss', loss, on_epoch=True, prog_bar=True)
    self.log('valid_acc', acc, on_epoch=True, prog_bar=True)

    return loss

class DistillCNNLightning(pl.LightningModule):
  def __init__(self, student_model, teacher_model, alpha, temperature, weights, num_classes):
    super().__init__()
    self.student_model = student_model
    self.teacher_model = teacher_model
    self.alpha = alpha
    self.temperature = temperature
    self.weights = weights
    self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
    self.valid_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
    self.teacher_model.eval()
    for p in self.teacher_model.parameters():
      p.requires_grad = False

    self.best_loss = 100.0
    self.best_accuracy = 1e-4
    self.best_epoch=0.0
    self.best_alpha=0.0
    self.best_temperature=0.0

  def forward(self, x):
    return self.student_model(x)

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
            [p for p in self.student_model.parameters() if p.requires_grad],
            lr=1e-3
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    return [optimizer], [{"scheduler": scheduler}]

  def training_step(self, train_batch, batch_idx):
    x, y = train_batch
    logits = self.student_model(x)
    with torch.no_grad():
      teacher_logits = self.teacher_model(x)
    loss = self.loss(logits, y, teacher_logits,
                     self.alpha, self.temperature, self.weights)

    pred = torch.argmax(logits, dim=1)
    acc = self.train_acc(pred, y)

    self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
    self.log('train_acc', acc, on_epoch=True, prog_bar=True)

    return loss

  def validation_step(self, valid_batch, batch_idx):
    x, y = valid_batch
    logits = self.student_model(x)
    with torch.no_grad():
      teacher_logits = self.teacher_model(x)
    loss = self.loss(logits, y, teacher_logits,
                     self.alpha, self.temperature, self.weights)

    pred = torch.argmax(logits, dim=1)
    acc = self.valid_acc(pred, y)

    self.log('valid_loss', loss, on_epoch=True, prog_bar=False)
    self.log('valid_acc', acc, on_epoch=True, prog_bar=False)

    return loss

  def loss(self, outputs, labels, teacher_outputs, alpha, T, weights):
    KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels, weight=weights) * (1. - alpha)

    return KD_loss

  def on_validation_epoch_end(self):
    val_acc = self.valid_acc.compute()
    val_loss = self.trainer.callback_metrics["valid_loss"]

    if val_acc > self.best_accuracy:
        self.best_accuracy = val_acc
        self.best_loss = val_loss
        self.best_epoch=self.current_epoch
        self.best_alpha=self.alpha
        self.best_temperature=self.temperature

    self.valid_acc.reset()

def init_model(num_classes, device, finetuning=True):
    model = timm.create_model(
        'convnextv2_tiny',
        pretrained=True,
        num_classes=num_classes,
        in_chans=1
    )
    model = model.to(device)

    if finetuning:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.head.parameters():
            param.requires_grad = True

        for param in model.stem.parameters():
            param.requires_grad = True

        for param in model.stages[-1].parameters():
            param.requires_grad = True

    return model

def trainer_work(
    pl_model,
    train_dl,
    valid_dl,
    filename,
    num_epochs,
    train=True,
    accelerator=None,
    precision=None,
):
    use_cuda = torch.cuda.is_available()
    trainer_accelerator = accelerator or ('gpu' if use_cuda else 'cpu')
    trainer_precision = precision or ('16-mixed' if trainer_accelerator == 'gpu' else '32-true')

    early_stop = pl.callbacks.EarlyStopping(
        monitor='valid_loss',
        mode='min',
        patience=5,
        verbose=True,
    )
    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor='valid_loss',
        mode='min',
        save_top_k=1,
        filename=filename,
    )

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator=trainer_accelerator,
        devices=1,
        precision=trainer_precision,
        callbacks=[early_stop, checkpoint],
        benchmark=True,
        enable_progress_bar=True,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
    )
    if train:
        trainer.fit(pl_model, train_dl, valid_dl)
        return checkpoint.best_model_path
    else:
        return trainer.validate(pl_model, valid_dl)

def load_from_ckpt(model, path, device=None):
  if device is None:
    device = next(model.parameters()).device

  ckpt = torch.load(path, map_location=device)
  state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

  new_state = {}
  for k, v in state_dict.items():
      new_k = k
      if new_k.startswith("model."):
          new_k = new_k[len("model."):]
      elif new_k.startswith("student_model."):
          new_k = new_k[len("student_model."):]
      elif new_k.startswith("teacher_model."):
          continue
      new_state[new_k] = v

  missing, unexpected = model.load_state_dict(new_state, strict=False)
  print("missing:", missing)
  print("unexpected:", unexpected)
  return model
