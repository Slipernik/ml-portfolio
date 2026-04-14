
from pathlib import Path

import urllib.request
import zipfile

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2, Compose, RandomApply


class Images(Dataset):
  def __init__(
      self,
      folder,
      train=True,
      int_label=False,
      transform=False,
      img_mode=ImageReadMode.UNCHANGED,
      seed=False,
      yolo=False,
      excluded=(1, 8),
      label_map=None,
      class_names=None,
      device='cuda'
    ):
    super().__init__()
    self.main_dir = folder
    self.images = []
    self.targets = []
    self.train = train
    self.int_label = int_label
    self.all_classes = []
    self.yolo = yolo
    self.excluded = tuple(excluded)
    self.label_map = label_map or {
      0: 0,
      2: 1,
      3: 2,
      4: 3,
      5: 4,
      6: 5,
      7: 6,
    }
    self.class_names = class_names or {
      0: 'angry',
      1: 'disgust',
      2: 'fear',
      3: 'happy',
      4: 'natural',
      5: 'sad',
      6: 'surprised',
    }
    self.filenames, self.class_lengths = self._get_file_paths()
    self.seed = seed
    self.transform = transform
    self.img_mode = ImageReadMode.GRAY if self.yolo else img_mode
    sample_count = min(15, len(self))
    if self.seed:
      torch.manual_seed(self.seed)
      self.rng = [torch.randint(len(self), (1,)) for _ in range(sample_count)]
    else:
      self.rng = [i for i in range(sample_count)]
    if self.int_label:
      self.class_lengths = {self.all_classes.index(k): v
                            for k, v in self.class_lengths.items()
                            if k != 'all'}
    self.device = device

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    one_file = self.filenames[idx]
    if self.yolo:
      one_file, label, bbox = one_file
      img = read_image(one_file, mode=self.img_mode)
      img = self._crop_yolo_box(img, bbox)
      label = torch.tensor(label, dtype=torch.long)
    else:
      img = read_image(one_file, mode=self.img_mode)
      label = self._get_target_label(one_file)
    if self.transform:
        img = self.transform(img)
    return img, label

  def show_images(
      self,
      suptitle='Изображения',
      cmap=None,
      text_pos=(0, -1),
      pred_labels=None
    ):
    if self.seed:
      torch.manual_seed(self.seed)
    fig, ax = plt.subplots(3, 5, figsize=(16, 9))
    ax = ax.ravel()
    current_cmap = cmap or ('Greys_r' if self.img_mode == ImageReadMode.GRAY else None)
    sample_indices = self._get_display_indices(pred_labels)
    for i, idx in enumerate(sample_indices):
      idx = self._as_int_index(idx)
      img, label = self.__getitem__(idx)
      if self.img_mode == ImageReadMode.GRAY:
        img = self._tensor_to_PIL(img) / 255.0
      else:
        img = self._tensor_to_PIL(img)
      ax[i].imshow(img, cmap=current_cmap)

      ax[i].set_xticks([])
      ax[i].set_yticks([])
      label = self._display_label(label)
      color = 'black'
      if pred_labels is not None:
        pred_label = self._display_label(self._get_pred_label(pred_labels, idx, i))
        if label != pred_label:
          color = 'red'
        label += f' / {pred_label}'
      ax[i].text(*text_pos, label, c=color)
    for i in range(len(sample_indices), len(ax)):
      ax[i].axis('off')
    fig.suptitle(suptitle)

  @property
  def weights(self):
    class_lengths = {k: v for k, v in self.class_lengths.items() if k != 'all'}
    weights = [class_lengths[k] for k in sorted(class_lengths)]
    weights = 1 / torch.sqrt(torch.tensor(weights, dtype=torch.float32))
    weights = weights / weights.mean()
    weights = weights.to(self.device)
    return weights

  def _get_file_paths(self):
    class_lengths = {}
    filenames = []
    if self.yolo:
      path = Path(self.main_dir) / ('train' if self.train else 'valid')
      images = path / 'images'
      labels = path / 'labels'
      for image in images.iterdir():
        label_path = labels / f'{image.stem}.txt'
        label, bbox = self._get_yolo_target(label_path)
        if label in self.excluded:
          continue
        label = self.label_map[label]
        class_lengths[label] = class_lengths.get(label, 0) + 1
        filenames.append((image, label, bbox))
    else:
      if self.train:
        path = Path(self.main_dir) / 'train'
      else:
        val_path = [p.name for p in Path(self.main_dir).glob('val*')][0]
        path = Path(self.main_dir) / val_path
      folders = [str(f) for f in path.iterdir() if f.is_dir()]
      for folder in folders:
        folder = Path(folder)
        label = folder.stem
        class_lengths[label] = 0
        for filename in folder.iterdir():
          if filename.is_file():
            filenames.append(filename)
            class_lengths[label] += 1
    self.all_classes = sorted(class_lengths.keys())
    class_lengths['all'] = len(filenames)
    return filenames, class_lengths

  def _get_target_label(self, path):
    path = Path(path)
    label_name = path.parent.name
    if self.int_label:
        return torch.tensor(self.all_classes.index(label_name), dtype=torch.long)
    return label_name

  def _get_yolo_target(self, path):
    with open(path, 'r') as f:
      label = f.readline()
      data = label.split(' ')
      return int(data[0]), list(map(float, data[1:5]))

  def _display_label(self, label):
    if torch.is_tensor(label):
      label = label.item()
    if self.yolo:
      return self.class_names.get(label, label)
    return label

  def _as_int_index(self, idx):
    if torch.is_tensor(idx):
      return int(idx.item())
    return int(idx)

  def _get_pred_label(self, pred_labels, idx, display_idx):
    if len(pred_labels) == len(self):
      return pred_labels[idx]
    if len(pred_labels) >= len(self.rng):
      return pred_labels[display_idx]
    raise ValueError(
      "pred_labels must have length equal to the dataset size "
      "or at least the number of displayed samples."
    )

  def _get_display_indices(self, pred_labels):
    display_count = min(len(self.rng), len(self))
    if pred_labels is None or len(pred_labels) != len(self):
      return [self._as_int_index(idx) for idx in self.rng[:display_count]]

    wrong_indices = []
    correct_indices = []

    for idx in range(len(self)):
      _, label = self.__getitem__(idx)
      true_label = self._display_label(label)
      pred_label = self._display_label(pred_labels[idx])

      if true_label != pred_label:
        wrong_indices.append(idx)
      else:
        correct_indices.append(idx)

    return (wrong_indices + correct_indices)[:display_count]

  def _crop_yolo_box(self, image, bbox):
    _, h, w = image.shape
    xc, yc, bw, bh = bbox
    x_center, y_center, box_w, box_h = xc * w, yc * h, bw * w, bh * h

    x1 = int(x_center - box_w / 2)
    y1 = int(y_center - box_h / 2)
    x2 = int(x_center + box_w / 2)
    y2 = int(y_center + box_h / 2)

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    if x2 <= x1:
      x2 = min(w, x1 + 1)
    if y2 <= y1:
      y2 = min(h, y1 + 1)

    return image[:, y1:y2, x1:x2]

  def _tensor_to_PIL(self, img: torch.Tensor) -> np.ndarray:
    return img.permute(1, 2, 0).numpy()

def download_and_unpack(url, archive_path, extract_dir, marker_dir=None):
    archive_path = Path(archive_path)
    extract_dir = Path(extract_dir)
    marker_dir = Path(marker_dir) if marker_dir else None


    if marker_dir and marker_dir.exists():
        print(f"Датасет уже доступен: {marker_dir}")
        return

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        urllib.request.urlretrieve(url, archive_path)
        print(f"Скачано: {archive_path}")
    else:
        print(f"Архив уже существует: {archive_path}")

    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Распаковано в: {extract_dir}")

def build_transforms(size, *, augment=False, normalize=True):
    train_transform_list = [v2.ToDtype(torch.float32, scale=True)]
    valid_transform_list = [v2.ToDtype(torch.float32, scale=True)]

    if augment:
        train_transform_list.extend(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomPerspective(distortion_scale=0.1, p=0.5),
                v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.1),
            ]
        )

    train_transform_list.append(v2.Resize(size=(size, size)))
    valid_transform_list.append(v2.Resize(size=(size, size)))

    if normalize:
        normalize_transform = v2.Normalize(mean=[0.5], std=[0.5])
        train_transform_list.append(normalize_transform)
        valid_transform_list.append(normalize_transform)

    return Compose(train_transform_list), Compose(valid_transform_list)

def build_dataloaders(
    data_dir,
    batch_size,
    train_transform,
    valid_transform,
    yolo=False,
    seed=1,
    num_workers=0,
    pin_memory=True,
    persistent_workers=True,
    img_mode=ImageReadMode.UNCHANGED,
    device='cuda'
):
    train_dt = Images(
        data_dir,
        int_label=True,
        img_mode=img_mode,
        transform=train_transform,
        yolo=yolo,
        seed=seed,
        device=device,
    )
    valid_dt = Images(
        data_dir,
        train=False,
        int_label=True,
        img_mode=img_mode,
        transform=valid_transform,
        yolo=yolo,
        seed=seed,
        device=device,
    )

    train_dl = DataLoader(
      train_dt,
      shuffle=True,
      batch_size=batch_size,
      num_workers=num_workers,
      pin_memory=pin_memory,
      persistent_workers=persistent_workers
    )
    valid_dl = DataLoader(
      valid_dt,
      shuffle=False,
      batch_size=batch_size,
      num_workers=num_workers,
      pin_memory=pin_memory,
      persistent_workers=persistent_workers
    )
    return train_dt, valid_dt, train_dl, valid_dl
