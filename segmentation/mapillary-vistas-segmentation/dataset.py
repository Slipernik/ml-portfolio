import os
from pathlib import Path
import numpy as np
import json
import tarfile

import requests
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import torchvision
from torchvision import tv_tensors
from torchvision.transforms import v2
from skimage.draw import polygon

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch


url = 'https://assets.supervisely.com/remote/eyJsaW5rIjogInMzOi8vc3VwZXJ2aXNlbHktZGF0YXNldHMvMjU3N19NYXBpbGxhcnkgVmlzdGFzL21hcGlsbGFyeS12aXN0YXMtRGF0YXNldE5pbmphLnRhciIsICJzaWciOiAiUTNadGllNFdHRkZtbTN5WFJYNnpvSmtua1QzNlpsMVdxdjJ6d0Z3RjN0MD0ifQ==?response-content-disposition=attachment%3B%20filename%3D%22mapillary-vistas-DatasetNinja.tar%22'


class Scene():
  """
    Класс для работы с датасетом сцен, содержащих изображения и
    аннотации в формате JSON с polygon-разметкой.

    Структура датасета:
      dst_path/
        meta.json
        training/
          img/
          ann/
        validation/
          img/
          ann/
        ...
    """

  def __init__(self, dst_path, dt_type='training', new_classes=False):
    """
    Инициализация датасета.

    Args:
      dst_path (str): Путь к корню датасета
      dt_type (str): Тип поднабора ('training', 'validation', etc.)
      new_classes (bool or dict):
        False — использовать классы из meta.json
        dict — маппинг старых классов в новые
    """

    self.dst_path = dst_path
    self.new_classes = new_classes
    self.dt_type = dt_type

    # Пути к данным
    self.dt_path = dst_path + '/' + dt_type
    self.img_path = self.dt_path + '/img/'
    self.ann_path = self.dt_path + '/ann/'

    # Список всех изображений
    self.files = sorted([
        f for f in Path(self.img_path).iterdir()
        if f.is_file()
    ])

    # Чтение мета-информации (классы, цвета)
    self._read_meta()

  def __len__(self):
    """
    Возвращает количество изображений в датасете.
    """
    return len(self.files)

  def __getitem__(self, idx):
    """
    Возвращает изображение и аннотацию по индексу.

    Args:
      idx (int): Индекс изображения

    Returns:
      img (Tensor): Изображение (C, H, W)
      ann (dict): Аннотация из JSON
    """
    idx = int(idx)
    idx = int(idx)
    self.filename = self.files[idx].name

    img = self._read_img(self.files[idx].as_posix())
    ann = self._read_json()

    return img, ann

  def img_board(self, own_idxs=None, size=3):
    """
    Визуализация изображений с наложенными polygon-аннотациями.

    Отображается 3 строки:
      1 — оригинал
      2 — полигоны с прозрачностью
      3 — полигоны без прозрачности

    Args:
      own_idxs (list[int], optional): Явно заданные индексы изображений
      size (int): Количество изображений для случайного выбора
    """
    if own_idxs:
      idxs = own_idxs
      size = len(idxs)
    else:
      idxs = torch.randint(high=len(self), size=(size,))

    fig, ax = plt.subplots(
        3, size+1,
        figsize=(6*size, 9),
        gridspec_kw={"width_ratios": [1]*size + [0.45]}
    )


    for i in range(3):
      for j in range(size):
        img, _ = self.__getitem__(idxs[j])
        ax[i, j].imshow(self._tensor_to_array(img))
        ax[i, j].set_aspect('auto')
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])

        # Наложение полигонов
        if i != 0:
          for cls, color, ext_points in zip(
              self.classes,
              self.colors,
              self.ext_points
          ):
            alpha = 0.4 if i == 1 else 1.0
            ax[i, j].fill(
                ext_points[:, 0],
                ext_points[:, 1],
                alpha=alpha,
                color=color,
                label=cls
            )

    # Последний столбец под легенду
    for i in range(3):
      ax[i, -1].axis("off")

    # Сбор легенды для всего Figure
    own_legend = {}
    for a in ax[:, :size].ravel():
      handles, labels = a.get_legend_handles_labels()
      for h, l in zip(handles, labels):
        if '-' in l:
          l = l.replace('- ', '-\n')
        own_legend[l] = h

    ax[1, -1].legend(
        own_legend.values(),
        own_legend.keys(),
        loc="center left",
        ncols=2 if len(own_legend) > 20 else 1,
        facecolor="#F0F0F0",
        borderaxespad=0.0,
        frameon=True
    )
    fig.suptitle('Изображения из Dataset с наложением polygons', y=0.96)
    fig.subplots_adjust(top=0.92, wspace=0.1, hspace=0.02)

  def _read_img(self, path):
    """
    Чтение изображения в формате torch.Tensor.

    Args:
      path (str): Путь к изображению

    Returns:
      Tensor: (C, H, W)
    """
    return torchvision.io.read_image(path)

  def _read_json(self):
    """
    Чтение JSON-аннотации для текущего изображения.

    Заполняет:
      self.objects
      self.classes
      self.colors
      self.ext_points

    Returns:
      dict: JSON-аннотация
    """
    json_file = self.ann_path + self.filename + '.json'

    with open(json_file) as f:
      ann = json.load(f)
      self.objects = ann['objects']

      # Обработка классов
      if self.new_classes:
        self.classes = [
            self.new_classes[obj['classTitle']]
            for obj in self.objects
        ]
      else:
        self.classes = [
            obj['classTitle']
            for obj in self.objects
        ]

      # Цвета и полигоны
      self.colors = [
          self.colors_map[cls]
          for cls in self.classes
      ]
      self.ext_points = [
          np.array(obj['points']['exterior'])
          for obj in self.objects
      ]

      return ann

  def _read_meta(self):
    """
    Чтение meta.json и формирование карты цветов классов.
    """
    json_file = self.dst_path + '/meta.json'

    with open(json_file) as f:
      self.meta = json.load(f)

      # Цвета новых классов
      if self.new_classes:
        self.colors_map = {
          'sky': '#4682B4',
          'nature': '#6B8E23',
          'nature-new': '#6B8E23',
          'flat': '#804080',
          'marking-new': '#FFFFFF',
          'human': '#DC143C',
          'animal': '#FF69B4',
          'vehicle': '#00008E',
          'structure': '#464646',
          'barrier': '#FF7F00',
          'traffic-light': '#FFD700',
          'sign': '#FF4500',
          'traffic-sign': '#ADFF2F',
          'support': '#999999',
          'object-new': '#1E90FF',
          'void': '#000000',
          'void-new': '#000000',
        }
      else:
        self.colors_map = {
            classes['title']: classes['color']
            for classes in self.meta['classes']
        }

  def _tensor_to_array(self, img):
    """
    Преобразует Tensor (C, H, W) → numpy (H, W, C)
    для отображения через matplotlib.
    """
    return img.permute(1, 2, 0)

class SceneDataset(Scene, Dataset):
  """
  PyTorch Dataset поверх Scene:
  - читает изображение + json-аннотацию (для полигонов / метаданных)
  - читает готовую маску из .pt (семантическая сегментация)
  - применяет resize и аугментации/трансформы
  - возвращает (img, mask) для train/val и (img, None) для теста/инференса
  """

  def __init__(
      self,
      dst_path,
      dt_type='training',
      new_classes=False,
      img_size=(512, 512),
      common_transform=None,
      img_transform=None
  ):
    """
    Инициализация датасета.

    Args:
      dst_path (str): Путь к корню датасета
      dt_type (str): Тип поднабора ('training', 'validation', etc.)
      new_classes (bool or dict):
        False — использовать классы из meta.json
        dict — маппинг старых классов в новые
      img_size (tuple[int,int]): итоговый размер изображения (H, W)
      common_transform (callable):
          трансформация, которая применяется одинаково к img и mask
          (аугментации, которые должны быть синхронны)
      img_transform (callable):
          трансформация только для img (нормализация и т.п.)
    """
    super().__init__(dst_path, dt_type, new_classes)

    self.img_size = img_size
    self.common_transform = common_transform
    self.img_transform = img_transform

  def __getitem__(self, idx):
    """
    Возвращает один элемент датасета.

    Для training/validation:
      img: float32, обычно [0..1] после ToDtype(scale=True)
      mask: long (int64) с классами в пикселях

    Для остальных сплитов:
      (img, None)
    """

    self.filename = self.files[idx].name

    # Читаем изображение (Tensor C,H,W, dtype uint8)
    self.img = self._read_img(self.files[idx].as_posix())

    # Читаем json-аннотацию (заполняет self.objects/self.classes/…)
    self._read_json()

    # Resize изображения, чтобы соответствовало размеру масок
    resizer = v2.Resize(self.img_size)
    self.img = resizer(self.img)

    # Для train/val ожидаем маску и делаем синхронные трансформации
    if self.dt_type in ['training', 'validation']:

      # Читаем маску (H,W с id классов)
      self.mask = self._read_mask()

      # Общие трансформы должны применяться одинаково к img и mask
      # Mask оборачиваем в tv_tensors.Mask, чтобы v2 корректно работал
      if self.common_transform:
        self.img, self.mask = self.common_transform(
            self.img,
            tv_tensors.Mask(self.mask)
        )

      # Переводим картинку в float32 и масштабируем в [0..1]
      self.img = v2.ToDtype(torch.float32, scale=True)(self.img)

      # Дополнительные трансформы только для изображения
      if self.img_transform:
        self.img = self.img_transform(self.img)

    else:
      # Для теста / инференса маски может не быть
      return self.img, None
    return self.img, self.mask.long()

  def mask_board(self, to_cls, mask_pred=None):
    """
    Визуализация масок по каждому классу.

    Если mask_pred передан:
      - показываются TP (green), FP (red), FN (blue)
    Иначе:
      - показывается GT для каждого класса

    Args:
      mask_pred (Tensor | None):
        ожидается one-hot/boolean формат [C, H, W] для предсказаний
        (mask_pred[i] используется как бинарная маска)

    Returns:
      ax (np.ndarray): массив осей matplotlib
    """

    gt_map = mcolors.ListedColormap([(0, 0, 0, 0), 'blue'])
    fig, ax = plt.subplots(3, 5, figsize=(16, 9))
    ax = ax.ravel()

    # C — количество классов (по new_classes)
    C = len(set(self.new_classes.values())) - 1

    # Преобразуем self.mask (H,W) -> набор бинарных масок по классам (C,H,W)
    new_mask = torch.zeros((C, self.mask.shape[0], self.mask.shape[1]))
    for i in range(C):
      new_mask[i] = (self.mask == i)

      ax[i].imshow(self._tensor_to_array(self.img))

      # Если передано предсказание — рисуем TP/FP/FN
      if mask_pred is not None:
        model_map = mcolors.ListedColormap([(0, 0, 0, 0), 'red'])
        diff_map = mcolors.ListedColormap([(0, 0, 0, 0), 'green'])

        # Пересечение (True Positive): и GT, и pred == 1
        diff = (new_mask[i] + mask_pred[i]) == 2

        # TP
        ax[i].imshow(diff, alpha=0.7, cmap=diff_map)
        # FP
        ax[i].imshow(mask_pred[i] != diff, alpha=0.7, cmap=model_map)
        # FN
        ax[i].imshow(
            new_mask[i] != diff,
            alpha=0.7,
            cmap=gt_map,
            label='Непредсказанные пиксели'
        )
      else:
        # Только GT-бинарная маска
        ax[i].imshow(new_mask[i], alpha=0.7, cmap=gt_map)

      ax[i].set_xticks([])
      ax[i].set_yticks([])
      ax[i].set_title(to_cls[i], size=10)

    # Легенда для режима сравнения
    if mask_pred is not None:
      legend_elements = [
        Patch(facecolor='green', edgecolor='green', label='Пересечение (TP)'),
        Patch(facecolor='red', edgecolor='red', label='Ложноположительные (FP)'),
        Patch(facecolor='blue', edgecolor='blue', label='Ложноотрицательные (FN)')
      ]
      fig.legend(
        handles=legend_elements,
        loc="lower center",
        facecolor="#F0F0F0",
        frameon=True
      )

    fig.subplots_adjust(top=0.92, wspace=0.02, hspace=0.1)
    fig.suptitle('Кастомные маски на каждый из классов')
    return ax

  def _read_mask(self):
    """
    Чтение маски сегментации из файла .pt.

    Ожидается файл:
      ann_path/<image_filename>.pt
    где image_filename — имя файла изображения (как в img/) без изменения.

    Returns:
        Tensor: маска (обычно H, W), dtype может быть любым — далее приводим к long()
    """

    pt_file = self.ann_path + self.filename + '.pt'
    mask = torch.load(pt_file)
    return mask

class EnlargedSceneDataset(SceneDataset):
  def __init__(
      self,
      dst_path,
      dt_type='training',
      new_classes=False,
      img_size=(512, 768),
      common_transform=None,
      img_transform=None
  ):
    super().__init__(
        dst_path,
        dt_type=dt_type,
        img_size=img_size,
        new_classes=new_classes,
        common_transform=common_transform,
        img_transform=img_transform
    )

  def _read_mask(self):
    pt_file = self.ann_path + self.filename + '768' + '.pt'
    mask = torch.load(pt_file)
    mask = mask.unsqueeze(0)
    mask = tv_tensors.Mask(mask)
    resizer = v2.Resize(self.img_size, interpolation=v2.InterpolationMode.NEAREST)
    mask = resizer(mask)
    return mask.squeeze(0)

def load_dataset(dst_path, archive_path):
    global url

    test_path = f"{dst_path}/mapillary-vistas-DatasetNinja.tar"    

    if os.path.exists(archive_path) == False:
      response = requests.get(url, stream=True)
      total_size = int(response.headers.get("content-length", 0))
      chunk_size = 1024 * 1024

      os.makedirs(os.path.dirname(archive_path), exist_ok=True)
      os.makedirs(dst_path, exist_ok=True)

      if response.status_code == 200:
          with open(archive_path, "wb") as f, tqdm(
              total=total_size,
              unit="B",
              unit_scale=True,
              desc="Downloading"
          ) as pbar:
              for chunk in response.iter_content(chunk_size=chunk_size):
                  if chunk:
                      f.write(chunk)
                      pbar.update(len(chunk))

    if os.path.exists(test_path) == False:
      with tarfile.open(archive_path) as tar:
          members = tar.getmembers()
          for member in tqdm(members, desc="Extracting"):
              tar.extract(member, path=dst_path)

def tag_to_cls(dst_path, classes):
  scn = Scene(dst_path)
  json_path = Path(dst_path) / "stats.json"

  if json_path.exists():
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    tags = {k: set(v) for k, v in data["tags"].items()}
    cls_counter = data["cls_counter"]
    tag_counter = data["tag_counter"]
    finded_cls = set(data["finded_cls"])

    print(f'Найдено классов: {len(finded_cls)}/{len(classes)}')

  else:
    tags = {}
    cls_counter = {}
    tag_counter = {}
    finded_cls = set()

    bar = tqdm(range(len(scn)))

    for i in bar:
      scene_classes = set()
      scene_tags = set()

      scn[i]

      for obj in scn.objects:
        cls = obj['classTitle']
        scene_classes.add(cls)
        finded_cls.add(cls)

        for tag in obj['tags']:
          tag_name = tag['name']
          scene_tags.add(tag_name)
          tags.setdefault(tag_name, set()).add(cls)

      for cls in scene_classes:
        cls_counter[cls] = cls_counter.get(cls, 0) + 1

      for tag_name in scene_tags:
        tag_counter[tag_name] = tag_counter.get(tag_name, 0) + 1

      bar.set_postfix(
        cls=f'{len(finded_cls)}/{len(classes)}',
        tags=len(tag_counter)
      )

    data = {
      "tags": {k: list(v) for k, v in tags.items()},
      "cls_counter": cls_counter,
      "tag_counter": tag_counter,
      "finded_cls": list(finded_cls),
    }

    with json_path.open("w", encoding="utf-8") as f:
      json.dump(data, f, ensure_ascii=False, indent=2)

  return tags, cls_counter, tag_counter, finded_cls

def new_mask(dst_path, new_classes, to_digit, size):
  for dt_type in ['training', 'validation']:
    scn = Scene(dst_path, dt_type=dt_type, new_classes=new_classes)
    bar = tqdm(range(len(scn)), desc=dt_type, leave=True)
    for i in bar:
      img, ann = scn[i]
      out_path = scn.ann_path + scn.filename
      if size == (512, 512):
        out_path += '.pt'
      else:
        out_path += f'{size[1]}.pt'
      H, W = img.shape[1], img.shape[2]
      mask = torch.zeros(size, dtype=torch.uint8)
      for cls, ext_point in zip(scn.classes, scn.ext_points):
        ext_point[:, 0] = (ext_point[:, 0] * (size[0] / W)).clip(0, size[0] - 1)
        ext_point[:, 1] = (ext_point[:, 1] * (size[1] / H)).clip(0, size[1] - 1)
        k = to_digit[cls]
        rr, cc = polygon(ext_point[:, 1], ext_point[:, 0], shape=size)
        mask[rr, cc] = k
      torch.save(mask, out_path)
