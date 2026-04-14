import monai
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

class BorderCropBBox:
  """
  Трансформ для совместной обработки (img, mask):

  - Применяет RandomAffine (поворот/сдвиг/масштаб) к изображению и маске.
  - Если после аффинного преобразования маска стала "битой",
    то возвращает исходные (img, mask).
  - Находит bounding box всех валидных пикселей маски (mask > 0),
    обрезает (crop) по этому bbox с опциональным pad.
  - Проставляет "void-new" класс (=14) там, где после преобразований у изображения
    нулевая сумма по каналам (чёрные области, появившиеся из-за аффинного заполнения).
  5) Ресайзит обратно до фиксированного размера (H, W).

  Returns:
    img: Tensor [C, H, W]
    mask: Tensor [H, W] (целочисленная маска классов)
  """

  def __init__(self, size):
    """
    Args:
      size (tuple[int,int]): итоговый размер (H, W) после ресайза
    """
    self.H, self.W = size

    # Финальный ресайз обратно к фиксированному размер
    self.resize = v2.Resize((self.H, self.W))

    # Аффинные преобразования
    self.affine = v2.RandomAffine(
        degrees=(0, 5),
        translate=(0.15, 0.1),
        scale=(0.75, 1.5)
    )

  def __call__(self, img, mask):
    """
    Args:
      img (Tensor): изображение [C, H, W]
      mask (Tensor): маска [H, W]

    Returns:
      (img, mask): преобразованные и приведённые к размеру (self.H, self.W)
    """

    # Пробуем применить аффинный трансформ к img и mask
    new_img, new_mask = self.affine(img, mask)

    # Защитная проверка, на наличие класса миноритария
    if not (new_mask == 0).any():
      return img, mask

    img, mask = new_img, new_mask

    # Строим карту валидных пикселей, по которым будем искать bbox
    valid = (mask > 0)

    # Координаты всех валидных пикселей
    ys, xs = torch.where(valid)

    # Если валидных пикселей нет — просто ресайзим и возвращаем
    if ys.numel() == 0:
        return self.resize(img, mask)

    # bbox: min/max координаты
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    # Высота/ширина окна crop
    h = (y2 - y1 + 1).item()
    w = (x2 - x1 + 1).item()

    # Crop изображения и маски по bbox
    img = v2.functional.crop(img, int(y1), int(x1), h, w)
    mask = v2.functional.crop(mask, int(y1), int(x1), h, w)

    # Помечаем "пустые" области (чёрные пиксели), появившиеся после affine,
    # специальным классом, как "void-new".
    mask[img.sum(dim=0) == 0] = 14

    # Ресайз к фиксированному размеру
    return self.resize(img, mask)

class MonaiCropRare:
  """
  Трансформ (img, mask), который использует MONAI RandCropByLabelClassesd,
  чтобы вырезать несколько кандидатов-кропов, смещённых в сторону редких классов,
  и затем выбрать один “подходящий” кроп.
  """

  def __init__(
      self,
      out_size=(512, 512),
      crop_size=(256, 256),
      num_classes=15,
      class_weight=None
    ):
    """
    Args:
      out_size (tuple[int,int]): итоговый размер (H, W) после ресайза
      crop_size (tuple[int,int]): размер кропа, который выбирает MONAI
      num_classes (int): количество классов в маске
      class_weight (list[float] | None):
        веса/ratio классов для RandCropByLabelClassesd, для выделения
        класса миноритария
    """
    self.num_classes = num_classes

    # Финальный ресайз обратно к фиксированному размер
    self.resize = v2.Resize(out_size)

    # Кропы по классам с большим весом
    self.crop = monai.transforms.Compose([
        monai.transforms.EnsureChannelFirstd(
            keys=["image"], channel_dim=0
            ),
        monai.transforms.EnsureChannelFirstd(
            keys=["label"], channel_dim="no_channel"
            ),
        monai.transforms.RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=crop_size,
            num_classes=self.num_classes,
            ratios=class_weight,
            num_samples=4,
        )
    ])

  def __call__(self, img, mask):
    """
    Args:
      img (Tensor): изображение [C, H, W]
      mask (Tensor): маска [H, W]

    Returns:
      (img, mask): преобразованные и приведённые к размеру (self.H, self.W)
    """
    # MONAI RandCropByLabelClassesd возвращает list[dict], длины num_samples.
    # Каждый элемент: {"image": crop_img, "label": crop_label}
    d = self.crop({"image": img, "label": mask})

    # Выбираем первый кроп, где в label присутствует класс 0
    for i in range(len(d)):
      if torch.any(d[i]['label'] == 0):
        xi, yi = d[i]["image"], d[i]["label"]

        if yi.ndim == 3 and yi.shape[0] == 1:
            yi = yi[0]

        # Оборачиваем в tv_tensors, чтобы torchvision v2 корректно применил resize
        xi = tv_tensors.Image(xi)
        yi = tv_tensors.Mask(yi)

        # Ресайз к фиксированному размеру
        return self.resize(xi, yi)

    # Если ни один кроп не подошёл — возвращаем исходное
    return img, mask

class SoftMonaiCropRare(MonaiCropRare):
  def __init__(
      self,
      out_size=(512, 512),
      crop_size=(256, 256),
      num_classes=15,
      class_weight=None
    ):
    super().__init__(
        out_size=out_size,
        crop_size=crop_size,
        num_classes=num_classes,
        class_weight=class_weight
    )

  def __call__(self, img, mask):
    d = self.crop({"image": img, "label": mask})
    xi, yi = d[0]["image"], d[0]["label"]

    if yi.ndim == 3 and yi.shape[0] == 1:
        yi = yi[0]

    xi = tv_tensors.Image(xi)
    yi = tv_tensors.Mask(yi)

    return self.resize(xi, yi)
