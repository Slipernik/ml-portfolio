import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def relation_pallete(
    tags,
    classes,
    tag_counter,
    cls_counter,
    length_scn
):
  """
  Рисует матрицу пересечения (tag -> classes) цветами:
    - каждая строка = тэг
    - каждый столбец = класс
    - если класс входит в тэг -> ячейка яркая (цвет тэга)
    - если не входит -> тёмная версия цвета тэга (фон строки)

  Дополнительно:
    - сортирует тэги по "редкости" (сцен на тэг)
    - сортирует классы взвешенно (через sort_classes_weighted)
    - подписывает тэги с процентом вхождения в датасет
    - рисует вертикальные линии для классов, которые встречаются в нескольких тэгах

  Args:
    tags (Dict[str, Set[str]]):
        mapping: tag -> множество классов, входящих в тэг
    classes (List[str]):
        список всех классов (будет пересортирован)
    tag_counter (Dict[str, int]):
        количество сцен/примеров на тэг (если есть)
    cls_counter (Dict[str, int]):
        количество сцен/примеров на класс
  """

  def _tag_scene_count(tag):
    """
    Оценка количества сцен для тэга:
      1) если есть явный tag_counter[tag] — используем его
      2) иначе берём максимум по cls_counter среди классов тэга
    """
    if tag in tag_counter:
      return tag_counter[tag]
    return max(cls_counter.get(cls, 0) for cls in tags.get(tag, set()))

  # Сортировка тэгов по редкости/частоте
  tags_sort = sorted(tags.items(), key=lambda x: _tag_scene_count(x[0]))

  # Сортировка классов
  cls_sort = sort_classes_weighted(tags_sort, classes)

  # Индексация (tag->row, class->col) + обратные отображения
  tag_to_digit = {tag: tag_num for tag_num, (tag, _) in enumerate(tags_sort)}
  digit_to_tag = {tag_num: tag for tag, tag_num in tag_to_digit.items()}

  cls_to_digit = {cls: cls_num for cls_num, cls in enumerate(cls_sort)}
  digit_to_cls = {cls_num: cls for cls, cls_num in cls_to_digit.items()}

  # one_hot_matrix хранит "входит ли класс в тэг"
  one_hot_matrix = np.zeros((len(tags), len(classes)))

  # color_matrix: [num_tags, num_classes, 3] — RGB цвет каждой ячейки
  color_matrix = np.zeros((len(tags), len(classes), 3), dtype=float)

  # Палитра цветов для тэгов (tab20 на число тэгов)
  color_pallete = plt.get_cmap("tab20")(np.linspace(0, 1, len(tags)))

  # Словарь цветов по ключам-тэгам
  tag_colors = {
      tag: color[:3]
      for tag, color in zip(
          tags.keys(),
          color_pallete
      )
  }

  # Заполнение матриц
  for tag, clss in tags.items():
    i = tag_to_digit[tag]
    tag_color = tag_colors[tag]

    # темный фон для строки тэга
    tag_color_dark = np.array(tag_color) * 0.15
    color_matrix[i, :, :] = tag_color_dark

    # яркие ячейки на пересечениях (tag, class)
    for cls in clss:
      j = cls_to_digit[cls]
      one_hot_matrix[i, j] = 1
      color_matrix[i, j] = tag_color

  # Настройки фигуры (ширина зависит от числа классов, высота — от тэгов)
  plt.figure(figsize=(0.15 * len(classes), 0.3 * len(tags)))

  # Горизонтальные линии между строками (для читаемости)
  for y in range(len(tags)):
    plt.hlines(
        y=y,
        xmin=0,
        xmax=len(classes),
        colors='white',
        linewidth=0.3,
        alpha=0.6
    )

  # Отрисовка матрицы соответствий
  plt.pcolormesh(color_matrix)
  plt.xticks([])

  # Подписи по оси Y: тэг + % от общего числа сцен
  ytick_labels = []
  for i in range(len(tags)):
    tag = digit_to_tag[i]
    cnt = _tag_scene_count(tag)
    pct = (cnt / length_scn * 100) if length_scn else 0.0
    ytick_labels.append(f"{tag} ({pct:.1f}%)")

  plt.yticks(np.arange(len(tags)) + 0.5, ytick_labels)

  # Вертикальные линии для классов, которые входят более чем в 1 тэг
  for x in range(len(classes)):
    all_y = np.where(one_hot_matrix[:, x] == 1)[0]
    if len(all_y) > 1:
      y_min, y_max = min(all_y), max(all_y)
      plt.plot(
          [x+0.5, x+0.5],
          [y_min+0.5, y_max+0.5],
          color='white',
          linewidth=2.5,
          alpha=0.5
      )

  # Текстовая статистика: сколько уникальных классов покрыто тегами
  count_clss = len({cls for clss in tags.values() for cls in clss})
  plt.text(0, -0.5, s=f'{count_clss}/{len(cls_counter)} классов осталось')

  plt.title(
      'Пересечение классов внутри тэгов',
      y=1.02,
      fontdict={'fontsize': 14}
  )
  plt.ylabel('Тэги и процент их вхождения во все примеры Dataset')
  plt.xlabel('Классы')
  plt.tight_layout()

def tags_cleaner(tags, class_name, union=True, suffix='-new'):
  """
  Обновляет один тэг:
    - union=True  → объединяет все пересекающиеся тэги
    - union=False → вычитает классы других тэгов

  Результат сохраняется как <class_name><suffix>.
  """
  new_name = class_name + suffix
  base_set = set(tags[class_name])

  if union:
    intersect_tags = [
      tag for tag, clss in tags.items()
      if tag != class_name and (base_set & clss)
    ]
    for tag in intersect_tags:
      base_set |= tags[tag]
      del tags[tag]
  else:
    for tag, clss in tags.items():
      if tag != class_name:
        base_set -= clss

  del tags[class_name]
  tags[new_name] = base_set

def sort_classes_weighted(tags_sort, classes):
  """
  Сортирует классы по:
    - максимальному размеру тэга, где класс встречается
    - количеству соседних классов
    - количеству тэгов, где класс встречается
    - имени класса (для стабильности)
  """
  classes_set = set(classes)

  tag_len = {tag: len(clss) for tag, clss in tags_sort}
  max_tag_len = {c: 0 for c in classes}
  neigh = {}
  cnt_tags = {c: 0 for c in classes}

  for tag, clss in tags_sort:
    clss = [c for c in clss if c in classes_set]
    L = tag_len[tag]
    for c in clss:
      if L > max_tag_len[c]:
        max_tag_len[c] = L
      cnt_tags[c] += 1
      neigh.setdefault(c, set())
      neigh[c].update(x for x in clss if x != c)

  def key(c):
    n = neigh.get(c, ())
    return (-max_tag_len[c], - len(n), -cnt_tags[c], c)

  return sorted(classes, key=key)

def bar_plot(bar_data, bar_names, title, xlabel, ylabel, to_cls):
  """
  Сгруппированная столбчатая диаграмма.

  bar_data — np.ndarray [groups, categories]
  bar_names — подписи групп (легенда)
  title, xlabel, ylabel — подписи графика

  В столбцах отображаются значения и их ранг
  внутри каждой категории (1 — максимум).
  """
  len_ticks = bar_data.shape[0]
  x = np.arange(bar_data.shape[1])
  width = 1 / (len_ticks + 1)
  steps = np.arange(0, 1, width)

  fig, ax = plt.subplots(figsize=(16, 9))

  for i, step in enumerate(steps[:-1]):
    height = bar_data[i]
    plt.bar(x=x+step, height=height, width=width, label=bar_names[i])
    for j, s in enumerate(height):
      plt.text(
          x=x[j]+step,
          y=s/2,
          s=f'{s:.3f}',
          rotation='vertical',
          va='center',
          ha='center'
      )
      plt.text(
          x=x[j]+step,
          y=s+0.01,
          s=int(
              np.where(
                  sorted(
                      list(bar_data[:, j]), reverse=True) == s
                  )[0]
              ) + 1,
          ha='center'
      )

  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)

  plt.xticks(
      ticks=x + width * (len_ticks - 1) / 2,
      labels=list(to_cls.values())[:-1],
      rotation=45
  )
  plt.yticks(np.arange(0, 1.01, 0.1))
  plt.legend();

def beauty_df(styler):
  """
  Стилизует DataFrame с метриками.

  - Форматирует числа до 3 знаков.
  - Для всех колонок:
      min → красный, max → зелёный.
  - Для `valid_loss`, `valid_time`:
      min → зелёный, max → красный.

  Использование:
    stats.style.pipe(beauty_df)
  """
  styler = styler.set_table_styles([
    {"selector": "table", "props": [("border-collapse", "collapse")]},
    {"selector": "th", "props": [("border", "1px solid #ccc"),
                                ("padding", "6px")]},
    {"selector": "td", "props": [("border", "1px solid #ccc"),
                                ("padding", "6px")]}
  ])

  styler = styler.format(precision=3)
  styler = styler.apply(lambda col: [
    "color: red;" if pd.to_numeric(v, errors="coerce") == pd.to_numeric(col, errors="coerce").min()
    else "color: green;" if pd.to_numeric(v, errors="coerce") == pd.to_numeric(col, errors="coerce").max()
    else ""
    for v in col
  ])

  styler = styler.apply(lambda col: [
    "color: green;" if pd.to_numeric(v, errors="coerce") == pd.to_numeric(col, errors="coerce").min()
    else "color: red;" if pd.to_numeric(v, errors="coerce") == pd.to_numeric(col, errors="coerce").max()
    else ""
    for v in col
  ], subset=["valid_loss", "valid_time"])

  return styler
