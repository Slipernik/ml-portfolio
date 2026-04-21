import gdown

demo = 'https://drive.google.com/drive/folders/1wooOS3KbT6snoJc0bJsfbmvK9j45gmhc?usp=sharing'
classification = 'https://drive.google.com/drive/folders/12wtIPcIAr4y1BxNiLj9iNJk-mPrxx8H1?usp=sharing'

gdown.download_folder(demo, quiet=False)
gdown.download_folder(classification, quiet=False)