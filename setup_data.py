import csv
import json
import os
import random
import requests
import shutil
import gdown
from tqdm import tqdm
from PIL import ImageFont
from utils.transform_image import draw_text_with_new_lines


def download_and_unzip(url):
    # Download the file from `url` and save it locally under `file_name`:
    # extract the filename from the url
    filename = url.split("/")[-1]
    if "drive.google.com" in url:
        filename = "gwfonts.zip"
        gdown.download(url, filename, quiet=False)
    else:
      data = requests.get(url, allow_redirects=True).content
      with open(filename, "wb") as f:
          f.write(data)

    # Unzip
    shutil.unpack_archive(filename)

    # remove __MACOSX folder
    shutil.rmtree("__MACOSX")

    if os.path.exists('gwfonts/._.DS_Store'):
      os.remove('gwfonts/._.DS_Store')


# see the website for more details: https://www.dgp.toronto.edu/~donovan/font/
urls = [
  # "https://www.dgp.toronto.edu/~donovan/font/gwfonts.zip",
  "https://drive.google.com/uc?id=1xpHkuqQtcpHt6r8xGL60KTMv4EYMEnRD",
  "https://www.dgp.toronto.edu/~donovan/font/attribute.zip",
  "https://www.dgp.toronto.edu/~donovan/font/similarity.zip",
]

for url in tqdm(urls):
    download_and_unzip(url)

  
# Prepare json files for training, validation, and testing
font_dir = "gwfonts"
csv_path = "attributeData/estimatedAttributes.csv"

tmp_font_to_attribute_values = {}
with open(csv_path, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    attributes = header[1:]
    for row in reader:
        tmp_font_to_attribute_values[row[0]] = {a: v for a, v in zip(attributes, row[1:])}
valid_font_to_attribute_values = {}
font_paths = [os.path.join(font_dir, p) for p in os.listdir(font_dir)]
for font_path in font_paths:
    font_name = os.path.splitext(os.path.basename(font_path))[0]
    if font_name in tmp_font_to_attribute_values.keys():
        valid_font_to_attribute_values[font_name] = tmp_font_to_attribute_values[font_name]
shuffled_font_names = list(valid_font_to_attribute_values.keys())
random.seed(123)
random.shuffle(shuffled_font_names)

all_font_to_attribute_values = {font_name: valid_font_to_attribute_values[font_name] for font_name in shuffled_font_names}

# split into train, validation, and test 120:40:40
train_font_names = shuffled_font_names[:int(len(shuffled_font_names) * 0.6)]
validation_font_names = shuffled_font_names[int(len(shuffled_font_names) * 0.6):int(len(shuffled_font_names) * 0.8)]
test_font_names = shuffled_font_names[int(len(shuffled_font_names) * 0.8):]
train_font_to_attribute_values = {font_name: all_font_to_attribute_values[font_name] for font_name in train_font_names}
validation_font_to_attribute_values = {font_name: all_font_to_attribute_values[font_name] for font_name in validation_font_names}
test_font_to_attribute_values = {font_name: all_font_to_attribute_values[font_name] for font_name in test_font_names}

# save json files
json.dump(all_font_to_attribute_values, open('attributeData/all_font_to_attribute_values.json', 'w'))
json.dump(train_font_to_attribute_values, open('attributeData/train_font_to_attribute_values.json', 'w'))
json.dump(validation_font_to_attribute_values, open('attributeData/validation_font_to_attribute_values.json', 'w'))
json.dump(test_font_to_attribute_values, open('attributeData/test_font_to_attribute_values.json', 'w'))


font_dir = '../gwfonts/'
output_dir = '../gwfonts_images/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
font_paths = os.listdir(font_dir)
print(f'font files: {len(font_paths)}')

char_size = 150
width_weight = 0.8
text = 'The quick\nbrown fox\njumps over\nthe lazy dog'
line_num = text.count('\n') + 1
width = int(char_size * len(text) * width_weight / line_num)
height = int(char_size * 1.5) * line_num
print(width, height)

for i in tqdm(range(len(font_paths))):
  font_file = font_paths[i]
  try:
    font_name = os.path.splitext(font_file)[0]
    font = ImageFont.truetype(os.path.join(font_dir, font_file), char_size)
    image = draw_text_with_new_lines(text, font, width, height)
    image.save(output_dir + font_name + '.png')
  except Exception as e:
    print(font_file)
    print(e)


# Vector Optimization
os.makedirs('svgs')
os.makedirs('svgs/init')
os.makedirs('output')