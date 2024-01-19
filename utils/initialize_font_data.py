import json
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# parent directory of CURRENT_DIR
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))

fox_text = 'The quick brown\nfox jumps over\nthe lazy dog'
fox_text_four_lines = 'The quick\nbrown fox\njumps over\nthe lazy dog'

font_dir = os.path.join(BASE_DIR, "gwfonts")
cj_font_dir = os.path.join(BASE_DIR, "all-fonts")
gray_scale_image_file_dir = os.path.join(BASE_DIR, "attributeData/grayscale_images")
all_gray_scale_image_file_dir = os.path.join(BASE_DIR, "attributeData/all_grayscale_images")
all_json_path = os.path.join(
    BASE_DIR, "attributeData/all_font_to_attribute_values.json"
)
train_json_path = os.path.join(
    BASE_DIR, "attributeData/train_font_to_attribute_values_cross_validation_5_0.json"
    #BASE_DIR, "attributeData/train.json"
)
test_json_path = os.path.join(
    BASE_DIR, "attributeData/test_font_to_attribute_values_cross_validation_5_0.json"
    #BASE_DIR, "attributeData/test.json"
)
validation_json_path = os.path.join(
    BASE_DIR, "attributeData/validation_font_to_attribute_values_cross_validation_5_0.json"
    #BASE_DIR, "attributeData/validation.json"
)
all_gwfonts_json_path = os.path.join(
    BASE_DIR, "attributeData/all_gwfonts.json"
)
train_all_gwfonts_json_path = os.path.join(
    BASE_DIR, "attributeData/train_all_gwfonts.json"
)
val_all_gwfonts_json_path = os.path.join(
    BASE_DIR, "attributeData/val_all_gwfonts.json"
)
one_leave_out_json_path = os.path.join(BASE_DIR, "attributeData/one_leave_out.json")
all_json = json.load(open(all_json_path, "r"))
train_json = json.load(open(train_json_path, "r"))
test_json = json.load(open(test_json_path, "r"))
validation_json = json.load(open(validation_json_path, "r"))
all_gwfonts_json = json.load(open(all_gwfonts_json_path, "r"))
train_all_gwfonts_json = json.load(open(train_all_gwfonts_json_path, "r"))
val_all_gwfonts_json = json.load(open(val_all_gwfonts_json_path, "r"))
train_font_names = list(train_json.keys())
test_font_names = list(test_json.keys())
validation_font_names = list(validation_json.keys())
all_font_names = list(all_json.keys())
font_names = list(all_json.keys())
all_gwfonts_names = list(all_gwfonts_json.keys())


def retrieve_font_path(font_name: str, font_dir: str = cj_font_dir) -> str:
    """
    Retrieve font path from font name

    Parameters
    ----------
    font_name : str
    font_dir : str

    Returns
    -------
    font_path : str
    """
    font_paths = os.listdir(font_dir)
    for font_path in font_paths:
        if font_name == os.path.splitext(os.path.basename(font_path))[0]:
            return os.path.join(font_dir, font_path)


font_paths = [
    retrieve_font_path(font_name, font_dir=font_dir) for font_name in font_names
]

# Attributes
with open(train_json_path, "r") as f:
    all_attributes = list(list(json.load(f).values())[0].keys())
exclusive_attributes = [
    "capitals",
    # "cursive",
    # "display",
    # "italic",
    "monospace",
    # "serif",
]
# exclusive_attributes = []
print("exclusive_attributes: ", exclusive_attributes)
inclusive_attributes = [
    attr for attr in all_attributes if attr not in exclusive_attributes
]
