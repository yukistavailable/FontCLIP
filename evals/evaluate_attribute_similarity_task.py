import torch
import json

from models.init_model import load_model, preprocess
from utils.initialize_font_data import (
    retrieve_font_path,
    inclusive_attributes,
    all_gray_scale_image_file_dir,
    font_dir,
    train_json_path,
    validation_json_path,
    test_json_path,
    all_json,
    fox_text,
)
from utils.transform_image import (
    generate_all_fonts_embedded_images,
    my_transform,
    generate_images_for_fonts,
)
from evals.evaluate_tools import (
    generate_all_attribute_embedded_prompts,
    user_attribute_choices_count,
    compare_two_fonts,
)


# If using GPU then use mixed precision training.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Must set jit=False for training
model_name = "ViT-B/32"
# model_name = "ViT-L/14"

# count font number
train_font_num = len(list(json.load(open(train_json_path, "r")).keys()))
print(train_font_num)

validation_font_num = len(list(json.load(open(validation_json_path, "r")).keys()))
print(validation_font_num)

test_font_num = len(list(json.load(open(test_json_path, "r")).keys()))
print(test_font_num)

font_names = list(all_json.keys())
font_paths = [
    retrieve_font_path(font_name, font_dir=font_dir) for font_name in font_names
]
# font_names = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir(font_dir)]

fox_text = fox_text
user_choices_count = user_attribute_choices_count


def retrieve_choices_with_font_name(user_choices, font_name):
    choices = []
    for user_choice in user_choices:
        if user_choice[3] == font_name or user_choice[4] == font_name:
            choices.append(user_choice)
    return choices


# target_font_names = extract_font_name_from_dir()
target_font_names = list(all_json.keys())
print("START!!!!!!!!!!!!!!!!!!!!!!!!!")


def evaluate_for_target_font_for_each_comparison(
    font_name, embedded_prompts, embedded_images, user_choices_count=user_choices_count
):
    result = []
    for user_choice_count in user_choices_count:
        # e.x., (('angular', 'ARSMaquetteWebOne', 'Kenia-Regular'), {'more': 4, 'less': 3})
        (attribute, font_a_name, font_b_name), tmp_ground_truth = user_choice_count
        if font_name is not None:
            if font_a_name != font_name and font_b_name != font_name:
                continue

        total_num = tmp_ground_truth["more"] + tmp_ground_truth["less"]
        ground_truth = (
            "more" if tmp_ground_truth["more"] > tmp_ground_truth["less"] else "less"
        )

        prediction = compare_two_fonts(
            attribute,
            font_a_name,
            font_b_name,
            ground_truth,
            embedded_prompts,
            embedded_images,
        )
        if prediction:
            correct_num = tmp_ground_truth[ground_truth]
        else:
            correct_num = total_num - tmp_ground_truth[ground_truth]

        tmp_result = (
            attribute,
            font_a_name,
            font_b_name,
            ground_truth,
            correct_num,
            total_num,
        )

        result.append(tmp_result)

    return result


aug = True
aug_num = 200
if not aug:
    aug_num = 1
result = []
correct_num = 0
total_num = 0
total_correct_rate = 0

font_name_to_image_tensor = None
preprocess_for_aug = my_transform(lower_bound_of_scale=1.0)
torch.manual_seed(1)
if aug:
    font_name_to_image_tensor = generate_images_for_fonts(
        font_paths=font_paths,
        text=fox_text,
        preprocess=preprocess_for_aug,
        image_file_dir=all_gray_scale_image_file_dir,
        aug_num=aug_num,
        crop_w=None,
        crop_h=None,
    )

tmp_test_json_path = test_json_path
tmp_test_json = json.load(open(tmp_test_json_path, "r"))
tmp_test_font_names = list(tmp_test_json.keys())
signature = "cv_20_ViT-B_32_bce_coop_precontext_length56_lr0.0001_91011_batch64_aug250_cj_lbound_of_scale0.35_max_attr_num_3_random_p_num_70000_geta0.2_use_negative_lr2e-05-0.1_image_file_dir"

checkpoint_path = f"model_checkpoints/{signature}.pt"
# CoOp
tmp_model = load_model(
    checkpoint_path,
    model_name="ViT-B/32",
    use_oft_vision=False,
    use_oft_text=False,
    oft_config_vision=None,
    oft_config_text=None,
    use_lora_text=False,
    use_lora_vision=False,
    lora_config_vision=None,
    lora_config_text=None,
    use_coop_text=True,
    use_coop_vision=False,
    precontext_length_vision=10,
    precontext_length_text=56,
    precontext_dropout_rate=0,
    pt_applied_layers=None,
)

# LoRA
# lora_config_text = LoRAConfig(
#     r = 256,
#     alpha = 1024.0,
#     bias = False,
#     learnable_alpha = False,
#     apply_q=True,
#     apply_k=True,
#     apply_v=True,
#     apply_out=True,
# )
# tmp_model = load_model(
#     checkpoint_path,
#     model_name="ViT-B/32",
#     use_oft_vision=False,
#     use_oft_text=False,
#     oft_config_vision=None,
#     oft_config_text=None,
#     use_lora_text=True,
#     use_lora_vision=False,
#     lora_config_vision=None,
#     lora_config_text=lora_config_text,
#     use_coop_text=False,
#     use_coop_vision=False,
#     precontext_length_vision=10,
#     precontext_length_text=77,
#     precontext_dropout_rate=0,
#     pt_applied_layers=None,
# )
tmp_model.eval()

embedded_prompts = generate_all_attribute_embedded_prompts(
    inclusive_attributes, model=tmp_model
)
embedded_images = generate_all_fonts_embedded_images(
    font_paths,
    fox_text,
    model=tmp_model,
    preprocess=preprocess_for_aug if aug else preprocess,
    image_file_dir=all_gray_scale_image_file_dir,
    aug_num=aug_num,
    crop_w=None,
    crop_h=None,
    font_name_to_image_tensor=font_name_to_image_tensor,
)

for target_font_name in tmp_test_font_names:
    # tmp_result = evaluate_for_target_font(target_font_name, embedded_prompts, embedded_images)
    # classification_rate = sum([1 for r in tmp_result if r[-1]])/len(tmp_result)
    tmp_result = evaluate_for_target_font_for_each_comparison(
        target_font_name, embedded_prompts, embedded_images
    )
    tmp_correct_num, tmp_total_num = (
        sum([r[-2] for r in tmp_result]),
        sum([r[-1] for r in tmp_result]),
    )
    tmp_classification_rate = tmp_correct_num / tmp_total_num
    total_correct_rate += tmp_classification_rate
    print(target_font_name, tmp_classification_rate)
    correct_num += tmp_correct_num
    total_num += tmp_total_num

average = correct_num / total_num
print(average)
