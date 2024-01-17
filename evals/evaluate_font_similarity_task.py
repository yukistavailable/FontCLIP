import torch
import json

import clip
import os
from models.lora_multiheadattention import LoRAConfig
from models.init_model import load_model, preprocess, my_preprocess, my_transform
from utils.initialize_font_data import (
    retrieve_font_path,
    inclusive_attributes,
    all_gray_scale_image_file_dir,
    cj_font_dir,
    font_dir,
    train_json_path,
    validation_json_path,
    test_json_path,
    all_json,
    fox_text,
    fox_text_four_lines,
)
from utils.transform_image import (
    generate_all_fonts_embedded_images,
    generate_images_for_fonts,
)
from evals.evaluate_tools import (
    generate_all_attribute_embedded_prompts,
    user_attribute_choices_count,
    compare_two_fonts,
    evaluate_attribute_comparison_task,
    evaluate_similarity_comparison_task,
    user_similarity_choices,
)
from cj_fonts import inclusive_fonts, fifty_fonts

# If using GPU then use mixed precision training.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Must set jit=False for training

fifty_font_paths = [
    os.path.join(cj_font_dir, f) for f in fifty_fonts.split("\n") if f != ""
]

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

# target_font_names = extract_font_name_from_dir()
target_font_names = list(all_json.keys())


def evaluate_for_target_font_for_each_comparison(
    font_name, embedded_images, user_choices=user_similarity_choices, device=device
):
    result = []
    for user_choice in user_choices:
        # e.x., ['Muli', 'ARSMaquetteWebOne', 'Kenia-Regular', 10, 4]
        (
            reference_font_name,
            font_a_name,
            font_b_name,
            vote_count_a,
            vote_count_b,
        ) = user_choice
        if font_name is not None:
            if font_name not in [reference_font_name, font_a_name, font_b_name]:
                continue

        ground_truth = vote_count_a > vote_count_b

        embedded_r = embedded_images[reference_font_name].to(device)
        embedded_a = embedded_images[font_a_name].to(device)
        embedded_b = embedded_images[font_b_name].to(device)

        # calculate the cos similarity
        cos_sim_a = torch.cosine_similarity(embedded_r, embedded_a, dim=-1)
        cos_sim_b = torch.cosine_similarity(embedded_r, embedded_b, dim=-1)
        tmp_prediction = cos_sim_a.item() > cos_sim_b.item()
        prediction = ground_truth == tmp_prediction
        total_num = vote_count_a + vote_count_b
        if prediction:
            correct_num = vote_count_a if ground_truth else vote_count_b
        else:
            correct_num = vote_count_b if ground_truth else vote_count_a
        tmp_result = (
            reference_font_name,
            font_a_name,
            font_b_name,
            ground_truth,
            prediction,
            correct_num,
            total_num,
        )
        result.append(tmp_result)
    return result


aug = True
aug_num = 200
if not aug:
    aug_num = 1
cross_validation_k = 20
correct_num = 0
total_num = 0
total_classification_rate = 0

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
for i in range(cross_validation_k):
    tmp_test_json_path = f"../attributeData/test_font_to_attribute_values_cross_validation_{cross_validation_k}_{i}.json"
    tmp_test_json = json.load(open(tmp_test_json_path, "r"))
    tmp_test_font_names = list(tmp_test_json.keys())

    signature = f"cv_20_{i}_ViT-B_32_bce_lora_v-qkvo_256-1024.0_t-qkvo_256-1024.0_91011_batch64_aug250_lbound_of_scale0.35_max_attr_num_3_random_p_num_70000_geta0.2_use_negative_lr2e-05-0.1_image_file_dir"
    signature = f"cv_20_{i}_ViT-B_32_bce_lora_t-qkvo_256-1024.0_91011_batch64_aug250_lbound_of_scale0.35_max_attr_num_3_random_p_num_70000_geta0.2_use_negative_lr2e-05-0.1_image_file_dir"
    signature = f"cv_20_{i}_ViT-B_32_bce_lora_t-qkvo_256-1024.0_91011_batch64_aug250_lbound_of_scale0.35_max_attr_num_3_random_p_num_70000_geta0.2_use_negative_lr2e-05-0.1_image_file_dir"
    signature = f"cv_20_{i}_ViT-B_32_bce_lora_t-qkvo_256-1024.0_91011_batch64_aug250_lbound_of_scale0.35_max_attr_num_3_random_p_num_70000_geta0.2_use_negative_til1.0_lr2e-05-0.1_image_file_dir"
    signature = f"cv_20_{i}_ViT-B_32_bce_lora_t-qkvo_256-1024.0_91011_batch64_aug250_lbound_of_scale0.35_max_attr_num_3_random_p_num_70000_geta0.2_use_negative_til1.0_lr2e-05-0.1_image_file_dir"
    signature = f"cv_20_{i}_ViT-B_32_bce_lora_t-qkvo_256-1024.0_91011_batch64_aug250_lbound_of_scale0.35_max_attr_num_3_random_p_num_70000_geta0.2_use_negative_lr2e-05-0.1_image_file_dir"
    # signature = f"cv_20_{i}_ViT-B_32_bce_9101191011_batch64_aug250_lbound_of_scale0.35_max_attr_num_3_random_p_num_70000_geta0.2_use_negative_lr2e-05-0.1_image_file_dir"
    signature = f"cv_20_{i}_ViT-B_32_bce_coop_precontext_length56_lr0.0001_91011_batch64_aug250_lbound_of_scale0.35_max_attr_num_3_random_p_num_70000_geta0.2_use_negative_lr2e-05-0.1_image_file_dir"
    signature = f"cv_20_{i}_ViT-B_32_bce_lora_t-qkvo_256-1024.0_91011_batch64_aug250_cj_lbound_of_scale0.35_max_attr_num_3_random_p_num_70000_geta0.2_use_negative_lr2e-05-0.1_image_file_dir"
    signature = f"cv_20_{i}_ViT-B_32_bce_coop_precontext_length56_lr0.0001_91011_batch64_aug250_cj_lbound_of_scale0.35_max_attr_num_3_random_p_num_70000_geta0.2_use_negative_lr2e-05-0.1_image_file_dir"

    checkpoint_path = f"model_checkpoints/{signature}.pt"

    # Direct fine-tuning
    # tmp_model = load_model(
    #     model,
    #     checkpoint_path,
    #     model_name="ViT-B/32",
    #     learnable_prompt=False,
    #     learnable_vision=False,
    #     precontext_length=77,
    #     precontext_vision_length=0,
    #     precontext_dropout_rate=0,
    #     vpt_applied_layers=None,
    #     use_oft_vision=False,
    #     use_oft_text=False,
    #     oft_config_vision=None,
    #     oft_config_text=None,
    #     inject_lora=False,
    #     lora_config_vision=None,
    #     lora_config_text=None,
    # )

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
        tmp_result = evaluate_for_target_font_for_each_comparison(
            target_font_name, embedded_images
        )
        tmp_correct_num, tmp_total_num = sum([e[-2] for e in tmp_result]), sum(
            [e[-1] for e in tmp_result]
        )
        tmp_classification_rate = tmp_correct_num / tmp_total_num
        total_classification_rate += tmp_classification_rate
        print(target_font_name, tmp_classification_rate)
        correct_num += tmp_correct_num
        total_num += tmp_total_num

average = correct_num / total_num
print(average)
print(total_classification_rate / 200)
