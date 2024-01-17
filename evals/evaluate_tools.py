import json
import os
import PIL
from PIL import ImageFont
from typing import List, Tuple


import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from utils.tokenizer import tokenize
from models.init_model import device, model, preprocess
from utils.transform_image import (
    char_size,
    draw_text_with_new_lines,
    generate_all_fonts_embedded_images,
)
from utils.initialize_font_data import (
    font_dir,
    font_paths,
    font_names,
    inclusive_attributes,
)
from dataset.dataset import TestDataset, TestImageDataset, TestTextDataset, PairedImageDataset


def retrieve_one_leave_out_model_path(
    font_name: str, checkpoint_dir: str = "model_checkpoints"
) -> str:
    return os.path.join(checkpoint_dir, f"one_leave_out_{font_name}.pt")


def extract_font_name_from_path(font_path: str) -> str:
    font_path = os.path.splitext(os.path.basename(font_path))[0]
    return font_path.replace("one_leave_out_", "")


def extract_font_name_from_dir(dir: str = "model_checkpoints") -> List[str]:
    file_paths = os.listdir(dir)
    result = []
    for file_path in file_paths:
        if file_path.startswith("one_leave_out_"):
            result.append(extract_font_name_from_path(file_path))
    return result


def predict_cos_sim(
    font: ImageFont.FreeTypeFont,
    target_attribute: str,
    text: str,
    model=model,
    preprocess=preprocess,
    device=device,
    char_size=char_size,
) -> Tuple[torch.Tensor, PIL.Image.Image]:
    """
    Predict cosine similarity between font and target attribute
    """
    line_num = text.count("\n") + 1
    width = int(char_size * len(text) / line_num)
    height = (char_size + 30) * line_num
    text_image: PIL.Image.Image = draw_text_with_new_lines(text, font, width, height)
    image_tensor = preprocess(text_image).unsqueeze(0).to(device)
    embedded_image = model.encode_image(image_tensor)
    prompt = f"{target_attribute} font"
    context_length = model.context_length
    if hasattr(model, "precontext_length"):
        context_length -= model.precontext_length
    embedded_prompt = model.encode_text(tokenize(prompt, context_length=context_length).to(device))
    cos_sim = torch.cosine_similarity(embedded_image, embedded_prompt, dim=-1)
    return cos_sim, text_image


def predict_cos_sim_with_embedded_vectors(
    embedded_prompt: torch.Tensor, embedded_image: torch.Tensor
) -> torch.Tensor:
    """
    Predict cosine similarity between font and target attribute
    """
    cos_sim = torch.cosine_similarity(embedded_image, embedded_prompt, dim=-1)
    return cos_sim


# attribute comparison task
def retrieve_elements_from_user_choice(user_choice) -> Tuple[str, str, str, str]:
    """
    Parameters
    ----------
    user_choice: list[str]
        [attribute,hit_id,user_id,font_A_name,font_B_name,user_choice]

    Returns
    -------
    attribute: str
    font_a_name: str
    font_b_name: str
    ground_truth: str
    """
    attribute = user_choice[0]
    font_a_name = user_choice[3]
    font_b_name = user_choice[4]
    ground_truth = user_choice[5]
    return attribute, font_a_name, font_b_name, ground_truth


user_choices_csv = "../attributeData/userChoices.csv"
with open(user_choices_csv, "r") as f:
    # the format is [attribute,hit_id,user_id,font_A_name,font_B_name,user_choice]
    tmp_user_choices = f.read().split("\n")

user_choices = []
for tmp_user_choice in tmp_user_choices:
    if tmp_user_choice == "":
        continue
    user_choice = tmp_user_choice.split(",")
    assert len(user_choice) == 6
    assert user_choice[0] in inclusive_attributes
    assert user_choice[3] in font_names
    assert user_choice[4] in font_names
    assert user_choice[5] in ["more", "less"]
    user_choices.append(user_choice)

user_choices_count = {}
for user_choice in user_choices:
    (
        attribute,
        font_a_name,
        font_b_name,
        ground_truth,
    ) = retrieve_elements_from_user_choice(user_choice)
    key = (attribute, font_a_name, font_b_name)
    if key not in user_choices_count:
        user_choices_count[key] = {"more": 0, "less": 0}
    else:
        user_choices_count[key][ground_truth] += 1

# sort user_choices_count
user_attribute_choices_count = sorted(user_choices_count.items())


def compare_two_fonts(
    attribute,
    font_a_name,
    font_b_name,
    ground_truth,
    embedded_prompts,
    embedded_images,
    device=device,
):
    """
    Compute the cos sims of attribute and font_a, font_b
    """
    cos_sim_a = predict_cos_sim_with_embedded_vectors(
        embedded_prompts[attribute].to(device), embedded_images[font_a_name].to(device)
    )
    cos_sim_b = predict_cos_sim_with_embedded_vectors(
        embedded_prompts[attribute].to(device), embedded_images[font_b_name].to(device)
    )

    if cos_sim_a >= cos_sim_b:
        return "more" == ground_truth
    else:
        return "less" == ground_truth


def evaluate_attribute_comparison_task(
    target_font_names,
    text,
    model=model,
    user_choices_count=user_attribute_choices_count,
    inclusive_attributes=inclusive_attributes,
    font_paths=font_paths,
    image_file_dir=None,
    use_clip_like_format=False,
):
    """
    Extract the user choices that contains target_font and evaluate the accuracy
    """
    embedded_prompts = generate_all_attribute_embedded_prompts(
        inclusive_attributes, model=model, use_clip_like_format=use_clip_like_format
    )
    embedded_images = generate_all_fonts_embedded_images(
        font_paths,
        text,
        model=model,
        preprocess=preprocess,
        image_file_dir=image_file_dir,
    )
    result = []
    for user_choice_count in user_choices_count:
        # e.x., (('angular', 'ARSMaquetteWebOne', 'Kenia-Regular'), {'more': 4, 'less': 3})
        (attribute, font_a_name, font_b_name), tmp_ground_truth = user_choice_count

        if attribute not in inclusive_attributes:
            continue

        # if (font_a_name not in target_font_names) and (font_b_name not in target_font_names):
        # continue
        if (font_a_name not in target_font_names) or (
            font_b_name not in target_font_names
        ):
            continue

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
        tmp_result = (attribute, font_a_name, font_b_name, ground_truth, prediction)
        result.append(tmp_result)
    classification_rate = sum([1 for r in result if r[-1]]) / len(result)
    return classification_rate


def generate_all_attribute_embedded_prompts(attributes, model=model, use_clip_like_format=False):
    context_length = model.context_length
    if hasattr(model, "precontext_length"):
        context_length -= model.precontext_length
    with torch.no_grad():
        result = {}
        for attribute in attributes:
            if use_clip_like_format:
                prompt = f"Text written in {attribute} font."
            else:
                prompt = f"{attribute} font"
            embedded_prompt = model.encode_text(tokenize(prompt, context_length=context_length).to(device)).cpu()
            result[attribute] = embedded_prompt
        return result


def evaluate_attribute_comparison_task_for_each_comparison(
    target_font_names,
    text,
    model=model,
    user_choices_count=user_attribute_choices_count,
    inclusive_attributes=inclusive_attributes,
    font_paths=font_paths,
    image_file_dir=None,
    use_clip_like_format=False,
):
    """
    Extract the user choices that contains target_font and evaluate the accuracy
    """
    embedded_prompts = generate_all_attribute_embedded_prompts(
        inclusive_attributes, model=model, use_clip_like_format=use_clip_like_format
    )
    embedded_images = generate_all_fonts_embedded_images(
        font_paths,
        text,
        model=model,
        preprocess=preprocess,
        image_file_dir=image_file_dir,
    )
    result = []
    for user_choice_count in user_choices_count:
        # e.x., (('angular', 'ARSMaquetteWebOne', 'Kenia-Regular'), {'more': 4, 'less': 3})
        (attribute, font_a_name, font_b_name), tmp_ground_truth = user_choice_count

        if attribute not in inclusive_attributes:
            continue

        if (font_a_name not in target_font_names) and (font_b_name not in target_font_names):
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
    classification_rate = sum([r[-2] for r in result]) / sum([r[-1] for r in result])
    return classification_rate


# user similarity task
font_id_txt = "../similarity/fontNames.txt"
user_choices_csv = "../similarity/compsCount.csv"

with open(user_choices_csv, "r") as f:
    tmp_user_choices = f.read().split("\n")

with open(font_id_txt, "r") as f:
    tmp_font_names = f.read().split("\n")
font_names = []
for tmp_font_name in tmp_font_names:
    if tmp_font_name != "":
        font_names.append(tmp_font_name)

user_similarity_choices = []
for tmp_user_choice in tmp_user_choices:
    if tmp_user_choice == "":
        continue
    reference_font_id, font_a_id, font_b_id, vote_count_a, vote_count_b = (
        int(float(e)) for e in tmp_user_choice.split(",") if e != ""
    )
    reference_font_name = font_names[reference_font_id]
    font_a_name = font_names[font_a_id]
    font_b_name = font_names[font_b_id]
    user_choice = [
        reference_font_name,
        font_a_name,
        font_b_name,
        vote_count_a,
        vote_count_b,
    ]
    user_similarity_choices.append(user_choice)


def evaluate_similarity_comparison_task(
    target_font_names,
    text,
    model=model,
    user_choices=user_similarity_choices,
    font_paths=font_paths,
    image_file_dir=None,
):
    embedded_images = generate_all_fonts_embedded_images(
        font_paths,
        text,
        model=model,
        preprocess=preprocess,
        image_file_dir=image_file_dir,
    )
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
        if (reference_font_name not in target_font_names) and (font_a_name not in target_font_names) and (font_b_name not in target_font_names):
            continue

        if vote_count_a == vote_count_b:
            continue

        ground_truth = vote_count_a > vote_count_b
        total_num = vote_count_a + vote_count_b

        embedded_r = embedded_images[reference_font_name].to(device)
        embedded_a = embedded_images[font_a_name].to(device)
        embedded_b = embedded_images[font_b_name].to(device)

        # calculate the cos similarity
        cos_sim_a = torch.cosine_similarity(embedded_r, embedded_a, dim=-1)
        cos_sim_b = torch.cosine_similarity(embedded_r, embedded_b, dim=-1)
        tmp_prediction = cos_sim_a.item() > cos_sim_b.item()
        prediction = ground_truth == tmp_prediction
        if tmp_prediction:
            correct_num = vote_count_a
        else:
            correct_num = vote_count_b
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
    classification_rate = sum([r[-2] for r in result]) / sum([r[-1] for r in result]) 
    return classification_rate


def calculate_cos_sim(embedded_images, embedded_texts):
    # assume each row of embedded_texts is the same

    embedded_images = embedded_images.cpu().detach().numpy()
    embedded_texts = embedded_texts.cpu().detach().numpy()

    norm_embedded_images = np.linalg.norm(embedded_images, axis=1)
    norm_embedded_texts = np.linalg.norm(embedded_texts, axis=1)
    cos_sim = embedded_images @ embedded_texts.T
    cos_sim = cos_sim / np.outer(norm_embedded_images, norm_embedded_texts)
    return cos_sim[:, 0]


def calculate_corr(
    model, dataset: TestDataset, predict_mode=False, device=device, use_dense_clip=False, return_variance=False,
):
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    it = iter(data_loader)
    cos_sim = None
    logits = None
    if dataset.image_file_dir is not None:
        texts_kind_num = 1
    else:
        texts_kind_num = len(dataset.texts_for_font_image)
    with torch.no_grad():
        if return_variance:
            embedded_images = None
        for batch in it:
            images, texts = batch
            images = images.to(device)
            texts = texts.to(device)
            if use_dense_clip:
                tmp_logits = model(images, texts)
                # to numpy
                tmp_logits = tmp_logits.cpu().numpy()
                if logits is None:
                    logits = tmp_logits
                else:
                    logits = np.concatenate((logits, tmp_logits), axis=0)
            else:
                tmp_embedded_images = model.encode_image(images).cpu()
                tmp_embedded_images = tmp_embedded_images / torch.norm(
                    tmp_embedded_images, dim=1, keepdim=True)
                tmp_embedded_texts = model.encode_text(texts).cpu()
                tmp_embedded_texts = tmp_embedded_texts / torch.norm(
                    tmp_embedded_texts, dim=1, keepdim=True)

                # assume each row of embedded_texts is the same
                tmp_cos_sim = calculate_cos_sim(tmp_embedded_images, tmp_embedded_texts)
                if cos_sim is None:
                    cos_sim = tmp_cos_sim
                else:
                    cos_sim = np.concatenate((cos_sim, tmp_cos_sim), axis=0)

                if return_variance:
                    if embedded_images is None:
                        embedded_images = tmp_embedded_images
                    else:
                        embedded_images = torch.cat(
                            (embedded_images, tmp_embedded_images), dim=0
                        )
    if predict_mode:
        if use_dense_clip:
            return logits
        if return_variance:
            variance_of_embedded_images = torch.var(embedded_images, dim=0)
            return cos_sim, variance_of_embedded_images
        return cos_sim

    # calculate the correlation coefficient
    tmp_ground_truth_attribute_values = dataset.flatten_ground_truth_attribute_values()
    ground_truth_attribute_values = torch.cat([tmp_ground_truth_attribute_values for _ in range(texts_kind_num)], dim=0)

    if use_dense_clip:
        corr = np.corrcoef(logits[:, 0], ground_truth_attribute_values)[0, 1]
        return corr, logits, ground_truth_attribute_values
    corr = np.corrcoef(cos_sim, ground_truth_attribute_values)[0, 1]
    if return_variance:
        variance_of_embedded_images = torch.var(embedded_images, dim=0)
        return corr, cos_sim, ground_truth_attribute_values, variance_of_embedded_images
    return corr, cos_sim, ground_truth_attribute_values


def evaluate(
    model,
    texts_for_font_image,
    json_path,
    target_attributes,
    font_dir=font_dir,
):
    model.eval()
    corr_sum = 0
    count = 0
    result = {}
    with torch.no_grad():
        for target_attribute in target_attributes:
            test_data = TestDataset(
                font_dir,
                json_path,
                texts_for_font_image,
                char_size=150,
                attribute_threshold=0,
                target_attributes=[target_attribute],
            )
            corr, cos_sim, ground_truth = calculate_corr(model, test_data)
            # if corr is np.nan
            if np.isnan(corr):
                print(f"nan corr for {target_attribute}")
                continue
            corr_sum += corr
            count += 1
            result[target_attribute] = corr
    if count == 0:
        return 0
    return corr_sum / count, result

def evaluate_correlation_coefficient_of_paired_images(model, image_dataset: PairedImageDataset):
    data_loader = DataLoader(image_dataset, batch_size=64, shuffle=False, num_workers=4)
    model.eval()
    total_loss = 0
    torch.cuda.empty_cache()
    with torch.no_grad():
        for batch in iter(data_loader):
            images_1, images_2 = batch
            images_1 = images_1.to(device)
            images_2 = images_2.to(device)
            image_features_1 = model.encode_image(images_1)
            image_features_2 = model.encode_image(images_2)
            image_features_1 = image_features_1 / image_features_1.norm(
                dim=-1, keepdim=True
            )
            image_features_2 = image_features_2 / image_features_2.norm(
                dim=-1, keepdim=True
            )
            logits_scale = model.logit_scale.exp()
            logits_per_image_1 = (
                logits_scale * image_features_1 @ image_features_2.t()
            )
            logits_per_image_2 = logits_per_image_1.t()
            loss_img = CrossEntropyLoss()
            ground_truth = torch.arange(
                len(images_1), dtype=torch.long, device=device
            )
            total_loss += (
                loss_img(logits_per_image_1, ground_truth)
                + loss_img(logits_per_image_2, ground_truth)
            ) / 2
        del images_1, images_2, image_features_1, image_features_2, logits_per_image_1, logits_per_image_2, ground_truth
        torch.cuda.empty_cache()
    return - total_loss


def evaluate_correlation_coefficient(model, json_path, image_dataset: TestImageDataset, text_dataset: TestTextDataset):
    model.eval()
    target_attributes = text_dataset.target_attributes
    font_names = image_dataset.font_names
    font_to_attributes = json.load(open(json_path, "r"))
    font_to_attributes = {font_name: [float(v) for a, v in attributes.items() if a in target_attributes] for font_name, attributes in font_to_attributes.items() if font_name in font_names}
    ground_truth_attributes = torch.tensor([font_to_attributes[font_name] for font_name in font_names]).T
    ground_truth_attributes = ground_truth_attributes.cpu().detach().numpy()

    image_dataloader = DataLoader(image_dataset, batch_size=32, shuffle=False, num_workers=4)
    text_dataloader = DataLoader(text_dataset, batch_size=32, shuffle=False, num_workers=4)
    embedded_images = None
    for images in iter(image_dataloader):
        images = images.to(device)
        tmp_embedded_images = model.encode_image(images)
        tmp_embedded_images = tmp_embedded_images / torch.norm(
            tmp_embedded_images, dim=1, keepdim=True)
        if embedded_images is None:
            embedded_images = tmp_embedded_images
        else:
            embedded_images = torch.cat((embedded_images, tmp_embedded_images), dim=0)

    embedded_texts = None
    for texts in iter(text_dataloader):
        texts = texts.to(device)
        tmp_embedded_texts = model.encode_text(texts)
        tmp_embedded_texts = tmp_embedded_texts / torch.norm(
            tmp_embedded_texts, dim=1, keepdim=True)
        if embedded_texts is None:
            embedded_texts = tmp_embedded_texts
        else:
            embedded_texts = torch.cat((embedded_texts, tmp_embedded_texts), dim=0)
    
    cos_sims = embedded_texts @ embedded_images.T
    cos_sims = cos_sims.cpu().detach().numpy()
    # embedded_images = embedded_images.cpu().detach().numpy()
    # embedded_texts = embedded_texts.cpu().detach().numpy()
    # norm_embedded_images = np.linalg.norm(embedded_images, axis=1)
    # norm_embedded_texts = np.linalg.norm(embedded_texts, axis=1)
    # cos_sims = embedded_images @ embedded_texts.T
    # cos_sims = cos_sims / np.outer(norm_embedded_images, norm_embedded_texts)
    # cos_sims = cos_sims.T

    corr_sum = 0
    count = 0
    for i in range(len(target_attributes)):
        corr = np.corrcoef(cos_sims[i], ground_truth_attributes[i])[0, 1]
        if np.isnan(corr):
            print(f"nan corr for {target_attributes[i]}")
        else:
            corr_sum += corr
            count += 1
    if count == 0:
        return 0
    else:
        return corr_sum / count



def evaluate_use_dumped_image(model, datasets, target_attributes, use_dense_clip=False, return_variance=False):
    assert len(datasets) == len(target_attributes)
    model.eval()
    corr_sum = 0
    count = 0
    result = {}
    with torch.no_grad():
        variance_sum = 0
        for dataset, target_attribute in zip(datasets, target_attributes):
            if return_variance:
                corr, cos_sim, ground_truth, tmp_variance_of_embedded_images = calculate_corr(
                    model, dataset, use_dense_clip=use_dense_clip, return_variance=return_variance
                )
                variance_sum += torch.sum(tmp_variance_of_embedded_images)
            else:
                corr, cos_sim, ground_truth = calculate_corr(
                    model, dataset, use_dense_clip=use_dense_clip, return_variance=return_variance
                )
            # if corr is np.nan
            if np.isnan(corr):
                print(f"nan corr for {target_attribute}")
                continue
            corr_sum += corr
            count += 1
            result[target_attribute] = corr
    if count == 0:
        return 0
    if return_variance:
        return corr_sum / count, result, variance_sum / count
    return corr_sum / count, result


def evaluate_used_dumped_image_use_score(model, dataset):
    loss_mse = torch.nn.MSELoss()
    model.eval()
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    it = iter(data_loader)
    loss = None
    torch.cuda.empty_cache()
    with torch.no_grad():
        for i, batch in enumerate(it):
            images, prompts, scores = batch
            images = images.to(device)
            prompts = prompts.to(device)
            scores = torch.tensor(scores, dtype=torch.float16)
            scores = scores.to(device).cpu()
            logits = model(images, prompts).cpu()
            if loss is None:
                loss = loss_mse(logits, scores)
            else:
                loss += loss_mse(logits, scores)
    return loss
