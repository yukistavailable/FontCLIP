import gc
import json
import os
from PIL import Image, ImageFont
import random
from tqdm import tqdm
from typing import Union, List, Optional, Tuple
import string


import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image, to_tensor, pil_to_tensor

from utils.tokenizer import tokenize
from models.init_model import (
    my_preprocess,
    preprocess_for_aug,
    preprocess_for_normalize,
    model,
)
from utils.initialize_font_data import all_attributes
from utils.transform_image import (
    my_transform,
    my_convert_to_rgb,
    draw_text_with_new_lines,
    transform_for_resize,
)


class MyDataset(Dataset):
    @staticmethod
    def generate_prompt(
        attribute: str,
        negative: bool = False,
        rich: bool = False,
        score: Optional[float] = None,
        single_character: bool = False,
    ) -> str:
        """
        generate a prompt like "bold font" bease on the attribute score

        Parameters
        ----------
        attribute : str
            attribute ex. "bold" or "not bold"
        negative : bool, optional
            negative or not, by default False
        rich : bool, optional
            rich or not, by default False
            if rich is True, score must be specified
            if rich is True, return prompt like "very bold font"
        score : float, optional
            score of attribute, by default None
        single_character : bool, optional
            single character or not, by default False
            if single_character is True, return prompt like "a photo of a character in a bold font"

        Returns
        -------
        str
            prompt
        """
        if single_character:
            if negative:
                return f"a photo of a character in a not {attribute} font"
            return f"a photo of a character in a {attribute} font"
        elif rich:
            assert score is not None
            if score >= 75:
                return f"very {attribute} font"
            elif score >= 50:
                return f"{attribute} font"
            elif score >= 25:
                return f"not so {attribute} font"
            else:
                return f"not {attribute} font"
        else:
            if negative:
                return f"not {attribute} font"
            return f"{attribute} font"

    @staticmethod
    def attribute_to_index(attribute: str) -> int:
        """
        convert attribute to index in all_attributes

        Parameters
        ----------
        attribute : str
            attribute ex. "bold"

        Returns
        -------
        int
            index of attribute
        """
        return all_attributes.index(attribute) + 1

    @staticmethod
    def signed_attribute_to_index(attribute: str, attribute_num: int = 37) -> int:
        """
        convert attribute to index

        Parameters
        ----------
        attribute : str
            attribute ex. "bold" or "not bold"
        attribute_num : int, optional

        Returns
        -------
        int
            index of attribute
        """
        index_length = attribute_num * 2 + 1
        if attribute.startswith("not"):
            unsigned_attribute = attribute[4:]
            return index_length - MyDataset.attribute_to_index(unsigned_attribute)
        return MyDataset.attribute_to_index(attribute)

    @staticmethod
    def index_to_signed_attribute(
        idx: int, attribute_num: int = 37, all_attributes: List[str] = all_attributes
    ) -> str:
        """
        convert index to attribute

        Parameters
        ----------
        idx : int
            index of attribute
        attribute_num : int, optional

        Returns
        -------
        str
            attribute ex. "bold" or "not bold"
        """
        index_length = attribute_num * 2 + 1
        # idx should be more equal than 1 because 1 is plussed in attribute_to_index
        assert 1 <= idx <= index_length, f"idx must be in [1, {index_length}]"

        # if idx is more than attribute_num, it means that the attribute is negative, so return "not attribute"
        if idx > attribute_num:
            tmp_idx = index_length - idx
            return f"not {all_attributes[tmp_idx - 1]}"
        return all_attributes[idx - 1]

    @staticmethod
    def generate_multiple_attributes_prompt(
        attributes: List[str],
        use_random: bool = False,
        max_sample_num: int = 3,
        p: List[float] = None,
    ) -> Tuple[str, torch.Tensor]:
        """
        generate a prompt including multiple attribute nums like "italic, formal, legible font" bease on each attribute scores

        Parameters
        ----------
        attributes : List[str]
            attributes to be used in the dataset ex. ["angular", "not artistic", "not attention-grabbing", ..., "wide"]
        use_random : bool, optional
            use random or not, by default False
        max_sample_num : int, optional
            max sample num, by default 3
        p : List[float], optional
            probability of each attribute, by default None
            probability of each attribute is calculated by the attribute score

        Returns
        -------
        str
            prompt ex. "italic, formal, legible font"
        torch.Tensor
            signed attribute indices ex. [24, 18, 25]
        """
        max_sample_num = min(max_sample_num, len(attributes))
        if use_random:
            sample_num = random.randint(1, max_sample_num)
            if p is None:
                attributes = random.sample(attributes, sample_num)
            else:
                attributes = np.random.choice(
                    attributes, sample_num, p=p, replace=False
                )
        attribute_num = len(all_attributes)
        signed_attribute_indices = torch.Tensor(
            [
                MyDataset.signed_attribute_to_index(attribute, attribute_num)
                for attribute in attributes
            ]
        )
        signed_attribute_indices = torch.cat(
            (
                signed_attribute_indices,
                torch.zeros(max_sample_num - len(attributes), dtype=torch.float32),
            )
        )
        return (
            ", ".join([f"{attribute}" for attribute in attributes]) + " font",
            signed_attribute_indices,
        )

    def do_apotosis(self):
        if hasattr(self, "font_text_to_image_tensors"):
            del self.font_text_to_image_tensors

        if hasattr(self, "font_text_to_unnormalized_image_tensors"):
            del self.font_text_to_unnormalized_image_tensors

        if hasattr(self, "font_to_attributes"):
            del self.font_to_attributes

        if hasattr(self, "font_to_attribute_values"):
            del self.font_to_attribute_values

        if hasattr(self, "font_to_attribute_indices"):
            del self.font_to_attribute_indices

        gc.collect()

    def create_font_idx_signed_attribute_matrix(self):
        """
        create font_idx_signed_attribute_matrix
        The matrix shape is (font_num, attribute_num * 2 + 1)
        The value of each element is 1 or -1, and the value of the element is 1 if the font has the attribute above threshold, otherwise -1.
        """
        font_idx_signed_attribute_matrix = torch.zeros(
            (len(self.font_paths), len(all_attributes) * 2 + 1), dtype=torch.float32
        )
        with open(self.json_path, "r") as f:
            tmp_font_to_attribute_values = json.load(f)
            for i, font_name in enumerate(self.font_names):
                vs = tmp_font_to_attribute_values[font_name]
                assert len(vs) == len(all_attributes)
                for j, attribute in enumerate(all_attributes):
                    v = vs[attribute]
                    if float(v) >= self.attribute_threshold:
                        font_idx_signed_attribute_matrix[i][j + 1] = 1
                        font_idx_signed_attribute_matrix[i][-(j + 1)] = -1
                    else:
                        font_idx_signed_attribute_matrix[i][j + 1] = -1
                        font_idx_signed_attribute_matrix[i][-(j + 1)] = 1
        return font_idx_signed_attribute_matrix

    # TODO: make it faster
    def mask_font_idx_signed_attribute_matrix(
        self, font_indices, signed_attribute_indices
    ):
        assert len(font_indices) == len(signed_attribute_indices)
        mask_matrix = torch.ones(
            (len(font_indices), len(font_indices)), dtype=torch.float32
        )
        for i, font_index in enumerate(font_indices):
            for j, signed_attribute_index in enumerate(signed_attribute_indices):
                for signed_attribute_idx in signed_attribute_index:
                    if (
                        self.font_idx_signed_attribute_matrix[font_index][
                            signed_attribute_idx
                        ]
                        == 1
                    ):
                        mask_matrix[i][j] = 0
                        break
        return mask_matrix

    def mask_font_idx_signed_attribute_matrix_ground_truth(
        self, font_indices, signed_attribute_indices
    ):
        assert len(font_indices) == len(signed_attribute_indices)
        mask_matrix = torch.ones(
            (len(font_indices), len(font_indices)), dtype=torch.float32
        )
        for i, font_index in enumerate(font_indices):
            for j, signed_attribute_index in enumerate(signed_attribute_indices):
                for signed_attribute_idx in signed_attribute_index:
                    if (
                        self.font_idx_signed_attribute_matrix[font_index][
                            signed_attribute_idx
                        ]
                        == -1
                    ):
                        mask_matrix[i][j] = 0
                        break
        return mask_matrix

    def mask_font_idx_signed_attribute_matrix_ground_truth_fast(
        self, font_indices, signed_attribute_indices
    ):
        assert len(font_indices) == len(signed_attribute_indices)

        # Pre-allocate the mask matrix
        mask_matrix = torch.ones(
            (len(font_indices), len(font_indices)), dtype=torch.float32
        )

        # Vectorized computation for the mask matrix
        for i, font_index in enumerate(font_indices):
            # Get the corresponding row from the pre-computed matrix
            font_row = self.font_idx_signed_attribute_matrix[font_index]

            for j, signed_attribute_index in enumerate(signed_attribute_indices):
                # Check if any signed attribute index in the font row is -1
                if any(font_row[idx] == -1 for idx in signed_attribute_index):
                    mask_matrix[i, j] = 0

        return mask_matrix

    def update_font_attribute_counts(self):
        if self.use_multiple_attributes:
            self.font_attribute_counts = (
                len(self.font_paths) * self.sample_num_each_epoch
            )
        else:
            tmp_font_attribute_counts = 0
            for attribute_values in self.font_to_attributes.values():
                tmp_font_attribute_counts += len(attribute_values)
            self.font_attribute_counts = tmp_font_attribute_counts

    def __init__(
        self,
        font_dir,
        json_path,
        texts_for_font_image,
        char_size=150,
        attribute_threshold=50,
        attribute_under_threshold=50,
        use_negative=False,
        use_weight=False,
        use_multiple_attributes=False,
        use_random_attributes=False,
        random_prompts_num=1000,
        max_sample_num=3,
        sample_num_each_epoch=5,
        rich_prompt=False,
        preprocess=None,
        dump_image=False,
        font_text_to_image_tensors=None,
        exclusive_attributes=None,
        image_file_dir=None,
        single_character=False,
        use_score=False,
        geta=0.5,
        use_negative_loss=False,
        use_contrastive_image_loss=False,
        use_vae_loss=False,
        vae_target_characters=string.ascii_uppercase,
        vae_target_characters_size=64,
        store_unnormalized_image=False,
        n_px_of_unnormalized_image=224,
        context_length=77,
    ):
        self.char_size = char_size
        self.attribute_threshold = attribute_threshold
        self.json_path = json_path
        self.texts_for_font_image = texts_for_font_image
        self.use_negative = use_negative
        self.use_weight = use_weight
        self.use_score = use_score
        self.use_multiple_attributes = use_multiple_attributes
        if self.use_multiple_attributes:
            self.use_weight = False
        self.use_random_attributes = use_random_attributes
        self.random_prompts_num = random_prompts_num
        self.max_sample_num = max_sample_num
        self.sample_num_each_epoch = sample_num_each_epoch
        self.predict_mode = False
        self.preprocess = preprocess
        self.dump_image = dump_image
        self.font_text_to_image_tensors = font_text_to_image_tensors
        self.exclusive_attributes = (
            exclusive_attributes if exclusive_attributes is not None else []
        )
        self.image_file_dir = image_file_dir
        self.rich_prompt = rich_prompt
        self.single_character = single_character
        self.use_negative_loss = use_negative_loss
        self.use_contrastive_image_loss = use_contrastive_image_loss
        self.use_vae_loss = use_vae_loss
        self.vae_target_characters = vae_target_characters
        self.vae_target_character_size = vae_target_characters_size
        self.store_unnormalized_image = store_unnormalized_image
        self.n_px_of_unnormalized_image = n_px_of_unnormalized_image
        self.font_idx_signed_attribute_matrix = None
        self.font_text_to_unnormalized_image_tensors = None
        self.context_length = context_length

        assert (not self.dump_image) or (self.font_text_to_image_tensors is None)

        if self.json_path is None:
            print("No json path, use predict mode.")
            self.predict_mode = True
            self.font_to_attributes = {}

        if not self.predict_mode:
            with open(self.json_path, "r") as f:
                tmp_font_to_attribute_values = json.load(f)
            self.font_to_attributes = {}
            self.font_to_attribute_indices = {}

            if self.use_multiple_attributes:
                for k, v in tqdm(tmp_font_to_attribute_values.items()):
                    if self.use_negative:
                        tmp_attributes = [
                            a
                            for a, v_v in v.items()
                            if (a not in self.exclusive_attributes)
                            and (float(v_v) >= self.attribute_threshold)
                        ] + [
                            f"not {a}"
                            for a, v_v in v.items()
                            if (a not in self.exclusive_attributes)
                            and (float(v_v) < attribute_under_threshold)
                        ]
                        p = [
                            float(v_v) / 100
                            for a, v_v in v.items()
                            if (a not in self.exclusive_attributes)
                            and (float(v_v) >= self.attribute_threshold)
                        ] + [
                            1 - float(v_v) / 100
                            for a, v_v in v.items()
                            if (a not in self.exclusive_attributes)
                            and (float(v_v) < attribute_under_threshold)
                        ]
                    else:
                        tmp_attributes = [
                            a
                            for a, v_v in v.items()
                            if (a not in self.exclusive_attributes)
                            and (float(v_v) >= self.attribute_threshold)
                        ]
                        p = [
                            float(v_v) / 100
                            for a, v_v in v.items()
                            if (a not in self.exclusive_attributes)
                            and (float(v_v) >= self.attribute_threshold)
                        ]

                    # tmp = [tokenize(self.generate_multiple_attributes_prompt([attribute], use_random=False)) for attribute in tmp_attributes]
                    tmp = []
                    tmp_attribute_indices = []
                    # tmp = [self.generate_multiple_attributes_prompt([attribute], use_random=False) for attribute in tmp_attributes]
                    if self.use_random_attributes:
                        # normalize p
                        p = [p_i / sum(p) for p_i in p]
                        for _ in range(random_prompts_num):
                            (
                                row_text,
                                signed_attribute_indices,
                            ) = self.generate_multiple_attributes_prompt(
                                tmp_attributes,
                                use_random=True,
                                max_sample_num=self.max_sample_num,
                                p=p,
                            )
                            tmp.append(
                                tokenize(row_text, context_length=self.context_length)
                            )
                            tmp_attribute_indices.append(signed_attribute_indices)
                    self.font_to_attributes[k] = tmp
                    self.font_to_attribute_indices[k] = tmp_attribute_indices
            else:
                # TODO: add row text to self.font_to_attributes_row
                for k, v in tmp_font_to_attribute_values.items():
                    tmp = []
                    for a, v_v in v.items():
                        if a in self.exclusive_attributes:
                            continue
                        if self.rich_prompt:
                            tmp.append(
                                tokenize(
                                    self.generate_prompt(
                                        a,
                                        rich=True,
                                        score=float(v_v),
                                    ),
                                    context_length=self.context_length,
                                )
                            )
                        else:
                            if float(v_v) >= self.attribute_threshold:
                                tmp.append(
                                    tokenize(
                                        self.generate_prompt(
                                            a,
                                            single_character=self.single_character,
                                        ),
                                        context_length=self.context_length,
                                    )
                                )
                                # tmp.append(self.generate_prompt(a, single_character=self.single_character))
                            elif float(v_v) <= attribute_under_threshold:
                                tmp.append(
                                    tokenize(
                                        self.generate_prompt(
                                            a,
                                            negative=self.use_negative,
                                            single_character=self.single_character,
                                        ),
                                        context_length=self.context_length,
                                    )
                                )
                                # tmp.append(self.generate_prompt(a, negative=self.use_negative, single_character=self.single_character))
                    self.font_to_attributes[k] = tmp
            self.attribute_kind_num = len(
                self.font_to_attributes[list(self.font_to_attributes.keys())[0]]
            )

            self.font_to_attribute_scores = None
            if use_score:
                self.font_to_attribute_scores = {
                    k: [float(v_v) / 100 for a, v_v in v.items()]
                    for k, v in tmp_font_to_attribute_values.items()
                }
            if self.use_negative:
                if self.use_weight:
                    self.font_to_attribute_weights = {
                        k: [
                            float(v_v) / 100 + geta
                            if float(v_v) >= self.attribute_threshold
                            else 1 - float(v_v) / 100 + geta
                            for a, v_v in v.items()
                        ]
                        for k, v in tmp_font_to_attribute_values.items()
                    }
            else:
                if self.use_weight:
                    self.font_to_attribute_weights = {
                        k: [
                            float(v_v) / 100 - self.attribute_threshold / 100
                            for a, v_v in v.items()
                        ]
                        for k, v in tmp_font_to_attribute_values.items()
                        if k not in self.exclusive_attributes
                    }

        # if font_dir is list
        if isinstance(font_dir, list):
            print("font_dir is list")
            tmp_font_paths = font_dir
        else:
            tmp_font_paths = [os.path.join(font_dir, f) for f in os.listdir(font_dir)]
        self.font_paths = []
        self.font_names = []
        for font_path in tmp_font_paths:
            font_name = os.path.splitext(os.path.basename(font_path))[0]
            if self.predict_mode or (font_name in self.font_to_attributes.keys()):
                self.font_names.append(font_name)
                self.font_paths.append(font_path)

        self.fonts = [
            ImageFont.truetype(font_path, char_size) for font_path in self.font_paths
        ]
        self.update_font_attribute_counts()

        self.dumped_images = None
        if self.dump_image:
            self.dump_image_tensor()

        if self.use_negative_loss:
            # self.font_idx_signed_attribute_matrix is referenced at mask_font_idx_signed_attribute_matrix in order to calculate mask matrix in training
            self.font_idx_signed_attribute_matrix = (
                self.create_font_idx_signed_attribute_matrix()
            )

        self.font_to_vae_target_character_tensor = {}
        self.character_to_one_hot_vector = {}
        if self.use_vae_loss:
            preprocess_for_resize_ = transform_for_resize(
                self.vae_target_character_size
            )
            for font_name, font_path in zip(self.font_names, self.font_paths):
                font = ImageFont.truetype(font_path, self.vae_target_character_size)
                self.font_to_vae_target_character_tensor[font_name] = torch.stack(
                    [
                        preprocess_for_resize_(
                            self.create_vae_target_character_image(
                                c, font, char_size=self.vae_target_character_size
                            )
                        )
                        for c in self.vae_target_characters
                    ]
                ).to(model.dtype)

            for c in self.vae_target_characters:
                one_hot_vector = torch.zeros(len(self.vae_target_characters)).to(
                    model.dtype
                )
                one_hot_vector[self.vae_target_characters.index(c)] = 1
                self.character_to_one_hot_vector[c] = one_hot_vector

    def __len__(self):
        if self.predict_mode:
            if self.image_file_dir is not None:
                return len(self.font_paths)
            return len(self.font_paths) * len(self.texts_for_font_image)
        if self.use_multiple_attributes:
            if self.image_file_dir is not None:
                return len(self.font_paths) * self.sample_num_each_epoch
            return (
                len(self.font_paths)
                * len(self.texts_for_font_image)
                * self.sample_num_each_epoch
            )

        count = 0
        for font_path in self.font_paths:
            font_name = os.path.splitext(os.path.basename(font_path))[0]
            count += len(self.font_to_attributes[font_name])
        if self.image_file_dir is not None:
            return int(count)
        return int(count * len(self.texts_for_font_image))

    @classmethod
    def create_vae_target_character_image(cls, char, font, char_size=64):
        assert len(char) == 1
        width = char_size
        height = char_size
        image = draw_text_with_new_lines(char, font, width, height)
        return image.convert("L")

    def create_image(self, text, font, font_path=None, no_preprocess=False, padding=0):
        if self.image_file_dir:
            assert font_path is not None
            font_name = os.path.splitext(os.path.basename(font_path))[0]
            image_file_path = os.path.join(self.image_file_dir, font_name + ".png")
            image = Image.open(image_file_path)
            image = my_convert_to_rgb(image)
            image = self.preprocess(image)
            return image

        else:
            if len(text) == 1:
                width = self.char_size + int(padding) * 2
                height = self.char_size + int(padding) * 2
            else:
                line_num = text.count("\n") + 1
                if self.dump_image:
                    width = (
                        int((self.char_size * 0.66) * len(text) / line_num)
                        + int(padding) * 2
                    )
                else:
                    width = (
                        int(self.char_size * len(text) * 1.8 / line_num)
                        + int(padding) * 2
                    )
                height = int(self.char_size) * line_num + int(padding) * 2

            image = draw_text_with_new_lines(text, font, width, height)
            if no_preprocess:
                return image

            image = self.preprocess(image)
            return image

    def __getitem__(self, idx):
        # Note that self.font_attribute_counts = len(self.font_paths) * self.sample_num_each_epoch if self.use_multiple_attributes else sum([len(v) for v in self.font_to_attributes.values()])
        text_idx = idx // self.font_attribute_counts
        text = None
        if self.image_file_dir is None:
            text = self.texts_for_font_image[text_idx]

        # Note that idx < self.font_attribute_counts * len(self.texts_for_font_image) if self.use_multiple_attributes else self.font_attribute_counts * len(self.texts_for_font_image)
        idx = idx % self.font_attribute_counts  # idx is the index in the specific text
        if self.use_multiple_attributes:
            font_idx = idx // self.sample_num_each_epoch
            font_path = self.font_paths[font_idx]
            font_name = os.path.splitext(os.path.basename(font_path))[0]
            # attribute_index = random.randint(0, self.random_prompts_num)
            attribute_idx = random.randint(
                0, len(self.font_to_attributes[font_name]) - 1
            )
        else:
            font_idx = 0
            while idx >= len(
                self.font_to_attributes[
                    os.path.splitext(os.path.basename(self.font_paths[font_idx]))[0]
                ]
            ):
                tmp_idx = len(
                    self.font_to_attributes[
                        os.path.splitext(os.path.basename(self.font_paths[font_idx]))[0]
                    ]
                )
                idx -= tmp_idx
                font_idx += 1
            attribute_idx = idx
            font_path = self.font_paths[font_idx]
            font_name = os.path.splitext(os.path.basename(font_path))[0]

        font = self.fonts[font_idx]
        tokenized_prompt = self.font_to_attributes[font_name][attribute_idx]

        # vae target character
        vae_target_character_tensor = None
        one_hot_vector = None
        if self.use_vae_loss:
            tmp_font_path = self.font_paths[font_idx]
            tmp_font_name = os.path.splitext(os.path.basename(tmp_font_path))[0]
            character_idx = random.randint(0, len(self.vae_target_characters) - 1)
            vae_target_character_tensor = self.font_to_vae_target_character_tensor[
                tmp_font_name
            ][character_idx]
            one_hot_vector = self.character_to_one_hot_vector[
                self.vae_target_characters[character_idx]
            ]

        image_2 = None
        if self.font_text_to_image_tensors is not None:
            pool_idx = text_idx * len(self.fonts) + font_idx
            images = self.font_text_to_image_tensors[pool_idx]
            # randomly sample one image
            random_index = random.randint(0, len(images) - 1)
            image = images[random_index]
            if self.use_contrastive_image_loss:
                random_index_2 = random.randint(0, len(images) - 1)
                image_2 = images[random_index_2]

            if self.store_unnormalized_image:
                assert self.font_text_to_unnormalized_image_tensors is not None
                unnormalized_images = self.font_text_to_unnormalized_image_tensors[
                    pool_idx
                ]
                unnormalized_image = unnormalized_images[random_index]

        else:
            if self.dump_image:
                image_idx = text_idx * len(self.fonts) + font_idx
                image = self.dumped_images[image_idx]
                # assert self.font_text_to_unnormalized_image_tensors is not None
                # unnormalized_images = self.font_text_to_unnormalized_image_tensors[]
                # unnormalized_image = unnormalized_images[random_index]
            else:
                image = self.create_image(text, font, font_path)

        if self.use_score:
            score = self.font_to_attribute_scores[font_name][attribute_idx]
            return image, tokenized_prompt.squeeze(0), score

        if self.use_weight:
            weight = self.font_to_attribute_weights[font_name][attribute_idx]
            return image, tokenized_prompt.squeeze(0), weight

        if self.use_multiple_attributes and self.use_negative_loss:
            attribute_indices = self.font_to_attribute_indices[font_name][attribute_idx]
            # unsinged_attribute_idx = [tmp_attribute_idx - 1 if tmp_attribute_idx <= len(all_attributes) else len(all_attributes)*2+1 - tmp_attribute_idx - 1 for tmp_attribute_idx in attribute_indices]
            if self.store_unnormalized_image:
                return (
                    image,
                    unnormalized_image,
                    tokenized_prompt.squeeze(0),
                    font_idx,
                    attribute_indices.type(torch.int64),
                )
            elif self.use_contrastive_image_loss:
                assert image_2 is not None
                if vae_target_character_tensor is not None:
                    return (
                        image,
                        image_2,
                        tokenized_prompt.squeeze(0),
                        font_idx,
                        attribute_indices.type(torch.int64),
                        vae_target_character_tensor,
                        one_hot_vector,
                    )
                return (
                    image,
                    image_2,
                    tokenized_prompt.squeeze(0),
                    font_idx,
                    attribute_indices.type(torch.int64),
                )
            elif vae_target_character_tensor is not None:
                return (
                    image,
                    tokenized_prompt.squeeze(0),
                    font_idx,
                    attribute_indices.type(torch.int64),
                    vae_target_character_tensor,
                    one_hot_vector,
                )
            return (
                image,
                tokenized_prompt.squeeze(0),
                font_idx,
                attribute_indices.type(torch.int64),
            )
        elif self.store_unnormalized_image:
            return image, unnormalized_image, tokenized_prompt.squeeze(0)
        return image, tokenized_prompt.squeeze(0)

    def set_preprocess(self, preprocess):
        self.preprocess = preprocess

    def set_font_text_to_image_tensors(self, font_text_to_image_tensors):
        self.font_text_to_image_tensors = font_text_to_image_tensors

    def set_font_text_to_unnormalized_image_tensors(
        self, font_text_to_unnormalized_image_tensors
    ):
        self.font_text_to_unnormalized_image_tensors = (
            font_text_to_unnormalized_image_tensors
        )

    def dump_image_tensor(self):
        self.dumped_images = []

        # trick
        self.dump_image = False

        # the image num is font num * text num
        if self.image_file_dir is None:
            for text in self.texts_for_font_image:
                for font, font_path in zip(self.fonts, self.font_paths):
                    image = self.create_image(text, font, font_path)
                    self.dumped_images.append(image)
        else:
            for font, font_path in zip(self.fonts, self.font_paths):
                image = self.create_image(None, font, font_path)
                self.dumped_images.append(image)

        self.dump_image = True


class TestDataset(MyDataset):
    def __init__(
        self,
        font_dir,
        json_path,
        texts_for_font_image,
        char_size=150,
        attribute_threshold=0,
        target_attributes=None,
        preprocess=None,
        dump_image=False,
        image_file_dir=None,
        single_character=False,
        use_score=False,
        context_length=77,
    ):
        super().__init__(
            font_dir,
            json_path,
            texts_for_font_image=texts_for_font_image,
            char_size=char_size,
            attribute_threshold=attribute_threshold,
            preprocess=preprocess,
            dump_image=False,
            image_file_dir=image_file_dir,
            single_character=single_character,
            use_score=use_score,
            context_length=context_length,
        )
        if self.predict_mode:
            assert target_attributes is not None

        self.target_attributes = target_attributes
        if not self.predict_mode:
            with open(self.json_path, "r") as f:
                self.font_to_attribute_values = json.load(f)
            self.font_to_attribute_values = {
                k: {a_k: float(a_v) for a_k, a_v in v.items()}
                for k, v in self.font_to_attribute_values.items()
            }

            if self.target_attributes is not None:
                self.font_to_target_attribute_values = {
                    k: {
                        a_k: float(a_v)
                        for a_k, a_v in v.items()
                        if (a_v >= attribute_threshold) and (a_k in target_attributes)
                    }
                    for k, v in self.font_to_attribute_values.items()
                }
                self.font_to_attributes = {
                    k: [
                        tokenize(
                            self.generate_prompt(
                                a_k,
                                single_character=self.single_character,
                            ),
                            context_length=self.context_length,
                        )
                        for a_k in v.keys()
                    ]
                    for k, v in self.font_to_target_attribute_values.items()
                }
            # else:
            # self.font_to_target_attribute_values = {k: {a_k: float(a_v) for a_k, a_v in v.items() if (a_v >= attribute_threshold)} for k, v in self.font_to_attribute_values.items()}
        else:
            if self.target_attributes is not None:
                self.font_to_attributes = {
                    os.path.splitext(os.path.basename(k))[0]: [
                        tokenize(
                            self.generate_prompt(a),
                            context_length=self.context_length,
                        )
                        for a in target_attributes
                    ]
                    for k in self.font_paths
                }
            else:
                raise ValueError("target_attributes must be specified in predict mode")

        self.update_font_attribute_counts()
        if dump_image:
            self.dump_image_tensor()

    def __len__(self):
        if self.predict_mode:
            return (
                len(self.font_paths)
                * len(self.texts_for_font_image)
                * len(self.target_attributes)
            )
        else:
            return super().__len__()

    def flatten_ground_truth_attribute_values(self):
        flattened_attribute_values = []
        for font_path in self.font_paths:
            font_name = os.path.splitext(os.path.basename(font_path))[0]
            flattened_attribute_values.extend(
                list(self.font_to_target_attribute_values[font_name].values())
            )
        return torch.tensor(flattened_attribute_values)


class TestImageDataset(Dataset):
    def __init__(
        self,
        font_dir,
        json_path,
        text_for_font_image,
        char_size=150,
        preprocess=None,
        dump_image=False,
        image_file_dir=None,
    ):
        # super init
        super().__init__()
        self.font_dir = font_dir
        self.json_path = json_path
        self.text_for_font_image = text_for_font_image
        self.char_size = char_size
        self.preprocess = preprocess
        self.image_file_dir = image_file_dir
        self.font_json = json.load(open(self.json_path, "r"))
        # if font_dir is list
        if isinstance(font_dir, list):
            print("font_dir is list")
            tmp_font_paths = font_dir
        else:
            tmp_font_paths = [os.path.join(font_dir, f) for f in os.listdir(font_dir)]
        self.font_paths = []
        self.font_names = []
        for font_path in tmp_font_paths:
            font_name = os.path.splitext(os.path.basename(font_path))[0]
            if font_name in list(self.font_json.keys()):
                self.font_names.append(font_name)
                self.font_paths.append(font_path)
        self.fonts = [
            ImageFont.truetype(font_path, char_size) for font_path in self.font_paths
        ]

        if dump_image:
            self.dump_image_tensor()

    def create_image(self, text, font, font_path=None, no_preprocess=False, padding=0):
        if self.image_file_dir:
            assert font_path is not None
            font_name = os.path.splitext(os.path.basename(font_path))[0]
            image_file_path = os.path.join(self.image_file_dir, font_name + ".png")
            image = Image.open(image_file_path)
            image = my_convert_to_rgb(image)
            image = self.preprocess(image)
            return image

        else:
            if len(text) == 1:
                width = self.char_size + int(padding) * 2
                height = self.char_size + int(padding) * 2
            else:
                line_num = text.count("\n") + 1
                if self.dump_image:
                    width = (
                        int((self.char_size * 0.66) * len(text) / line_num)
                        + int(padding) * 2
                    )
                else:
                    width = (
                        int(self.char_size * len(text) * 1.8 / line_num)
                        + int(padding) * 2
                    )
                height = int(self.char_size) * line_num + int(padding) * 2

            image = draw_text_with_new_lines(text, font, width, height)
            if no_preprocess:
                return image

            image = self.preprocess(image)
            return image

    def dump_image_tensor(self):
        self.dumped_images = []
        # trick
        self.dump_image = False
        # the image num is font num * text num
        if self.image_file_dir is None:
            for font, font_path in zip(self.fonts, self.font_paths):
                image = self.create_image(self.text_for_font_image, font, font_path)
                self.dumped_images.append(image)
        else:
            for font, font_path in zip(self.fonts, self.font_paths):
                image = self.create_image(None, font, font_path)
                self.dumped_images.append(image)
        self.dump_image = True

    def __len__(self):
        return len(self.font_paths)

    def __getitem__(self, idx):
        return self.dumped_images[idx]


class TestTextDataset(Dataset):
    def __init__(
        self,
        target_attributes=None,
        context_length=77,
    ):
        super().__init__()
        self.target_attributes = target_attributes
        self.context_length = context_length

        self.prompts = []
        if self.target_attributes is not None:
            for a in target_attributes:
                self.prompts.append(
                    tokenize(
                        MyDataset.generate_prompt(a),
                        context_length=self.context_length,
                    )
                )

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx][0]


def set_image_tensors(dataset: MyDataset, preprocess=my_preprocess, sample_num=5):
    dataset.set_preprocess(preprocess)
    font_text_to_image_tensors = []
    font_text_to_unnormalized_image_tensors = []
    preprocess_for_resize = transform_for_resize(
        n_px=dataset.n_px_of_unnormalized_image
    )
    if dataset.n_px_of_unnormalized_image == 224:
        preprocess_for_resize = to_tensor
    if dataset.image_file_dir is not None:
        print("load image tensors from image files ...")
        assert len(dataset.texts_for_font_image) == 1
        for font_path in tqdm(dataset.font_paths):
            font_name = os.path.splitext(os.path.basename(font_path))[0]
            image_file_path = os.path.join(dataset.image_file_dir, font_name + ".png")
            image = Image.open(image_file_path)
            image = my_convert_to_rgb(image)
            if dataset.store_unnormalized_image:
                unnormalized_images = [
                    preprocess_for_aug(image) for _ in range(sample_num)
                ]
                images = [
                    preprocess_for_normalize(unnormalized_image)
                    for unnormalized_image in unnormalized_images
                ]
                unnormalized_images = [
                    preprocess_for_resize(to_pil_image(unnormalized_image).convert("L"))
                    for unnormalized_image in unnormalized_images
                ]
                font_text_to_unnormalized_image_tensors.append(
                    torch.stack(unnormalized_images).to(model.dtype)
                )
            else:
                images = [preprocess(image).to(model.dtype) for _ in range(sample_num)]
            font_text_to_image_tensors.append(torch.stack(images))
    else:
        print("create image tensors from font files ...")
        for text in dataset.texts_for_font_image:
            for font in tqdm(dataset.fonts):
                # images = []
                # for _ in range(sample_num):
                #     image = dataset.create_image(text, font)
                #     images.append(image)
                unpreprocessed_image = dataset.create_image(
                    text, font, no_preprocess=True
                )
                if dataset.store_unnormalized_image:
                    unnormalized_images = [
                        preprocess_for_aug(unpreprocessed_image)
                        for _ in range(sample_num)
                    ]
                    images = [
                        preprocess_for_normalize(unnormalized_image)
                        for unnormalized_image in unnormalized_images
                    ]
                    unnormalized_images = [
                        preprocess_for_resize(
                            to_pil_image(unnormalized_image).convert("L")
                        )
                        for unnormalized_image in unnormalized_images
                    ]
                    font_text_to_unnormalized_image_tensors.append(
                        torch.stack(unnormalized_images).to(model.dtype)
                    )
                else:
                    images = [
                        preprocess(unpreprocessed_image).to(model.dtype)
                        for _ in range(sample_num)
                    ]
                font_text_to_image_tensors.append(torch.stack(images).to(model.dtype))
    if dataset.store_unnormalized_image:
        dataset.set_font_text_to_unnormalized_image_tensors(
            font_text_to_unnormalized_image_tensors
        )
    dataset.set_font_text_to_image_tensors(font_text_to_image_tensors)


def set_image_tensors(
    dataset: MyDataset,
    preprocess=my_preprocess,
    sample_num=5,
    padding=0,
    color_jitter_sample_num=0,
):
    color_jitter_preprocess = my_transform(do_color_jitter=True)
    dataset.set_preprocess(preprocess)
    font_text_to_image_tensors = []
    font_text_to_unnormalized_image_tensors = []
    preprocess_for_resize = transform_for_resize(
        n_px=dataset.n_px_of_unnormalized_image
    )
    if dataset.n_px_of_unnormalized_image == 224:
        preprocess_for_resize = to_tensor
    if color_jitter_sample_num > 0:
        assert dataset.image_file_dir is not None
        # TODO: implement for image_file_dir is None
    if dataset.image_file_dir is not None:
        print("load image tensors from image files ...")
        assert len(dataset.texts_for_font_image) == 1
        for font_path in tqdm(dataset.font_paths):
            font_name = os.path.splitext(os.path.basename(font_path))[0]
            image_file_path = os.path.join(dataset.image_file_dir, font_name + ".png")
            image = Image.open(image_file_path)
            image = my_convert_to_rgb(image)
            if dataset.store_unnormalized_image:
                unnormalized_images = [
                    preprocess_for_aug(image) for _ in range(sample_num)
                ]
                images = [
                    preprocess_for_normalize(unnormalized_image)
                    for unnormalized_image in unnormalized_images
                ]
                unnormalized_images = [
                    preprocess_for_resize(to_pil_image(unnormalized_image).convert("L"))
                    for unnormalized_image in unnormalized_images
                ]
                font_text_to_unnormalized_image_tensors.append(
                    torch.stack(unnormalized_images).to(model.dtype)
                )
            else:
                images = [preprocess(image).to(model.dtype) for _ in range(sample_num)]
            if color_jitter_sample_num > 0:
                image_tensor = pil_to_tensor(image)
                # change black to red and white to yellow, which helps to use color jitter. (black and white are not changed by color jitter)
                image_tensor[0, :, :] = 255
                # image_tensor[2, :, :] = 0
                image = to_pil_image(image_tensor)
                color_jitter_images = [
                    color_jitter_preprocess(image).to(model.dtype)
                    for _ in range(color_jitter_sample_num)
                ]
                images.extend(color_jitter_images)
            font_text_to_image_tensors.append(torch.stack(images))
    else:
        print("create image tensors from font files ...")
        for text in dataset.texts_for_font_image:
            for font in tqdm(dataset.fonts):
                # images = []
                # for _ in range(sample_num):
                #     image = dataset.create_image(text, font)
                #     images.append(image)
                unpreprocessed_image = dataset.create_image(
                    text, font, no_preprocess=True, padding=padding
                )
                if dataset.store_unnormalized_image:
                    unnormalized_images = [
                        preprocess_for_aug(unpreprocessed_image)
                        for _ in range(sample_num)
                    ]
                    images = [
                        preprocess_for_normalize(unnormalized_image)
                        for unnormalized_image in unnormalized_images
                    ]
                    unnormalized_images = [
                        preprocess_for_resize(
                            to_pil_image(unnormalized_image).convert("L")
                        )
                        for unnormalized_image in unnormalized_images
                    ]
                    font_text_to_unnormalized_image_tensors.append(
                        torch.stack(unnormalized_images).to(model.dtype)
                    )
                else:
                    images = [
                        preprocess(unpreprocessed_image).to(model.dtype)
                        for _ in range(sample_num)
                    ]
                font_text_to_image_tensors.append(torch.stack(images).to(model.dtype))
    if dataset.store_unnormalized_image:
        dataset.set_font_text_to_unnormalized_image_tensors(
            font_text_to_unnormalized_image_tensors
        )
    dataset.set_font_text_to_image_tensors(font_text_to_image_tensors)
