from dataclasses import dataclass
import string
from typing import Optional

from models.init_model import (
    device,
    preprocess,
    load,
    load_model,
    preprocess_for_single_character,
)
from utils.initialize_font_data import (
    fox_text,
    font_dir,
    gray_scale_image_file_dir,
    train_json_path,
    all_json,
    one_leave_out_json_path,
    all_gwfonts_json_path,
    train_all_gwfonts_json_path,
    val_all_gwfonts_json_path,
    unlabeled_predicted_attributes_json_path,
)
from utils.transform_image import (
    char_size,
    my_transform,
)
from models.oft import OFTConfig
from models.lora import LoRAConfig


@dataclass
class Config:
    checkpoint_path: Optional[str] = None
    use_unpretrained_model: bool = False
    font_dir: str = font_dir
    json_path: str = train_json_path
    vision_text: str = fox_text
    texts_for_font_image: list = None
    val_texts_for_font_image: list = None
    use_unlabeled_data: bool = False
    unlabeled_sampling_ratio: float = 0.1
    unlabeled_sample_num: int = 250
    unlabeled_random_prompts_num: int = 1000
    EPOCH: int = 1000
    BATCH_SIZE: int = 48
    attribute_threshold: int = 50
    attribute_under_threshold: int = 50
    lr: float = 2e-5
    oft_lr: float = 1e-4
    lora_lr: float = 1e-4
    coop_lr: float = 1e-3
    lr_schedular_end_factor: float = 0.1
    use_negative_loss: bool = False
    negative_loss_weight: float = 1e-3
    use_bce_loss: bool = False
    use_contrastive_image_loss: bool = False
    contrastive_image_loss_weight: float = 0.1
    use_triplet_image_loss: bool = False
    triplet_image_loss_weight: float = 1.0
    triplet_image_loss_margin: float = 0.05
    use_negative: bool = True
    use_weight: bool = True
    use_score: bool = False
    use_multiple_attributes: bool = True
    use_random_attributes: bool = True
    use_single_character: bool = False
    random_prompts_num: int = 1000
    max_sample_num: int = 3
    sample_num_each_epoch: int = 20
    use_oft_vision: bool = False
    use_oft_text: bool = False
    oft_config_vision: OFTConfig = None
    oft_config_text: OFTConfig = None
    use_lora_text: bool = False
    use_lora_vision: bool = False
    lora_config_vision: LoRAConfig = None
    lora_config_text: LoRAConfig = None
    use_coop_text: bool = False
    use_coop_vision: bool = False
    do_coop_text_optimize: bool = False
    do_coop_vision_optimize: bool = False
    precontext_length_text: int = 16
    precontext_length_vision: int = 10
    precontext_dropout_rate: float = 0.1
    pt_applied_layers: list = None
    use_chopped_clip: bool = False
    chopped_clip_vision_layers: int = 3
    chopped_clip_text_layers: int = 3
    use_clip_dual_adapter: bool = False

    # train only visual encoder with paired images
    train_only_visual_encoder: bool = False
    use_same_text_for_each_pair: bool = True

    use_clip_like_format: bool = False
    char_size: int = char_size
    test_char_size: int = 150

    # for managing model num to be trained
    start_index_for_train_model: int = 0
    trained_model_num: int = 1

    # use aug or not
    use_aug: bool = True
    use_color_jitter: bool = False
    color_jitter_sample_num: int = 10
    geta: float = 1.0
    train_dump_image: bool = False
    tmp_dump_image: bool = True
    rich_prompt: bool = False
    sample_num: int = 250
    single_character: bool = False
    task_for_validation: bool = False
    do_optimize: bool = True
    do_cross_validation: bool = False
    do_profile: bool = False
    image_file_dir: str = gray_scale_image_file_dir
    image_file_dir_for_validation: str = gray_scale_image_file_dir
    lower_bound_of_scale: float = 0.01
    leave_out_attributes = None
    one_leave_out: bool = False
    model_name: str = "ViT-B/32"

    use_fast_evaluator: bool = False

    def __post_init__(self):
        # TODO: set available model name
        # assert self.model_name in clip.available_models()
        self.model = load_model(
            checkpoint_path=self.checkpoint_path,
            device=device,
            model_name=self.model_name,
            use_oft_vision=self.use_oft_vision,
            use_oft_text=self.use_oft_text,
            oft_config_vision=self.oft_config_vision,
            oft_config_text=self.oft_config_text,
            use_lora_vision=self.use_lora_vision,
            use_lora_text=self.use_lora_text,
            lora_config_vision=self.lora_config_vision,
            lora_config_text=self.lora_config_text,
            use_coop_vision=self.use_coop_vision,
            use_coop_text=self.use_coop_text,
            precontext_length_text=self.precontext_length_text,
            precontext_length_vision=self.precontext_length_vision,
            precontext_dropout_rate=self.precontext_dropout_rate,
            pt_applied_layers=self.pt_applied_layers,
        )
        if self.use_unpretrained_model:
            print("Warning: use unpretrained model")
            self.model.initialize_parameters()
        self.context_length = self.model.context_length
        if hasattr(self.model, "precontext_length_text"):
            self.context_length -= self.model.precontext_length_text

        if self.one_leave_out:
            assert not self.leave_out_attributes
            self.one_leave_out_json_path = one_leave_out_json_path
            self.trained_model_num = len(all_json)
        elif self.leave_out_attributes:
            self.trained_model_num = len(self.leave_out_attributes)
        elif self.do_cross_validation:
            self.trained_model_num = 5

        self.val_texts_for_font_image = self.texts_for_font_image
        self.unlabeled_json_path = None
        if self.use_unlabeled_data:
            self.unlabeled_json_path = unlabeled_predicted_attributes_json_path
        if self.use_single_character:
            self.image_file_dir = None
            self.train_dump_image = True
            self.char_size = 250
            self.test_char_size = 250
            self.tmp_preprocess = preprocess_for_single_character
            self.lower_bound_of_scale = 0.85
            # all alphabet
            self.texts_for_font_image = [
                # c for c in (string.ascii_uppercase + string.ascii_lowercase)
                # c for c in (string.ascii_uppercase)
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
                "K",
                "L",
                "M",
                "N",
                "O",
                "P",
                "Q",
                "R",
                "S",
                "T",
            ]
            self.val_texts_for_font_image = [
                "U",
                "V",
                "W",
                "X",
                "Y",
                "Z",
            ]
            # self.texts_for_font_image.extend([fox_text for _ in range(20)])
        elif self.texts_for_font_image is None:
            self.texts_for_font_image = [self.vision_text]
            self.val_texts_for_font_image = [self.vision_text]

        self.init_lr = self.lr
        self.default_lr = self.lr
        if self.use_aug:
            if self.use_single_character:
                # use preprocess_for_single_character
                self.tmp_preprocess = preprocess_for_single_character
            else:
                # self.tmp_preprocess = my_preprocess
                print("preprocess: lower_bound_of_scale: ", self.lower_bound_of_scale)
                self.tmp_preprocess = my_transform(
                    self.model.visual.input_resolution, self.lower_bound_of_scale
                )
        else:
            self.sample_num = 1
            self.tmp_preprocess = preprocess
        self.target_layers_text = []
        self.target_layers_vision = []
        if self.model_name == "ViT-B/32":
            if (
                self.use_oft_vision
                or self.use_oft_text
                or self.use_lora_vision
                or self.use_lora_text
                or self.use_clip_dual_adapter
                or self.use_coop_text
                or self.use_coop_vision
            ):
                self.target_layers_text = []
                self.target_layers_vision = []
                self.target_layers_vision = [
                    "transformer.resblocks.9",
                    "transformer.resblocks.10",
                    "transformer.resblocks.11",
                ]
                if self.pt_applied_layers is None:
                    self.pt_applied_layers = [
                        0,
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        10,
                        11,
                    ]
                    self.pt_applied_layers = [0, 1, 2]
            elif self.use_chopped_clip:
                self.target_layers_text = [
                    "resblocks.9",
                    "resblocks.10",
                    "resblocks.11",
                ]
                self.target_layers_vision = [
                    "transformer.resblocks.9",
                    "transformer.resblocks.10",
                    "transformer.resblocks.11",
                ]
                if self.chopped_clip_text_layers is not None:
                    self.target_layers_text = [
                        f"resblocks.{i}" for i in range(self.chopped_clip_text_layers)
                    ]
                if self.chopped_clip_vision_layers is not None:
                    self.target_layers_vision = [
                        f"transformer.resblocks.{i}"
                        for i in range(self.chopped_clip_vision_layers)
                    ]
            else:
                if not self.train_only_visual_encoder:
                    self.target_layers_text = [
                        "resblocks.9",
                        "resblocks.10",
                        "resblocks.11",
                    ]
                self.target_layers_vision = [
                    "transformer.resblocks.9",
                    "transformer.resblocks.10",
                    "transformer.resblocks.11",
                ]
        elif self.model_name == "ViT-L/14":
            if (
                self.use_clip_dual_adapter
                or self.use_coop_text
                or self.use_coop_vision
            ):
                self.target_layers_text = []
                self.target_layers_vision = []
            else:
                self.target_layers_text = [
                    "resblocks.9",
                    "resblocks.10",
                    "resblocks.11",
                ]
                self.target_layers_vision = [
                    "transformer.resblocks.17",
                    "transformer.resblocks.18",
                    "transformer.resblocks.19",
                    "transformer.resblocks.20",
                    "transformer.resblocks.21",
                    "transformer.resblocks.22",
                    "transformer.resblocks.23",
                ]

        if self.train_only_visual_encoder:
            self.json_path = train_all_gwfonts_json_path
            self.set_signature()
        elif not self.do_cross_validation:
            self.set_signature()
        if not self.do_optimize and self.checkpoint_path is None:
            # self.checkpoint_path = f"model_checkpoints/{self.signature}.pt"
            pass

    def set_signature(self, cross_validation_index=None):
        self.signature = ""
        if self.do_cross_validation:
            assert cross_validation_index is not None
            self.signature += f"cv_{self.trained_model_num}_{cross_validation_index}_"
        if self.task_for_validation:
            self.signature += "task_for_validation_"
        self.signature += f"{str(self.model_name).replace('/', '_')}_"
        if self.use_unlabeled_data:
            self.signature += f"use_unlabeled_data_{self.unlabeled_sample_num}_{self.unlabeled_random_prompts_num}_{self.unlabeled_sampling_ratio}_"
        if self.train_only_visual_encoder:
            self.signature += "train_only_visual_encoder_"
            if self.use_same_text_for_each_pair:
                self.signature += "same_text_"
        if not self.use_multiple_attributes:
            self.signature += "single_attribute_"
        if self.use_bce_loss:
            self.signature += "bce_"
        if self.use_chopped_clip:
            self.signature += f"chopped_clip_vision_layers{self.chopped_clip_vision_layers}_text_layers{self.chopped_clip_text_layers}_"
        if self.use_oft_vision or self.use_oft_text:
            self.signature += f"oft{self.oft_lr}_"
            if self.use_oft_vision:
                self.signature += f"v{self.oft_config_vision.r}-"
                if self.oft_config_vision.apply_q:
                    self.signature += "q"
                if self.oft_config_vision.apply_k:
                    self.signature += "k"
                if self.oft_config_vision.apply_v:
                    self.signature += "v"
                if self.oft_config_vision.apply_out:
                    self.signature += "o"
                self.signature += "_"

            if self.use_oft_text:
                self.signature += f"t{self.oft_config_text.r}-"
                if self.oft_config_text.apply_q:
                    self.signature += "q"
                if self.oft_config_text.apply_k:
                    self.signature += "k"
                if self.oft_config_text.apply_v:
                    self.signature += "v"
                if self.oft_config_text.apply_out:
                    self.signature += "o"
                self.signature += "_"
        if self.use_lora_text or self.use_lora_vision:
            self.signature += f"lora_"
            if self.use_lora_vision:
                self.signature += "v-"
                if self.lora_config_vision.apply_q:
                    self.signature += "q"
                if self.lora_config_vision.apply_k:
                    self.signature += "k"
                if self.lora_config_vision.apply_v:
                    self.signature += "v"
                if self.lora_config_vision.apply_out:
                    self.signature += "o"
                self.signature += "_"
                self.signature += f"{self.lora_config_vision.r}-{self.lora_config_vision.alpha}_"
                if self.lora_config_vision.bias:
                    self.signature += f"b_"
            if self.use_lora_text:
                self.signature += "t-"
                if self.lora_config_text.apply_q:
                    self.signature += "q"
                if self.lora_config_text.apply_k:
                    self.signature += "k"
                if self.lora_config_text.apply_v:
                    self.signature += "v"
                if self.lora_config_text.apply_out:
                    self.signature += "o"
                self.signature += "_"
                self.signature += f"{self.lora_config_text.r}-{self.lora_config_text.alpha}_"
                if self.lora_config_text.bias:
                    self.signature += f"b_"
            if self.lora_config_vision.learnable_alpha or self.lora_config_text.learnable_alpha:
                self.signature += "lalpha_"
        if self.use_coop_text:
            self.signature += (
                f"coop_precontext_length{self.precontext_length_text}_lr{self.coop_lr}_"
            )
        if self.use_coop_vision:
            self.signature += f"coop_vision_length{self.precontext_length_vision}_layers{''.join(map(str, self.pt_applied_layers))}_dr{self.precontext_dropout_rate}_lr{self.coop_lr}_"
        if self.use_unpretrained_model:
            self.signature += "unpretrained_"
        for i in range(len(self.target_layers_text)):
            self.signature += f"{11 - (len(self.target_layers_text) - 1 - i)}"
        for i in range(len(self.target_layers_vision)):
            if self.model_name == "ViT-B/32":
                self.signature += f"{11 - (len(self.target_layers_vision) - 1 - i)}"
            elif self.model_name == "ViT-L/14":
                self.signature += f"{23 - (len(self.target_layers_vision) - 1 - i)}"
        self.signature += f"_batch{self.BATCH_SIZE}"
        if self.use_aug:
            self.signature += f"_aug{self.sample_num}"
            if self.use_color_jitter:
                self.signature += f"_cj{self.color_jitter_sample_num}"
            self.signature += f"_lbound_of_scale{self.lower_bound_of_scale}"
        if not self.train_only_visual_encoder:
            self.signature += f"_max_attr_num_{self.max_sample_num}"
            self.signature += f"_random_p_num_{self.random_prompts_num}"
            self.signature += f"_geta{self.geta}"
        if self.use_single_character:
            self.signature += "_single_char"
        if self.use_negative:
            self.signature += "_use_negative"
        if self.use_clip_like_format:
            self.signature += "_clip_like_format"
        if self.use_negative_loss:
            self.signature += f"_use_negative_loss{self.negative_loss_weight}"
        if self.use_contrastive_image_loss:
            self.signature += f"_cil{self.contrastive_image_loss_weight}"
        if self.use_triplet_image_loss:
            self.signature += f"_til{self.triplet_image_loss_weight}"
        self.signature += f"_lr{self.lr}-{self.lr_schedular_end_factor}"
        if self.image_file_dir:
            self.signature += f"_image_file_dir"
