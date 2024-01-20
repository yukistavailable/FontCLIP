from dataclasses import dataclass
from typing import Optional, List

from models.init_model import (
    device,
    preprocess,
    load_model,
)
from utils.initialize_font_data import (
    fox_text,
    font_dir,
    gray_scale_image_file_dir,
    train_json_path,
)
from utils.transform_image import (
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
    texts_for_font_image: List[str] = None
    val_texts_for_font_image: List[str] = None
    EPOCH: int = 1000
    BATCH_SIZE: int = 48
    attribute_threshold: int = 50
    attribute_under_threshold: int = 50
    lr: float = 2e-5
    coop_lr: float = 1e-3
    lr_schedular_end_factor: float = 0.1
    use_bce_loss: bool = True
    use_negative: bool = True
    use_multiple_attributes: bool = True
    use_random_attributes: bool = True
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
    pt_applied_layers: List[str] = None

    char_size: int = 250
    test_char_size: int = 150

    # use aug or not
    use_aug: bool = True
    use_color_jitter: bool = False
    color_jitter_sample_num: int = 10
    geta: float = 1.0
    train_dump_image: bool = False
    tmp_dump_image: bool = True
    rich_prompt: bool = False
    sample_num: int = 250
    do_optimize: bool = True
    do_profile: bool = False
    image_file_dir: str = gray_scale_image_file_dir
    image_file_dir_for_validation: str = gray_scale_image_file_dir
    lower_bound_of_scale: float = 0.01
    model_name: str = "ViT-B/32"


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

        self.val_texts_for_font_image = self.texts_for_font_image
        if self.texts_for_font_image is None:
            self.texts_for_font_image = [self.vision_text]
            self.val_texts_for_font_image = [self.vision_text]

        self.init_lr = self.lr
        self.default_lr = self.lr
        if self.use_aug:
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
                or self.use_coop_text
                or self.use_coop_vision
            ):
                self.target_layers_text = []
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
            else:
                self.target_layers_text = [
                    "resblocks.9",
                    "resblocks.10",
                    "resblocks.11",
                ]

        self.set_signature()

    def set_signature(self):
        self.signature = ""
        self.signature += f"{str(self.model_name).replace('/', '_')}_"
        if not self.use_multiple_attributes:
            self.signature += "single_attribute_"
        if self.use_bce_loss:
            self.signature += "bce_"
        if self.use_oft_vision or self.use_oft_text:
            self.signature += "oft_"
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
            self.signature += "lora_"
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
                    self.signature += "b_"
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
                    self.signature += "b_"
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
        self.signature += f"_max_attr_num_{self.max_sample_num}"
        self.signature += f"_random_p_num_{self.random_prompts_num}"
        self.signature += f"_geta{self.geta}"
        if self.use_negative:
            self.signature += "_use_negative"
        self.signature += f"_lr{self.lr}-{self.lr_schedular_end_factor}"
        if self.image_file_dir:
            self.signature += "_image_file_dir"
