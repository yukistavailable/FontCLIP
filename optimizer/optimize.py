import os
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    Normalize,
)
from tqdm import tqdm
from typing import Optional
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms.functional import to_pil_image
from dataclasses import dataclass, field
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from svgwrite import Drawing
from svgpathtools import svg2paths, Path

import yaml
from easydict import EasyDict as edict
import os.path as osp
import random
import numpy.random as npr
import numpy as np
import wandb
import pydiffvg
from utils.tokenizer import tokenize
from utils.transform_image import draw_text_with_new_lines
from models.init_model import load_model, device, preprocess as clip_preprocess
from models.lora import LoRAConfig


from optimizer.util_tools import (
    edict_2_dict,
    check_and_create_dir,
    update,
    preprocess,
    get_data_augs,
    init_shapes,
    learning_rate_decay,
    save_image,
    combine_word,
    create_video,
)
from optimizer.losses import (
    ToneLoss,
    ConformalLoss,
    FCLIPLoss,
    LaplacianLoss,
    CosLoss,
    sample_contour_from_bezier_curves,
    L2Loss,
    LaplacianLossBetweenEdge,
    XingLoss,
    DirectionLoss,
)
from optimizer import save_svg

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

gamma = 1.0
CURRENT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR_PATH = os.path.dirname(CURRENT_DIR_PATH)
OUTPUT_PATH = os.path.join(PARENT_DIR_PATH, "output")
SVGS_PATH = os.path.join(PARENT_DIR_PATH, "svgs")

# set random seed
random_seed = 123
random_seed = None
if random_seed is not None:
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

model_name = "ViT-B/32"


def draw_single_character(
    char, font, img_width, img_height, x_start=None, y_start=None
):
    char_size = font.size
    if x_start is None:
        x_start = (img_width - char_size) / 2
        if char.isascii():
            x_start += 20
    if y_start is None:
        y_start = (img_height - char_size) / 2 - 30
    image = Image.new("RGB", (img_width, img_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    print(x_start, y_start)
    draw.text((x_start, y_start), char, font=font, fill=(0, 0, 0))
    return image


def my_transform(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


my_preprocess = my_transform(224)


def change_size_transform(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
        ]
    )


# def show_outline(
#     svg_path,
#     show_control_points=True,
#     figsize=(10, 10),
#     num_samples_per_stroke=100,
#     save_image=False,
#     save_path=None,
# ):
#     paths, attributes = svg2paths(svg_path)
#     fig, ax = plt.subplots(figsize=figsize)
#     num_samples = num_samples_per_stroke

#     for path in paths:
#         for segment in path:
#             if segment.__class__.__name__ == "CubicBezier":
#                 if show_control_points:
#                     ax.plot(
#                         [
#                             segment.start.real,
#                             segment.control1.real,
#                             segment.control2.real,
#                             segment.end.real,
#                         ],
#                         [
#                             segment.start.imag,
#                             segment.control1.imag,
#                             segment.control2.imag,
#                             segment.end.imag,
#                         ],
#                         "ro-",
#                     )
#                 # sample points on real outline
#                 ax.plot(
#                     [
#                         Path(segment).point(i / num_samples).real
#                         for i in range(num_samples)
#                     ],
#                     [
#                         Path(segment).point(i / num_samples).imag
#                         for i in range(num_samples)
#                     ],
#                     "b-",
#                 )

#             elif segment.__class__.__name__ == "QuadraticBezier":
#                 if show_control_points:
#                     ax.plot(
#                         [segment.start.real, segment.control.real, segment.end.real],
#                         [segment.start.imag, segment.control.imag, segment.end.imag],
#                         "ro-",
#                     )

#                 # sample points on real outline
#                 ax.plot(
#                     [
#                         Path(segment).point(i / num_samples).real
#                         for i in range(num_samples)
#                     ],
#                     [
#                         Path(segment).point(i / num_samples).imag
#                         for i in range(num_samples)
#                     ],
#                     "b-",
#                 )
#             elif segment.__class__.__name__ == "Line":
#                 if show_control_points:
#                     ax.plot(
#                         [segment.start.real, segment.end.real],
#                         [segment.start.imag, segment.end.imag],
#                         "ro-",
#                     )
#                 # sample points on real outline
#                 ax.plot(
#                     [
#                         Path(segment).point(i / num_samples).real
#                         for i in range(num_samples)
#                     ],
#                     [
#                         Path(segment).point(i / num_samples).imag
#                         for i in range(num_samples)
#                     ],
#                     "b-",
#                 )
#     ax.axes.set_aspect("equal")
#     if save_image:
#         if save_path:
#             plt.savefig(save_path)
#         else:
#             base_dir = os.path.dirname(svg_path)
#             save_path = os.path.join(base_dir, "outline.png")
#             plt.savefig(os.path.join(save_path))
#             print(f"save_path: {save_path}")
#     plt.show()


def create_image(text, font_path, char_size=150):
    font = ImageFont.truetype(font_path, char_size)
    line_num = text.count("\n") + 1
    # print(line_num)
    width = int(char_size * len(text) * 1.8 / line_num)
    height = int(char_size * 1.5) * line_num

    image = draw_text_with_new_lines(text, font, width, height)
    return image


USE_WANDB = 0
WANDB_USER = ""
EXPERIMENT = "conformal_1.0_dist_pixel_100_kernel201"
YAML_PATH = os.path.join(CURRENT_DIR_PATH, "base.yaml")


@dataclass
class Config:
    font_path: str
    word: str
    semantic_concept: str
    optimized_letter: str
    render_size: int = 300
    cut_size: int = 300
    num_iter: int = 500
    config: str = YAML_PATH
    experiment: str = EXPERIMENT
    use_wandb: int = USE_WANDB
    wandb_user: str = WANDB_USER
    seed: int = 0
    log_dir: str = OUTPUT_PATH
    # prompt_suffix: str = 'minimal flat 2d vector. lineal color. trending on artstation'
    prompt_suffix: str = ""
    batch_size: int = 1
    char_size: int = 150
    size: int = 300
    do_preprocess: bool = True
    lr: float = 1.0
    lr_init: float = 0.002
    lr_final: float = 0.0008
    lr_delay_mult: float = 0.1
    lr_delay_steps: int = 100
    use_aug: bool = False
    use_single_character_image: bool = False
    create_init_svg: bool = True
    fclip_loss: bool = True
    use_fclip_direction_loss: bool = False
    use_fclip_direction_loss_vision: bool = False
    use_fclip_direction_loss_only: bool = False
    ref_semantic_concept: str = None
    ref_image_file_path: str = None
    fclip_loss_w: float = 5
    fclip_direction_loss_w: float = 1
    tone_loss_w: float = 1.0
    conformal_loss_w: float = 1.0
    laplacian_loss_w: float = 1.0
    laplacian_between_beziers_loss_w: float = 1.0
    cos_loss_w: float = 1.0
    G1_loss_w: float = 1.0
    L2_loss_w: float = 1.0
    Xing_loss_w: float = 1.0
    direction_loss_w: float = 1.0
    use_visual_encoder: bool = False
    use_tone_loss: bool = True
    use_tone_loss_schedular: bool = False
    use_conformal_loss: bool = True
    use_laplacian_loss: bool = True
    use_laplacian_between_beziers_loss: bool = False
    use_cos_loss: bool = True
    use_G1_loss: bool = True
    use_L2_loss: bool = True
    use_Xing_loss: bool = True
    use_direction_loss: bool = True
    multiple_attributes: bool = False
    multiple_text_encoders: bool = False
    target_attributes: list = field(default_factory=list)
    target_attributes_weights: list = field(default_factory=list)
    reduce_cp: bool = False
    epsilon: float = 0
    checkpoint_path: Optional[str] = None
    image_file_path: str = ""
    visual_optimize: bool = False
    use_lr_scheduler: bool = False
    is_counter: bool = False
    num_per_curve: int = 10
    skip_control_points: bool = False
    skip_corners: bool = False
    skip_corner_threshold: float = 0.8
    laplacian_between_beziers_loss_threshold: float = -0.8
    skip_edge_laplacian: bool = False
    only_edge_laplacian: bool = False
    skip_edge_cos: bool = False
    target_img: torch.Tensor = None

    def __post_init__(self):
        if " " in self.word:
            raise ValueError("word should not contain space")

        if len(self.optimized_letter) != 1:
            raise ValueError("optimized_letter should be a single character")

        if self.create_init_svg:
            self.set_target()

        self.log_dir = f"{self.log_dir}/{self.experiment}_{self.word}"
        self.letter = self.optimized_letter
        self.font = os.path.splitext(os.path.basename(self.font_path))[0]
        self.render_size = self.size
        self.cut_size = self.size

        assert self.multiple_attributes is False or self.multiple_text_encoders is False
        if self.multiple_attributes or self.multiple_text_encoders:
            assert len(self.target_attributes) > 0
            assert len(self.target_attributes) == len(self.target_attributes_weights)

        if self.use_fclip_direction_loss:
            assert not self.use_fclip_direction_loss_vision
            assert self.fclip_loss
            assert self.ref_semantic_concept is not None

        if self.use_fclip_direction_loss_vision:
            assert not self.use_fclip_direction_loss
            assert self.fclip_loss
            assert self.image_file_path is not None
            assert self.ref_image_file_path is not None

    def set_target(self):
        self.target = self.create_svg_from_font(
            self.font_path,
            self.optimized_letter,
            char_size=self.char_size,
            size=self.size,
        )

    def set_target_img(self, target_img: torch.Tensor):
        self.target_img = target_img

    @staticmethod
    def create_svg_from_font(
        font_path, character, output_path=None, char_size=200, size=300
    ):
        if output_path is None:
            font_name = os.path.splitext(os.path.basename(font_path))[0]
            output_path = os.path.join(SVGS_PATH, f"init/{font_name}_{character}_{size}.svg")

        # get unicode of character
        unicode = ord(character)

        font = TTFont(font_path)
        glyph_set = font.getGlyphSet()

        # get cmap
        cmap = font["cmap"].getBestCmap()
        glyph_id = cmap[unicode]

        glyph = glyph_set[glyph_id]
        scale = char_size / font["head"].unitsPerEm

        dwg = Drawing(output_path, profile="tiny", size=(size, size))
        pen = SVGPathPen(glyph_set)
        glyph.draw(pen)

        # transform and scale
        # g = dwg.g(transform=f'scale({scale}) translate(0, { char_size })')
        # x_offset = (size - char_size) / 2 + 20
        # y_offset = (size - char_size) / 2 - 30

        x_offset = (size - char_size) / 2
        y_offset = (size - char_size) / 2 - 30
        # TODO: fix offset
        if character.isascii():
            print("ascii character")
            x_offset += 20
            if character == "W" or character == "M":
                x_offset -= 10
            if character == "y":
                y_offset += 30

        g = dwg.g(
            transform=f"translate({x_offset}, {char_size - y_offset}) scale({scale}, -{scale})"
        )
        # g = dwg.g(transform=f'translate({0}, {0}) scale({scale}, -{scale})')
        path = dwg.path(pen.getCommands(), fill="black")
        g.add(path)
        dwg.add(g)
        dwg.save()
        return output_path


def draw_image_through_svg_from_font_path(
    font_path, char, svg_size=200, char_size=150, image_size=200, device=device
):
    output_path = "tmp.svg"
    Config.create_svg_from_font(
        font_path, char, output_path=output_path, char_size=char_size, size=svg_size
    )
    # shapes, shape_groups, parameters = init_shapes(svg_path=output_path, trainable=False)
    _, _, shapes, shape_groups = pydiffvg.svg_to_scene(output_path)
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        image_size, image_size, shapes, shape_groups
    )
    render = pydiffvg.RenderFunction.apply
    img = render(image_size, image_size, 2, 2, 0, None, *scene_args)
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
        img.shape[0], img.shape[1], 3, device=device
    ) * (1 - img[:, :, 3:4])
    img = img[:, :, :3]
    img = img.permute(2, 0, 1)  # HWC -> NCHW

    # delete tmp file
    os.remove(output_path)
    return img


def get_signature(
    cfg: Config,
    font_path: str,
    visual_font_path: str = None,
    target_image_path: str = None,
) -> str:
    font_name = os.path.splitext(os.path.basename(font_path))[0]
    # set experiment dir
    signature = f"{cfg.letter}_{font_name}"
    if cfg.use_aug:
        signature += "_aug"
    if cfg.use_single_character_image:
        signature += "_single_character"
    if cfg.visual_optimize:
        if target_image_path is not None:
            signature += (
                f"_visual_{os.path.splitext(os.path.basename(target_image_path))[0]}"
            )
        elif visual_font_path:
            visual_font_name = os.path.splitext(os.path.basename(visual_font_path))[0]
            signature += f"_visual_{visual_font_name}"
        else:
            tmp_image_file_path = os.path.splitext(
                os.path.basename(cfg.image_file_path)
            )[0]
            signature += f"_visual_{tmp_image_file_path}"
    elif cfg.multiple_text_encoders:
        tmp_attribute_signature = [
            str(a) + str(w)
            for a, w in zip(cfg.target_attributes, cfg.target_attributes_weights)
        ]
        # signature = f"{cfg.letter}_multiple_attribute_preserve_init_{target_attribute}{target_attributes_weights[0]}_all{target_attributes_weights[1]}"
        signature += (
            f"_multiple_attribute_preserve_init_{'-'.join(tmp_attribute_signature)}"
        )
    elif cfg.multiple_attributes:
        tmp_attribute_signature = [
            str(a) + str(w)
            for a, w in zip(cfg.target_attributes, cfg.target_attributes_weights)
        ]
        signature += f"_multiple_attribute_{'-'.join(tmp_attribute_signature)}"
    else:
        signature += f"_{cfg.semantic_concept}"
    if not cfg.do_preprocess:
        signature += "_no_preprocess"
    # if 'use_negative_loss' in cfg.checkpoint_path:
    #     signature += '_use_negative_loss'
    if cfg.fclip_loss:
        signature += f"_fclip{cfg.fclip_loss_w}"
        if cfg.checkpoint_path is None:
            signature += "_no_checkpoint"
        elif "ViT-L" in cfg.checkpoint_path:
            signature += "_ViT-L"
        if cfg.use_fclip_direction_loss:
            signature += f"_fclip_direction{cfg.fclip_direction_loss_w}"
        if cfg.use_fclip_direction_loss_vision:
            signature += f"_fclip_direction_vision{cfg.fclip_direction_loss_w}"
        if cfg.use_fclip_direction_loss_only:
            signature += f"only"
    if cfg.use_L2_loss:
        signature += f"_L2{cfg.L2_loss_w}"
    if cfg.use_tone_loss:
        signature += f"_tone{cfg.tone_loss_w}-{cfg.loss.tone.pixel_dist_sigma}-{cfg.loss.tone.pixel_dist_kernel_blur}"
        if cfg.use_tone_loss_schedular:
            signature += f"_schedular"
    if cfg.use_conformal_loss:
        signature += f"_conf{cfg.conformal_loss_w}"
    if cfg.use_laplacian_loss:
        signature += f"_lap"
        if cfg.skip_edge_laplacian:
            signature += f"se"
        if cfg.only_edge_laplacian:
            signature += f"oe"
        signature += f"{cfg.laplacian_loss_w}"
    if cfg.use_laplacian_between_beziers_loss:
        signature += f"_lap_bez{cfg.laplacian_between_beziers_loss_w}-{cfg.laplacian_between_beziers_loss_threshold}"
    if cfg.use_cos_loss:
        signature += f"_cos{cfg.cos_loss_w}"
        signature += f"_num_per_curve{cfg.num_per_curve}"
    if cfg.use_G1_loss:
        assert cfg.skip_control_points
        signature += f"_G1{cfg.G1_loss_w}"
        if cfg.skip_corners:
            signature += f"_skipcorners_{cfg.skip_corner_threshold}"
    if cfg.use_Xing_loss:
        signature += f"_Xing{cfg.Xing_loss_w}"
    if cfg.use_direction_loss:
        signature += f"_direction{cfg.direction_loss_w}"

    signature += f"_cp{cfg.level_of_cc}"
    if cfg.reduce_cp:
        signature += f"_reducecp{cfg.epsilon}"
    if cfg.use_lr_scheduler:
        signature += f"_lr{cfg.lr}"
        signature += "use_lr_scheduler"
    else:
        # signature += f'_lr{cfg.lr_base.point * cfg.lr.lr_delay_mult}'
        signature += f"_lr{cfg.lr}"
    if cfg.is_counter and cfg.use_laplacian_loss:
        signature += "_counter"
        signature += f"_num_per_curve{cfg.num_per_curve}"
    if random_seed is not None:
        signature += f"_random_seed{random_seed}"

    cfg.experiment_dir = os.path.join(cfg.log_dir, cfg.font, signature)
    configfile = os.path.join(cfg.experiment_dir, "config.yaml")

    # create experiment dir and save config
    check_and_create_dir(configfile)
    with open(os.path.join(configfile), "w") as f:
        yaml.dump(edict_2_dict(cfg), f)

    if cfg.use_wandb:
        wandb.init(
            project="Word-As-Image",
            entity=cfg.wandb_user,
            config=cfg,
            name=f"{signature}",
            id=wandb.util.generate_id(),
        )

    if cfg.seed is not None:
        random.seed(cfg.seed)
        npr.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.backends.cudnn.benchmark = False
    else:
        assert False

    return signature


def train(cfg: Config, signature: str):
    pydiffvg.set_use_gpu(torch.cuda.is_available())

    print("preprocessing")
    if cfg.do_preprocess:
        preprocess(
            cfg.font,
            cfg.word,
            cfg.optimized_letter,
            cfg.level_of_cc,
            cfg.font_path,
            init_path=os.path.join(SVGS_PATH, "init"),
            reduce_cp=cfg.reduce_cp,
            epsilon=cfg.epsilon,
            svg_path=cfg.target,
        )

    h, w = cfg.render_size, cfg.render_size

    data_augs = get_data_augs(cfg.cut_size)

    render = pydiffvg.RenderFunction.apply

    # initialize shape
    print("initializing shape")
    shapes, shape_groups, parameters = init_shapes(
        svg_path=cfg.target, trainable=cfg.trainable
    )

    scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shape_groups)
    img_init = render(w, h, 2, 2, 0, None, *scene_args)
    img_init = img_init[:, :, 3:4] * img_init[:, :, :3] + torch.ones(
        img_init.shape[0], img_init.shape[1], 3, device=device
    ) * (1 - img_init[:, :, 3:4])
    img_init = img_init[:, :, :3]
    c_w_h_img_init = img_init.permute(2, 0, 1)  # HWC -> NCHW

    if cfg.use_wandb:
        plt.imshow(img_init.detach().cpu())
        wandb.log({"init": wandb.Image(plt)}, step=0)
        plt.close()

    if cfg.use_tone_loss:
        tone_loss = ToneLoss(cfg)
        tone_loss.set_image_init(img_init)

    if cfg.fclip_loss:
        print("use fclip loss")
        print(cfg.checkpoint_path)
        model_name = "ViT-B/32"
        use_lora_text = False
        lora_config_text = None
        if cfg.checkpoint_path is None:
            model_name = "ViT-B/32"
        else:
            if "ViT-L" in cfg.checkpoint_path:
                model_name = "ViT-L/14"
                print("use ViT-L/14")
            if "lora" in cfg.checkpoint_path:
                print("lora")
                use_lora_text = True
                lora_config_text = LoRAConfig(
                    r=256,
                    alpha=1024.0,
                    bias=False,
                    learnable_alpha=False,
                    apply_q=True,
                    apply_k=True,
                    apply_v=True,
                    apply_out=True,
                )

        model = load_model(
            checkpoint_path=cfg.checkpoint_path,
            model_name=model_name,
            use_lora_text=use_lora_text,
            lora_config_text=lora_config_text,
        )
        model = model.to(device)
        image_init = None
        if cfg.use_fclip_direction_loss or cfg.use_fclip_direction_loss_vision:
            image_init = c_w_h_img_init
            image_init = to_pil_image(image_init)
        fclip_loss = FCLIPLoss(
            cfg,
            clip_model=model,
            device=device,
            preprocess=my_preprocess,
            clip_preprocess=clip_preprocess,
            image_init=image_init,
        )

    if cfg.save.init:
        print("saving init")
        filename = os.path.join(cfg.experiment_dir, "svg-init", "init.svg")
        check_and_create_dir(filename)
        save_svg.save_svg(filename, w, h, shapes, shape_groups)

    num_iter = cfg.num_iter
    pg = [{"params": parameters["point"], "lr": cfg.lr}]
    optim = torch.optim.Adam(pg, betas=(0.9, 0.9), eps=1e-6)

    if cfg.loss.conformal.use_conformal_loss:
        conformal_loss = ConformalLoss(
            parameters, device, cfg.optimized_letter, shape_groups
        )

    if cfg.use_laplacian_loss:
        laplacian_loss = LaplacianLoss(
            parameters,
            device,
            shape_groups,
            is_contour=cfg.is_counter,
            num_per_curve=cfg.num_per_curve,
            skip_edge=cfg.skip_edge_laplacian,
            only_edge=cfg.only_edge_laplacian,
        )

    if cfg.use_laplacian_between_beziers_loss:
        laplacian_between_beziers_loss = LaplacianLossBetweenEdge(
            parameters=parameters,
            threshold=cfg.laplacian_between_beziers_loss_threshold,
        )

    if cfg.use_cos_loss:
        cos_loss = CosLoss(
            parameters,
            device,
            shape_groups,
            is_contour=True,
            skip_control_points=False,
            skip_corners=False,
            num_per_curve=cfg.num_per_curve,
            skip_edge=cfg.skip_edge_cos,
        )

    if cfg.use_G1_loss:
        G1_loss = CosLoss(
            parameters,
            device,
            shape_groups,
            is_contour=False,
            skip_control_points=cfg.skip_control_points,
            skip_corners=cfg.skip_corners,
            skip_corner_threshold=cfg.skip_corner_threshold,
            use_angle=False,
        )

    if cfg.use_L2_loss:
        L2_loss = L2Loss(cfg)

    if cfg.use_Xing_loss:
        Xing_loss = XingLoss(parameters).to(device)

    if cfg.use_direction_loss:
        direction_loss = DirectionLoss(parameters).to(device)

    lr_lambda = (
        lambda step: learning_rate_decay(
            step,
            cfg.lr_init,
            cfg.lr_final,
            num_iter,
            lr_delay_steps=cfg.lr_delay_steps,
            lr_delay_mult=cfg.lr_delay_mult,
        )
        / cfg.lr_init
    )

    scheduler = LambdaLR(
        optim, lr_lambda=lr_lambda, last_epoch=-1
    )  # lr.base * lrlambda_f
    print(w, h)

    print("start training")
    # training loop
    t_range = tqdm(range(num_iter))
    for step in t_range:
        if cfg.use_wandb:
            wandb.log({"learning_rate": optim.param_groups[0]["lr"]}, step=step)
        optim.zero_grad()

        # render image
        scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shape_groups)
        img = render(w, h, 2, 2, step, None, *scene_args)

        # compose image with white background
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
            img.shape[0], img.shape[1], 3, device=device
        ) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]

        if cfg.save.video and (
            step % cfg.save.video_frame_freq == 0 or step == num_iter - 1
        ):
            save_image(
                img,
                os.path.join(cfg.experiment_dir, "video-png", f"iter{step:04d}.png"),
                gamma,
            )
            filename = os.path.join(
                cfg.experiment_dir, "video-svg", f"iter{step:04d}.svg"
            )
            check_and_create_dir(filename)
            save_svg.save_svg(filename, w, h, shapes, shape_groups)
            if cfg.use_wandb:
                plt.imshow(img.detach().cpu())
                wandb.log({"img": wandb.Image(plt)}, step=step)
                plt.close()

        x = img.unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> NCHW
        x = x.repeat(cfg.batch_size, 1, 1, 1)
        if cfg.use_aug:
            x_aug = data_augs.forward(x)

        loss = 0
        if cfg.use_tone_loss:
            tone_loss_res = tone_loss(x, step)
            if cfg.use_wandb:
                wandb.log({"dist_loss": tone_loss_res}, step=step)
            if cfg.use_tone_loss_schedular:
                loss = loss + tone_loss_res * tone_loss.get_scheduler(step)
            else:
                loss = loss + tone_loss_res * tone_loss.get_scheduler(None)

        if cfg.use_conformal_loss:
            loss_angles = conformal_loss()
            if cfg.use_wandb:
                wandb.log({"loss_angles": loss_angles}, step=step)
            loss_angles = cfg.conformal_loss_w * loss_angles
            loss = loss + loss_angles
        if cfg.fclip_loss:
            if cfg.use_fclip_direction_loss or cfg.use_fclip_direction_loss_vision:
                tmp_w = 1
                if cfg.use_fclip_direction_loss_only:
                    tmp_w = 0
                loss_fclip, loss_fclip_direction = fclip_loss(
                    x_aug if cfg.use_aug else x, step=None
                )
                loss = (
                    loss
                    + loss_fclip * tmp_w
                    + loss_fclip_direction * cfg.fclip_direction_loss_w
                )
                if cfg.use_wandb and cfg.use_fclip_direction_loss:
                    wandb.log({"loss_fclip_direction": loss_fclip_direction}, step=step)
                if cfg.use_wandb and cfg.use_fclip_direction_loss_vision:
                    wandb.log(
                        {"loss_fclip_direction_vision": loss_fclip_direction}, step=step
                    )
            else:
                loss_fclip = fclip_loss(x, step=None)
                loss = loss + loss_fclip

            if cfg.use_wandb and not cfg.use_fclip_direction_loss_only:
                wandb.log({"loss_fclip": loss_fclip}, step=step)
        if cfg.use_laplacian_loss:
            loss_laplacian_coordinates = laplacian_loss()
            if cfg.use_wandb:
                wandb.log(
                    {"loss_laplacian_coordinates": loss_laplacian_coordinates},
                    step=step,
                )
            loss_laplacian_coordinates = (
                loss_laplacian_coordinates * cfg.laplacian_loss_w
            )
            loss = loss + loss_laplacian_coordinates
        if cfg.use_laplacian_between_beziers_loss:
            loss_laplacian_between_beziers = laplacian_between_beziers_loss()
            if cfg.use_wandb:
                wandb.log(
                    {"loss_laplacian_between_beziers": loss_laplacian_between_beziers},
                    step=step,
                )
            loss_laplacian_between_beziers = (
                loss_laplacian_between_beziers * cfg.laplacian_between_beziers_loss_w
            )
            loss = loss + loss_laplacian_between_beziers

        if cfg.use_cos_loss:
            loss_cos = cos_loss()
            if cfg.use_wandb:
                wandb.log({"loss_cos": loss_cos}, step=step)
            loss_cos = loss_cos * cfg.cos_loss_w
            loss = loss + loss_cos
        if cfg.use_G1_loss:
            loss_G1 = G1_loss()
            if cfg.use_wandb:
                wandb.log({"loss_G1": loss_G1}, step=step)
            loss_G1 = loss_G1 * cfg.G1_loss_w
            loss = loss + loss_G1
        if cfg.use_L2_loss:
            loss_L2 = L2_loss(x)
            if cfg.use_wandb:
                wandb.log({"loss_L2": loss_L2}, step=step)
            loss_L2 = loss_L2 * cfg.L2_loss_w
            loss = loss + loss_L2
        if cfg.use_Xing_loss:
            loss_Xing = Xing_loss()
            if cfg.use_wandb:
                wandb.log({"loss_Xing": loss_Xing}, step=step)
            loss_Xing = loss_Xing * cfg.Xing_loss_w
            loss = loss + loss_Xing.to(device)
        if cfg.use_direction_loss:
            loss_direction = direction_loss()
            if cfg.use_wandb:
                wandb.log({"loss_direction": loss_direction}, step=step)
            loss_direction = loss_direction * cfg.direction_loss_w
            loss = loss + loss_direction.to(device)

        # print(loss_laplacian_coordinates)

        t_range.set_postfix({"loss": loss.item()})
        loss.backward()
        optim.step()
        if cfg.use_lr_scheduler:
            scheduler.step()

    filename = os.path.join(cfg.experiment_dir, "output-svg", "output.svg")
    check_and_create_dir(filename)
    save_svg.save_svg(filename, w, h, shapes, shape_groups)

    word_svg_scaled = os.path.join(SVGS_PATH, f"init/{cfg.font}_{cfg.word}_scaled.svg")
    combine_word(
        cfg.word,
        cfg.optimized_letter,
        cfg.font,
        cfg.experiment_dir,
        word_svg_scaled=word_svg_scaled,
    )

    if cfg.save.image:
        filename = os.path.join(cfg.experiment_dir, "output-png", "output.png")
        check_and_create_dir(filename)
        imshow = img.detach().cpu()
        pydiffvg.imwrite(imshow, filename, gamma=gamma)
        if cfg.use_wandb:
            plt.imshow(img.detach().cpu())
            wandb.log({"img": wandb.Image(plt)}, step=step)
            plt.close()

    if cfg.save.video:
        print("saving video")
        create_video(cfg.num_iter, cfg.experiment_dir, cfg.save.video_frame_freq)

    if cfg.use_wandb:
        wandb.finish()


def create_single_character_svg_from_font(
    font_path, character, output_path=None, char_size=200, size=300
):
    if output_path is None:
        font_name = os.path.splitext(os.path.basename(font_path))[0]
        output_path = os.path.join(SVGS_PATH, f"init/{font_name}_{character}_{size}.svg")

    # get unicode of character
    unicode = ord(character)

    font = TTFont(font_path)
    glyph_set = font.getGlyphSet()

    # get cmap
    cmap = font["cmap"].getBestCmap()
    glyph_id = cmap[unicode]

    glyph = glyph_set[glyph_id]
    scale = char_size / font["head"].unitsPerEm

    dwg = Drawing(output_path, profile="tiny", size=(size, size))
    pen = SVGPathPen(glyph_set)
    glyph.draw(pen)

    # transform and scale
    # g = dwg.g(transform=f'scale({scale}) translate(0, { char_size })')
    # x_offset = (size - char_size) / 2 + 20
    # y_offset = (size - char_size) / 2 - 30

    x_offset = (size - char_size) / 2
    y_offset = (size - char_size) / 2 - 30
    if character.isascii():
        x_offset += 20
        if character == "W" or character == "M":
            x_offset -= 10
        if character == "y":
            y_offset += 30

    g = dwg.g(
        transform=f"translate({x_offset}, {char_size - y_offset}) scale({scale}, -{scale})"
    )
    path = dwg.path(pen.getCommands(), fill="black")
    g.add(path)
    dwg.add(g)
    dwg.save()
    return output_path


def create_svg_from_font(
    target_iters,
    character,
    font_path,
    signature,
    target_attribute,
    output_path=None,
    char_size=150,
    size=None,
    base_font_path=os.path.join(PARENT_DIR_PATH, "gwfonts/ABeeZee-Regular.ttf"),
    visual_optimize=False,
    visual_font_path=None,
    base_path=os.path.join(OUTPUT_PATH, f"{EXPERIMENT}_"),
):
    optimized_character = character

    font_name = os.path.splitext(os.path.basename(font_path))[0]
    svg_dir_path = os.path.join(
        base_path + character, font_name, signature, "video-svg"
    )
    if output_path is None:
        output_path = os.path.join(OUTPUT_PATH, f"{signature}.svg")
        counter = 0
        while os.path.exists(output_path):
            counter += 1
            output_path = os.path.join(OUTPUT_PATH, f"{signature}_{counter}.svg")
    output_path = output_path.replace(".png", "")

    if size is None:
        size = char_size * (len(target_iters) + 2)

    dwg = Drawing(output_path, profile="tiny", size=(size, char_size * 2))
    for i, target_iter in enumerate(target_iters):
        iter = str(target_iter).zfill(4)
        svg_path = os.path.join(svg_dir_path, f"iter{iter}.svg")
        if not os.path.exists(svg_path):
            raise Exception(f"{svg_path} does not exist")

        # read svg path and draw
        paths, attributes = svg2paths(svg_path)
        for path in paths:
            g = dwg.g(transform=f"translate({char_size * i}, 10)")
            g.add(dwg.path(d=path.d(), fill="black"))
            dwg.add(g)

        # draw iter
        str_target_iter = str(target_iter)
        iter_char_size = char_size / 6
        x_start = char_size * (i + 1) - iter_char_size / 2 * len(str_target_iter)
        y_iter = char_size + 50
        font = TTFont(base_font_path)
        glyph_set = font.getGlyphSet()
        m = len(iter) // 2
        for j, character in enumerate(str_target_iter):
            # get unicode of character
            unicode = ord(character)
            # font = TTFont(font_path)
            # glyph_set = font.getGlyphSet()
            # get cmap
            cmap = font["cmap"].getBestCmap()
            glyph_id = cmap[unicode]
            glyph = glyph_set[glyph_id]

            pen = SVGPathPen(glyph_set)
            glyph.draw(pen)
            scale = iter_char_size / font["head"].unitsPerEm
            g = dwg.g(
                transform=f"translate({x_start + iter_char_size / 1 * (j - m)}, {iter_char_size + y_iter + 10}) scale({scale}, -{scale})"
            )
            path = dwg.path(pen.getCommands(), fill="black")
            g.add(path)
            dwg.add(g)

        # draw visual optimized letter
        if visual_optimize and i == len(target_iters) - 1:
            # draw "target"
            target_sentence = "target"
            x_start += char_size
            for j, character in enumerate(target_sentence):
                # get unicode of character
                unicode = ord(character)
                # font = TTFont(font_path)
                # glyph_set = font.getGlyphSet()
                # get cmap
                cmap = font["cmap"].getBestCmap()
                glyph_id = cmap[unicode]
                glyph = glyph_set[glyph_id]

                pen = SVGPathPen(glyph_set)
                glyph.draw(pen)
                scale = iter_char_size / font["head"].unitsPerEm
                g = dwg.g(
                    transform=f"translate({x_start + iter_char_size / 1 * (j - m)}, {iter_char_size + y_iter + 10}) scale({scale}, -{scale})"
                )
                path = dwg.path(pen.getCommands(), fill="black")
                g.add(path)
                dwg.add(g)

            x_visual = char_size * (i + 1.5)
            y_visual = 10 + char_size
            font = TTFont(visual_font_path)
            glyph_set = font.getGlyphSet()
            unicode = ord(optimized_character)
            cmap = font["cmap"].getBestCmap()
            try:
                glyph_id = cmap[unicode]
                glyph = glyph_set[glyph_id]
                pen = SVGPathPen(glyph_set)
                glyph.draw(pen)
                scale = char_size / font["head"].unitsPerEm
                g = dwg.g(
                    transform=f"translate({x_visual}, {y_visual}) scale({scale}, -{scale})"
                )
                path = dwg.path(pen.getCommands(), fill="black")
                g.add(path)
                dwg.add(g)
            except:
                pass

    # draw target attribute
    x_center = size / 2
    y_attribute = 0
    font = TTFont(base_font_path)
    glyph_set = font.getGlyphSet()
    attribute_char_size = char_size / 4
    m = len(target_attribute) // 2
    if not visual_optimize:
        for i, character in enumerate(target_attribute):
            # get unicode of character
            unicode = ord(character)
            # font = TTFont(font_path)
            # glyph_set = font.getGlyphSet()
            # get cmap
            cmap = font["cmap"].getBestCmap()
            glyph_id = cmap[unicode]
            glyph = glyph_set[glyph_id]
            pen = SVGPathPen(glyph_set)
            glyph.draw(pen)
            scale = attribute_char_size / font["head"].unitsPerEm
            g = dwg.g(
                transform=f"translate({x_center + attribute_char_size / 1 * (i - m)}, {attribute_char_size + y_attribute + 10}) scale({scale}, -{scale})"
            )
            path = dwg.path(pen.getCommands(), fill="black")
            g.add(path)
            dwg.add(g)

    # draw line
    line_x1 = 0
    line_x2 = size - 20
    line_y = attribute_char_size + y_attribute + 175
    line_y = y_iter - 10
    dwg.add(
        dwg.line((line_x1, line_y), (line_x2, line_y), stroke="black", stroke_width=5)
    )

    # draw triangle
    triangle_size = 10
    triangle_y1 = line_y - triangle_size
    triangle_y2 = line_y + triangle_size
    triangle_y3 = line_y
    triangle_x1 = line_x2 - triangle_size
    triangle_x2 = line_x2 - triangle_size
    triangle_x3 = line_x2 + triangle_size
    dwg.add(
        dwg.polygon(
            [
                (triangle_x1, triangle_y1),
                (triangle_x2, triangle_y2),
                (triangle_x3, triangle_y3),
            ],
            fill="black",
        )
    )

    dwg.save()
    return output_path


def show_outline(
    character,
    signature=None,
    iteration=0,
    show_control_points=True,
    convert_cubic_bezier=True,
    figsize=(10, 10),
    num_samples_per_stroke=100,
    save_image=False,
    save_path=None,
    reverse=False,
    svg_path=None,
    font_path=None,
):
    if svg_path is None:
        if signature is not None:
            assert font_path is not None
            base_path = os.path.join(OUTPUT_PATH, f"{EXPERIMENT}_")
            font_name = os.path.splitext(os.path.basename(font_path))[0]
            svg_dir_path = os.path.join(
                base_path + character, font_name, signature, "video-svg"
            )
            iter = str(iteration).zfill(4)
            svg_path = os.path.join(svg_dir_path, f"iter{iter}.svg")
        elif font_path is not None:
            tmp_path = os.path.join(OUTPUT_PATH, "tmp.svg")
            create_single_character_svg_from_font(
                font_path, character, output_path=tmp_path, char_size=200, size=300
            )
            svg_path = tmp_path

            font_name = os.path.splitext(os.path.basename(font_path))[0]
            save_path = os.path.join(OUTPUT_PATH, f"{font_name}_{character}.svg")
            counter = 0
            while os.path.exists(save_path):
                counter += 1
                save_path = os.path.join(
                    OUTPUT_PATH, f"{font_name}_{character}_{counter}.svg"
                )
        else:
            raise ValueError("Either signature or font_path must be given.")

    paths, attributes = svg2paths(svg_path)
    fig, ax = plt.subplots(figsize=figsize)
    num_samples = num_samples_per_stroke

    if convert_cubic_bezier:
        Q_to_C_real = lambda Q: [
            Q.start.real,
            Q.start.real + 2 / 3 * (Q.control.real - Q.start.real),
            Q.end.real + 2 / 3 * (Q.control.real - Q.end.real),
            Q.end.real,
        ]
        Q_to_C_imag = lambda Q: [
            Q.start.imag,
            Q.start.imag + 2 / 3 * (Q.control.imag - Q.start.imag),
            Q.end.imag + 2 / 3 * (Q.control.imag - Q.end.imag),
            Q.end.imag,
        ]
        L_to_C_real = lambda L: [
            L.start.real * (1 - t) + L.end.real * t for t in np.linspace(0, 1, 4)
        ]
        L_to_C_imag = lambda L: [
            L.start.imag * (1 - t) + L.end.imag * t for t in np.linspace(0, 1, 4)
        ]

    for path in paths:
        # print(path)
        for segment in path:
            if segment.__class__.__name__ == "CubicBezier":
                if show_control_points:
                    start_real = segment.start.real
                    start_imag = segment.start.imag
                    control1_real = segment.control1.real
                    control1_imag = segment.control1.imag
                    control2_real = segment.control2.real
                    control2_imag = segment.control2.imag
                    end_real = segment.end.real
                    end_imag = segment.end.imag
                    if not convert_cubic_bezier:
                        ax.plot(
                            [start_real, control1_real],
                            [start_imag, control1_imag],
                            "g-",
                            linewidth=2,
                            alpha=0.3,
                        )
                        ax.plot(
                            [control2_real, end_real],
                            [control2_imag, end_imag],
                            "g-",
                            linewidth=2,
                            alpha=0.3,
                        )
                        # color control points differently
                        ax.plot(
                            [control1_real, control2_real],
                            [control1_imag, control2_imag],
                            "go",
                            markersize=3,
                            alpha=0.3,
                        )
                        ax.plot(
                            [start_real, end_real],
                            [start_imag, end_imag],
                            "ro",
                            markersize=5,
                        )

                # sample points on real outline
                ax.plot(
                    [
                        Path(segment).point(i / num_samples).real
                        for i in range(num_samples)
                    ],
                    [
                        Path(segment).point(i / num_samples).imag
                        for i in range(num_samples)
                    ],
                    "b-",
                    linewidth=1,
                )

            elif segment.__class__.__name__ == "QuadraticBezier":
                if show_control_points:
                    if convert_cubic_bezier:
                        # convert quadratic bezier to cubic bezier
                        (
                            start_real,
                            control1_real,
                            control2_real,
                            end_real,
                        ) = Q_to_C_real(segment)
                        (
                            start_imag,
                            control1_imag,
                            control2_imag,
                            end_imag,
                        ) = Q_to_C_imag(segment)
                        # ax.plot(Q_to_C_real(segment), Q_to_C_imag(segment), 'g-')
                    else:
                        # ax.plot([segment.start.real, segment.control.real, segment.end.real], [segment.start.imag, segment.control.imag, segment.end.imag], 'ro-')
                        ax.plot(
                            [segment.start.real, segment.control.real],
                            [segment.start.imag, segment.control.imag],
                            "g-",
                            linewidth=2,
                            alpha=0.3,
                        )
                        ax.plot(
                            [segment.control.real, segment.end.real],
                            [segment.control.imag, segment.end.imag],
                            "g-",
                            linewidth=2,
                            alpha=0.3,
                        )
                        ax.plot(
                            [segment.start.real, segment.end.real],
                            [segment.start.imag, segment.end.imag],
                            "ro",
                            markersize=5,
                        )

                # sample points on real outline
                ax.plot(
                    [
                        Path(segment).point(i / num_samples).real
                        for i in range(num_samples)
                    ],
                    [
                        Path(segment).point(i / num_samples).imag
                        for i in range(num_samples)
                    ],
                    "b-",
                    linewidth=1,
                )
            elif segment.__class__.__name__ == "Line":
                if show_control_points:
                    if convert_cubic_bezier:
                        # convert quadratic bezier to cubic bezier
                        (
                            start_real,
                            control1_real,
                            control2_real,
                            end_real,
                        ) = L_to_C_real(segment)
                        (
                            start_imag,
                            control1_imag,
                            control2_imag,
                            end_imag,
                        ) = L_to_C_imag(segment)
                        # ax.plot(L_to_C_real(segment), L_to_C_imag(segment), 'g-')
                    else:
                        ax.plot(
                            [segment.start.real, segment.end.real],
                            [segment.start.imag, segment.end.imag],
                            "ro",
                            markersize=5,
                        )
                # sample points on real outline
                ax.plot(
                    [
                        Path(segment).point(i / num_samples).real
                        for i in range(num_samples)
                    ],
                    [
                        Path(segment).point(i / num_samples).imag
                        for i in range(num_samples)
                    ],
                    "b-",
                    linewidth=1,
                )
            else:
                raise ValueError(f"Unknown segment type: {segment.__class__.__name__}")
            if convert_cubic_bezier:
                ax.plot(
                    [start_real, control1_real],
                    [start_imag, control1_imag],
                    "g-",
                    linewidth=2,
                    alpha=0.3,
                )
                ax.plot(
                    [control2_real, end_real],
                    [control2_imag, end_imag],
                    "g-",
                    linewidth=2,
                    alpha=0.3,
                )
                # color control points differently
                ax.plot(
                    [control1_real, control2_real],
                    [control1_imag, control2_imag],
                    "go",
                    markersize=3,
                    alpha=0.3,
                )
                ax.plot(
                    [start_real, end_real], [start_imag, end_imag], "ro", markersize=5
                )

    ax.axes.set_aspect("equal")

    # reverse
    if reverse:
        ax.invert_yaxis()

    if save_image:
        if save_path:
            plt.savefig(save_path)
        else:
            save_path = os.path.join(OUTPUT_PATH, f"{signature}_{iter}.svg")
            counter = 0
            while os.path.exists(save_path):
                counter += 1
                save_path = os.path.join(OUTPUT_PATH, f"{signature}_{iter}_{counter}.svg")
            # save_path = save_path.replace('.png', '')

            plt.savefig(os.path.join(save_path))
        print(f"save_path: {save_path}")
    plt.show()
