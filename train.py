from trains.config_fine_tune import Config
from models.oft import OFTConfig
from models.lora import LoRAConfig
from trains.trainer import Trainer
from utils.initialize_font_data import (
    fox_text_four_lines,
    all_gray_scale_image_file_dir,
)
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune ExCLIP")
    parser.add_argument("--model_name", type=str, default="ViT-B/32")
    parser.add_argument(
        "--use_unpretrained_model",
        action="store_true",
        default=False,
        help="Use unpretrained model. This means that the model is initialized with random weights and trained from scratch.",
    )
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--char_size", type=int, default=250)
    parser.add_argument("--test_char_size", type=int, default=250)
    parser.add_argument("--texts_for_font_image", nargs="+", default=[fox_text_four_lines], help="Texts for rendering font images. This is ignored if image_file_dir is specified.")
    parser.add_argument("--image_file_dir", type=str, default=all_gray_scale_image_file_dir)
    parser.add_argument(
        "--image_file_dir_for_validation", type=str, default=all_gray_scale_image_file_dir
    )
    parser.add_argument("--do_optimize", action="store_true", default=True)
    parser.add_argument("--use_multiple_attributes", action="store_true", default=True)
    parser.add_argument("--max_sampled_attributes_num", type=int, default=3)
    parser.add_argument("--random_prompt_num_per_font", type=int, default=10000)
    parser.add_argument("--num_of_prompt_per_font_per_epoch", type=int, default=30)
    parser.add_argument(
        "--use_aug", action="store_true", default=True, help="Use image augmentation"
    )
    parser.add_argument(
        "--use_color_jitter",
        action="store_true",
        default=True,
        help="Use various colors besides black and white in augmentation",
    )
    parser.add_argument(
        "--sample_num", type=int, default=50, help="Number of augmented images per font"
    )
    parser.add_argument(
        "--color_jitter_sample_num",
        type=int,
        default=200,
        help="Number of augmented images with various colors per font",
    )
    parser.add_argument(
        "--lower_bound_of_scale",
        type=float,
        default=0.35,
        help="Lower bound of scale in augmentation",
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--lr_schedular_end_factor",
        type=float,
        default=0.1,
        help="Factor of learning rate schedular",
    )
    parser.add_argument(
        "--use_negative",
        action="store_true",
        default=True,
        help="Use negative prompt like 'not bold font'",
    )
    parser.add_argument(
        "--geta",
        type=float,
        default=0.2,
        help="geta is added to the attribute score for calculating the probability of each attribute",
    )
    parser.add_argument(
        "--use_oft_vision",
        action="store_true",
        default=False,
        help="Use OFT for vision transformer",
    )
    parser.add_argument(
        "--use_oft_text",
        action="store_true",
        default=False,
        help="Use OFT for text transformer",
    )
    parser.add_argument(
        "--oft_config_vision_r", type=int, default=4, help="OFT r for vision transformer"
    )
    parser.add_argument(
        "--oft_config_vision_eps",
        type=float,
        default=6e-3,
        help="OFT eps for vision transformer",
    )
    parser.add_argument(
        "--oft_config_vision_block_share",
        action="store_true",
        default=False,
        help="OFT block_share for vision transformer",
    )
    parser.add_argument(
        "--oft_config_vision_is_coft",
        action="store_true",
        default=True,
        help="OFT is_coft for vision transformer",
    )
    parser.add_argument(
        "--oft_config_vision_apply_q",
        action="store_true",
        default=True,
        help="OFT apply_q for vision transformer",
    )
    parser.add_argument(
        "--oft_config_vision_apply_k",
        action="store_true",
        default=False,
        help="OFT apply_k for vision transformer",
    )
    parser.add_argument(
        "--oft_config_vision_apply_v",
        action="store_true",
        default=True,
        help="OFT apply_v for vision transformer",
    )
    parser.add_argument(
        "--oft_config_vision_apply_out",
        action="store_true",
        default=False,
        help="OFT apply_out for vision transformer",
    )
    parser.add_argument(
        "--oft_config_text_r", type=int, default=4, help="OFT r for text transformer"
    )
    parser.add_argument(
        "--oft_config_text_eps",
        type=float,
        default=6e-3,
        help="OFT eps for text transformer",
    )
    parser.add_argument(
        "--oft_config_text_block_share",
        action="store_true",
        default=False,
        help="OFT block_share for text transformer",
    )
    parser.add_argument(
        "--oft_config_text_is_coft",
        action="store_true",
        default=True,
        help="OFT is_coft for text transformer",
    )
    parser.add_argument(
        "--oft_config_text_apply_q",
        action="store_true",
        default=True,
        help="OFT apply_q for text transformer",
    )
    parser.add_argument(
        "--oft_config_text_apply_k",
        action="store_true",
        default=False,
        help="OFT apply_k for text transformer",
    )
    parser.add_argument(
        "--oft_config_text_apply_v",
        action="store_true",
        default=True,
        help="OFT apply_v for text transformer",
    )
    parser.add_argument(
        "--oft_config_text_apply_out",
        action="store_true",
        default=True,
        help="OFT apply_out for text transformer",
    )
    parser.add_argument(
        "--use_lora_vision",
        action="store_true",
        default=False,
        help="Use LoRA for vision transformer",
    )
    parser.add_argument(
        "--use_lora_text",
        action="store_true",
        default=False,
        help="Use LoRA for text transformer",
    )
    parser.add_argument(
        "--lora_config_vision_r",
        type=int,
        default=256,
        help="LoRA r for vision transformer",
    )
    parser.add_argument(
        "--lora_config_vision_alpha",
        type=float,
        default=1024.0,
        help="LoRA alpha for vision transformer",
    )
    parser.add_argument(
        "--lora_config_vision_bias",
        action="store_true",
        default=False,
        help="LoRA bias for vision transformer",
    )
    parser.add_argument(
        "--lora_config_vision_learnable_alpha",
        action="store_true",
        default=False,
        help="LoRA learnable_alpha for vision transformer",
    )
    parser.add_argument(
        "--lora_config_vision_apply_q",
        action="store_true",
        default=True,
        help="LoRA apply_q for vision transformer",
    )
    parser.add_argument(
        "--lora_config_vision_apply_k",
        action="store_true",
        default=True,
        help="LoRA apply_k for vision transformer",
    )
    parser.add_argument(
        "--lora_config_vision_apply_v",
        action="store_true",
        default=True,
        help="LoRA apply_v for vision transformer",
    )
    parser.add_argument(
        "--lora_config_vision_apply_out",
        action="store_true",
        default=True,
        help="LoRA apply_out for vision transformer",
    )
    parser.add_argument(
        "--lora_config_text_r", type=int, default=256, help="LoRA r for text transformer"
    )
    parser.add_argument(
        "--lora_config_text_alpha",
        type=float,
        default=1024.0,
        help="LoRA alpha for text transformer",
    )
    parser.add_argument(
        "--lora_config_text_bias",
        action="store_true",
        default=False,
        help="LoRA bias for text transformer",
    )
    parser.add_argument(
        "--lora_config_text_learnable_alpha",
        action="store_true",
        default=False,
        help="LoRA learnable_alpha for text transformer",
    )
    parser.add_argument(
        "--lora_config_text_apply_q",
        action="store_true",
        default=True,
        help="LoRA apply_q for text transformer",
    )
    parser.add_argument(
        "--lora_config_text_apply_k",
        action="store_true",
        default=True,
        help="LoRA apply_k for text transformer",
    )
    parser.add_argument(
        "--lora_config_text_apply_v",
        action="store_true",
        default=True,
        help="LoRA apply_v for text transformer",
    )
    parser.add_argument(
        "--lora_config_text_apply_out",
        action="store_true",
        default=True,
        help="LoRA apply_out for text transformer",
    )
    parser.add_argument(
        "--use_coop_vision",
        action="store_true",
        default=False,
        help="Use CoOp for vision transformer",
    )
    parser.add_argument(
        "--use_coop_text",
        action="store_true",
        default=False,
        help="Use CoOp for text transformer",
    )
    parser.add_argument("--coop_lr", type=float, default=1e-4, help="CoOp learning rate")
    parser.add_argument(
        "--precontext_length_text",
        type=int,
        default=48,
        help="CoOp precontext_length for text transformer",
    )
    parser.add_argument(
        "--precontext_length_vision",
        type=int,
        default=48,
        help="CoOp precontext_length for vision transformer",
    )
    parser.add_argument(
        "--precontext_dropout_rate",
        type=float,
        default=0.0,
        help="CoOp precontext_dropout_rate for text transformer",
    )
    parser.add_argument(
        "--pt_applied_layers",
        type=int,
        nargs="+",
        default=[0],
        help="CoOp pt_applied_layers for text transformer",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None, help="Path to the checkpoint file"
    )
    args = parser.parse_args()
    # OFT
    oft_config_vision = OFTConfig(
        r=args.oft_config_vision_r,
        eps=args.oft_config_vision_eps,
        block_share=args.oft_config_vision_block_share,
        is_coft=args.oft_config_vision_is_coft,
        apply_q=args.oft_config_vision_apply_q,
        apply_k=args.oft_config_vision_apply_k,
        apply_v=args.oft_config_vision_apply_v,
        apply_out=args.oft_config_vision_apply_out,
    )
    oft_config_text = OFTConfig(
        r=args.oft_config_text_r,
        eps=args.oft_config_text_eps,
        block_share=args.oft_config_text_block_share,
        is_coft=args.oft_config_text_is_coft,
        apply_q=args.oft_config_text_apply_q,
        apply_k=args.oft_config_text_apply_k,
        apply_v=args.oft_config_text_apply_v,
        apply_out=args.oft_config_text_apply_out,
    )

    # LoRA
    lora_config_vision = LoRAConfig(
        r=args.lora_config_vision_r,
        alpha=args.lora_config_vision_alpha,
        bias=args.lora_config_vision_bias,
        learnable_alpha=args.lora_config_vision_learnable_alpha,
        apply_q=args.lora_config_vision_apply_q,
        apply_k=args.lora_config_vision_apply_k,
        apply_v=args.lora_config_vision_apply_v,
        apply_out=args.lora_config_vision_apply_out,
    )
    lora_config_text = LoRAConfig(
        r=args.lora_config_text_r,
        alpha=args.lora_config_text_alpha,
        bias=args.lora_config_text_bias,
        learnable_alpha=args.lora_config_text_learnable_alpha,
        apply_q=args.lora_config_text_apply_q,
        apply_k=args.lora_config_text_apply_k,
        apply_v=args.lora_config_text_apply_v,
        apply_out=args.lora_config_text_apply_out,
    )

    config = Config(
        use_unpretrained_model=args.use_unpretrained_model,
        EPOCH=args.epoch,
        BATCH_SIZE=args.batch_size,
        char_size=args.char_size,
        test_char_size=args.test_char_size,
        image_file_dir=args.image_file_dir,
        image_file_dir_for_validation=args.image_file_dir_for_validation,
        do_optimize=args.do_optimize,
        use_multiple_attributes=args.use_multiple_attributes,
        checkpoint_path=args.checkpoint_path,
        lr=args.lr,
        coop_lr=args.coop_lr,
        lr_schedular_end_factor=args.lr_schedular_end_factor,
        use_aug=args.use_aug,
        use_color_jitter=args.use_color_jitter,
        color_jitter_sample_num=args.color_jitter_sample_num,
        use_negative=args.use_negative,
        max_sample_num=args.max_sampled_attributes_num,
        random_prompts_num=args.random_prompt_num_per_font,
        sample_num_each_epoch=args.num_of_prompt_per_font_per_epoch,
        sample_num=args.sample_num,
        model_name=args.model_name,
        use_oft_vision=args.use_oft_vision,
        use_oft_text=args.use_oft_text,
        oft_config_vision=oft_config_vision,
        oft_config_text=oft_config_text,
        use_lora_vision=args.use_lora_vision,
        use_lora_text=args.use_lora_text,
        lora_config_vision=lora_config_vision,
        lora_config_text=lora_config_text,
        use_coop_text=args.use_coop_text,
        use_coop_vision=args.use_coop_vision,
        precontext_length_text=args.precontext_length_text,
        precontext_length_vision=args.precontext_length_vision,
        precontext_dropout_rate=args.precontext_dropout_rate,
        pt_applied_layers=args.pt_applied_layers,
        lower_bound_of_scale=args.lower_bound_of_scale,
        texts_for_font_image=args.texts_for_font_image,
        geta=args.geta,
    )
    trainer = Trainer(config)
    result = trainer.train()
