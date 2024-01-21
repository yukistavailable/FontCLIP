from trains.config_fine_tune import Config
from models.oft import OFTConfig
from models.lora import LoRAConfig
from trains.trainer import Trainer
from utils.initialize_font_data import (
    fox_text_four_lines,
    all_gray_scale_image_file_dir,
)

if __name__ == "__main__":
    use_unpretrained_model = False
    model_name = "ViT-B/32"
    EPOCH = 250
    BATCH_SIZE = 64
    char_size = 250
    test_char_size = 60
    texts_for_font_image = [fox_text_four_lines]
    image_file_dir = all_gray_scale_image_file_dir
    image_file_dir_for_validation = all_gray_scale_image_file_dir
    trained_model_num = 1
    do_optimize = True
    do_profile = False
    use_multiple_attributes = True
    max_sampled_attributes_num = 3
    max_sampled_attributes_nums = [4, 5, 6]
    random_prompt_num_per_font = 70000
    random_prompt_num_per_font = 10000
    random_prompt_num_per_font = 100
    num_of_prompt_per_font_per_epoch = 30
    # num_of_prompt_per_font_per_epoch = 1
    sample_num = 50
    # sample_num = 200
    # sample_num = 10
    sample_num = 1
    use_aug = True
    use_color_jitter = True
    color_jitter_sample_num = 1
    lower_bound_of_scale = 0.35
    lr = 2e-5 # for direct fine-tuning
    lr = 2e-4 # for lora
    lr_schedular_end_factor = 0.1
    use_negative = True
    geta = 0.2

    predict_mode = False
    if predict_mode:
        num_of_prompt_per_font_per_epoch = 1
        random_prompt_num_per_font = 100
        sample_num = 1
        color_jitter_sample_num = 1
        trained_model_num = 5
        do_optimize = False

    # OFT
    use_oft_vision = False
    use_oft_text = False
    oft_config_vision = OFTConfig(
        r=4,
        eps=6e-3,
        block_share=False,
        is_coft=True,
        apply_q=True,
        apply_k=False,
        apply_v=True,
        apply_out=False,
    )
    oft_config_text = OFTConfig(
        r=4,
        eps=6e-3,
        block_share=False,
        is_coft=True,
        apply_q=True,
        apply_k=False,
        apply_v=True,
        apply_out=True,
    )

    # LoRA
    use_lora_vision = False
    use_lora_text = True
    lora_config_vision = LoRAConfig(
        r=256,
        alpha=512.0,
        bias=False,
        learnable_alpha=False,
        apply_q=True,
        apply_k=True,
        apply_v=True,
        apply_out=True,
    )
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

    # CoOp
    coop_lr = 1e-4
    use_coop_vision = False
    use_coop_text = False
    do_coop_vision_optimize = False
    do_coop_text_optimize = False
    precontext_length_text = 48
    precontext_length_vision = 48
    precontext_dropout_rate = 0.0
    pt_applied_layers = [0]


    # signature = f"cv_5_{i}_ViT-B_32_bce_lora_t-qkvo_256-1024.0_91011_batch64_aug50_cj400_lbound_of_scale0.35_max_attr_num_3_random_p_num_10000_geta0.2_use_negative_lr2e-05-0.1_image_file_dir"
    # checkpoint_path = f"model_checkpoints/{signature}.pt"

    checkpoint_path = None

    config = Config(
        use_unpretrained_model=use_unpretrained_model,
        EPOCH=EPOCH,
        BATCH_SIZE=BATCH_SIZE,
        char_size=char_size,
        test_char_size=test_char_size,
        image_file_dir=image_file_dir,
        image_file_dir_for_validation=image_file_dir_for_validation,
        do_optimize=do_optimize,
        do_profile=do_profile,
        use_multiple_attributes=use_multiple_attributes,
        checkpoint_path=checkpoint_path,
        lr=lr,
        coop_lr=coop_lr,
        lr_schedular_end_factor=lr_schedular_end_factor,
        use_aug=use_aug,
        use_color_jitter=use_color_jitter,
        color_jitter_sample_num=color_jitter_sample_num,
        use_negative=use_negative,
        max_sample_num=max_sampled_attributes_num,
        random_prompts_num=random_prompt_num_per_font,
        sample_num_each_epoch=num_of_prompt_per_font_per_epoch,
        sample_num=sample_num,
        model_name=model_name,
        use_oft_vision=use_oft_vision,
        use_oft_text=use_oft_text,
        oft_config_vision=oft_config_vision,
        oft_config_text=oft_config_text,
        use_lora_vision=use_lora_vision,
        use_lora_text=use_lora_text,
        lora_config_vision=lora_config_vision,
        lora_config_text=lora_config_text,
        use_coop_text=use_coop_text,
        use_coop_vision=use_coop_vision,
        do_coop_vision_optimize=do_coop_vision_optimize,
        do_coop_text_optimize=do_coop_text_optimize,
        precontext_length_text=precontext_length_text,
        precontext_length_vision=precontext_length_vision,
        precontext_dropout_rate=precontext_dropout_rate,
        pt_applied_layers=pt_applied_layers,
        lower_bound_of_scale=lower_bound_of_scale,
        texts_for_font_image=texts_for_font_image,
        geta=geta,
    )
    trainer = Trainer(config)
    result = trainer.train()