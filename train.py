from utils.config_fine_tune import Config
from utils.oft import OFTConfig
from utils.lora import LoRAConfig
from utils.trainer import Trainer
from utils.initialize_font_data import (
    gray_scale_image_file_dir,
    fox_text_four_lines,
    all_gray_scale_image_file_dir,
    fox_text,
)

if __name__ == "__main__":
    use_unpretrained_model = False
    model_name = "ViT-B/32"
    # model_name = "ViT-L/14"
    EPOCH = 250
    BATCH_SIZE = 64
    char_size = 250
    test_char_size = 60
    texts_for_font_image = [fox_text_four_lines]
    use_unlabeled_data = False
    unlabeled_sampling_ratio = 0.25
    unlabeled_sampling_ratios = [0.1, 0.2, 0.3, 0.4]
    unlabeled_sample_num = 100
    unlabeled_random_prompts_num = 5000
    image_file_dir = None
    image_file_dir_for_validation = None
    image_file_dir = all_gray_scale_image_file_dir
    image_file_dir_for_validation = all_gray_scale_image_file_dir
    trained_model_num = 1
    do_optimize = True
    do_profile = False
    task_for_validation = False
    do_cross_validation = True
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
    lr = 2e-5
    lr_schedular_end_factor = 0.1
    use_bce_loss = True
    use_contrastive_image_loss = False
    contrastive_image_loss_weight = 0.1
    use_triplet_image_loss = False
    triplet_image_loss_weight = 1.0
    triplet_image_loss_margin = 0.05
    use_negative_loss = False
    use_negative = True
    use_single_character = False
    use_clip_like_format = False
    geta = 0.2

    predict_mode = False
    if predict_mode:
        num_of_prompt_per_font_per_epoch = 1
        random_prompt_num_per_font = 100
        sample_num = 1
        color_jitter_sample_num = 1
        trained_model_num = 5
        do_optimize = False
        task_for_validation = True
        do_cross_validation = True

    train_only_visual_encoder = False
    use_same_text_for_each_pair = False
    if train_only_visual_encoder:
        use_single_character = False

    # OFT
    oft_lr = 2e-4
    # oft_lr = 5e-4
    use_oft_vision = False
    use_oft_text = False
    oft_config_vision = OFTConfig(
        r=4,
        # eps = 6e-5,
        eps=6e-3,
        # eps = 6e-2,
        # eps = 2e-3,
        # eps = 6e-1,
        block_share=False,
        is_coft=True,
        apply_q=True,
        apply_k=False,
        apply_v=True,
        apply_out=False,
    )
    oft_config_text = OFTConfig(
        r=4,
        # eps = 6e-5,
        eps=6e-3,
        # eps = 6e-2,
        # eps = 2e-3,
        # eps = 6e-1,
        block_share=False,
        is_coft=True,
        apply_q=True,
        apply_k=False,
        apply_v=True,
        apply_out=True,
    )

    # LoRA
    lora_lr = 2e-4
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
    # precontext_length_text = 62
    precontext_lengths = [1, 4, 8, 16, 32, 48]
    precontext_length_vision = 48
    precontext_dropout_rate = 0.0
    pt_applied_layers = [0]
    # pt_applied_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    use_chopped_clip = False
    chopped_clip_vision_layers = None
    chopped_clip_text_layers = 3

    negative_loss_weights = [0.0, 1e-3, 1e-4, 1e-5, 1e-6]
    negative_loss_weight = 1e-6
    lower_bound_of_scales = [0.5, 0.7, 0.9]
    lower_bound_of_scale = 0.35

    # for precontext_vision_length in precontext_vision_lengths:
    # for negative_loss_weight in negative_loss_weights:
    # for lr in lrs:
    # for triplet_image_loss_weight in [0.1, 0.01, 0.2, 0.3]:
    if True:
        # for unlabeled_sampling_ratio in unlabeled_sampling_ratios:
        # for max_sampled_attributes_num in max_sampled_attributes_nums:
        # for use_clip_like_format in [True, False]:
        # for lower_bound_of_scale in lower_bound_of_scales:
        result = []
        for i in range(0, trained_model_num):
            signature = f"cv_5_4_ViT-B_32_bce_loss_9101191011_batch64_aug250_lower_bound_of_scale0.35_max_attribute_num_3_random_prompts_num_70000_use_negative_lr2e-05-0.1_image_file_dir"
            signature = f"cv_5_{i}_ViT-B_32_bce_coop_precontext_length48_lr0.0001_91011_batch64_aug250_lbound_of_scale0.35_max_attr_num_3_random_p_num_70000_geta0.2_use_negative_lr2e-05-0.1_image_file_dir"
            signature = f"cv_5_{i}_ViT-B_32_bce_lora_t-qkvo_256-1024.0_91011_batch64_aug10_lbound_of_scale0.35_max_attr_num_3_random_p_num_100_geta0.2_use_negative_til1.0_lr2e-05-0.1_image_file_dir"
            signature = f"cv_5_{i}_ViT-B_32_bce_lora_t-qkvo_256-1024.0_91011_batch64_aug250_lbound_of_scale0.35_max_attr_num_3_random_p_num_70000_geta0.2_use_negative_til1.0_lr2e-05-0.1_image_file_dir"
            signature = f"cv_5_{i}_ViT-B_32_bce_9101191011_batch64_aug250_lbound_of_scale0.35_max_attr_num_3_random_p_num_70000_geta0.2_use_negative_til0.2_lr2e-05-0.1_image_file_dir"
            signature = f"cv_5_{i}_ViT-B_32_bce_lora_t-qkvo_256-1024.0_91011_batch64_aug250_lbound_of_scale0.35_max_attr_num_3_random_p_num_70000_geta0.2_use_negative_til1.0_lr2e-05-0.1_image_file_dir"
            signature = f"cv_5_{i}_ViT-B_32_bce_lora_t-qkvo_256-1024.0_91011_batch64_aug10_lbound_of_scale0.35_max_attr_num_3_random_p_num_100_geta0.2_use_negative_lr2e-05-0.1_image_file_dir"
            signature = f"cv_5_{i}_ViT-B_32_bce_lora_t-qkvo_256-1024.0_91011_batch64_aug50_cj200_lbound_of_scale0.35_max_attr_num_3_random_p_num_70000_geta0.2_use_negative_lr2e-05-0.1_image_file_dir"
            signature = f"cv_5_{i}_ViT-B_32_bce_lora_t-qkvo_256-1024.0_91011_batch64_aug50_cj400_lbound_of_scale0.35_max_attr_num_3_random_p_num_10000_geta0.2_use_negative_lr2e-05-0.1_image_file_dir"

            checkpoint_path = f"model_checkpoints/{signature}.pt"

            checkpoint_path = None

            config = Config(
                use_unpretrained_model=use_unpretrained_model,
                EPOCH=EPOCH,
                BATCH_SIZE=BATCH_SIZE,
                use_unlabeled_data=use_unlabeled_data,
                unlabeled_sampling_ratio=unlabeled_sampling_ratio,
                unlabeled_sample_num=unlabeled_sample_num,
                unlabeled_random_prompts_num=unlabeled_random_prompts_num,
                char_size=char_size,
                test_char_size=test_char_size,
                image_file_dir=image_file_dir,
                image_file_dir_for_validation=image_file_dir_for_validation,
                do_optimize=do_optimize,
                do_cross_validation=do_cross_validation,
                do_profile=do_profile,
                use_multiple_attributes=use_multiple_attributes,
                checkpoint_path=checkpoint_path,
                lr=lr,
                coop_lr=coop_lr,
                lr_schedular_end_factor=lr_schedular_end_factor,
                use_aug=use_aug,
                use_color_jitter=use_color_jitter,
                color_jitter_sample_num=color_jitter_sample_num,
                use_bce_loss=use_bce_loss,
                use_contrastive_image_loss=use_contrastive_image_loss,
                contrastive_image_loss_weight=contrastive_image_loss_weight,
                use_triplet_image_loss=use_triplet_image_loss,
                triplet_image_loss_weight=triplet_image_loss_weight,
                triplet_image_loss_margin=triplet_image_loss_margin,
                use_negative=use_negative,
                use_negative_loss=use_negative_loss,
                negative_loss_weight=negative_loss_weight,
                max_sample_num=max_sampled_attributes_num,
                random_prompts_num=random_prompt_num_per_font,
                sample_num_each_epoch=num_of_prompt_per_font_per_epoch,
                sample_num=sample_num,
                task_for_validation=task_for_validation,
                model_name=model_name,
                use_single_character=use_single_character,
                train_only_visual_encoder=train_only_visual_encoder,
                use_same_text_for_each_pair=use_same_text_for_each_pair,
                use_chopped_clip=use_chopped_clip,
                chopped_clip_vision_layers=chopped_clip_vision_layers,
                chopped_clip_text_layers=chopped_clip_text_layers,
                oft_lr=oft_lr,
                use_oft_vision=use_oft_vision,
                use_oft_text=use_oft_text,
                oft_config_vision=oft_config_vision,
                oft_config_text=oft_config_text,
                use_lora_vision=use_lora_vision,
                use_lora_text=use_lora_text,
                lora_lr=lora_lr,
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
                use_clip_like_format=use_clip_like_format,
                lower_bound_of_scale=lower_bound_of_scale,
                texts_for_font_image=texts_for_font_image,
                start_index_for_train_model=i,
                trained_model_num=trained_model_num,
                use_fast_evaluator=True,
                geta=geta,
            )
            trainer = Trainer(config)
            tmp_result = trainer.train()
            if tmp_result is not None:
                result.append(tmp_result)
        if tmp_result is not None:
            final_result = {}
            for key in tmp_result.keys():
                average = sum([r[key] for r in result]) / len(result)
                variance = (
                    sum([(r[key] - average) ** 2 for r in result]) / len(result)
                ) ** 0.5
                final_result[key] = (average, variance)
            print(final_result)
