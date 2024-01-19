from copy import copy
from dataclasses import dataclass
import json
import numpy as np
from logging import getLogger
import os
from tqdm import tqdm
import tracemalloc

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

from models.init_model import (
    load_model,
    device,
    preprocess,
    convert_weights,
    preprocess_for_single_character,
)
from utils.initialize_font_data import (
    font_dir,
    validation_json_path,
    test_json_path,
    all_json,
    validation_font_names,
    test_font_names,
    inclusive_attributes,
    exclusive_attributes,
    val_all_gwfonts_json_path,
)
from evals.evaluate_tools import (
    retrieve_one_leave_out_model_path,
    evaluate_attribute_comparison_task_for_each_comparison,
    evaluate_similarity_comparison_task,
    evaluate_use_dumped_image,
    evaluate_correlation_coefficient,
)
from dataset.dataset import (
    MyDataset,
    TestDataset,
    set_image_tensors,
    TestImageDataset,
    TestTextDataset,
)
from trains.config_fine_tune import Config

logger = getLogger(__name__)


@dataclass
class Trainer:
    config: Config = None

    def __post_init__(self):
        pass

    def train(self):
        for i in range(
            self.config.start_index_for_train_model, self.config.trained_model_num
        ):
            if self.config.do_profile:
                tracemalloc.start()
            tmp_exclusive_attributes = copy(exclusive_attributes)
            tmp_inclusive_attributes = copy(inclusive_attributes)
            tmp_validation_json_path = validation_json_path
            tmp_test_json_path = test_json_path
            val_texts_for_font_image = self.config.val_texts_for_font_image
            test_texts_for_font_image = self.config.val_texts_for_font_image
            print("val_texts_for_font_image", val_texts_for_font_image)
            if self.config.use_single_character:
                self.config.image_file_dir_for_validation = None
            tmp_validation_font_names = validation_font_names
            tmp_test_font_names = test_font_names

            if (
                self.config.do_cross_validation
            ):
                # update signature
                self.config.set_signature(cross_validation_index=i)
                # update json path and font names
                # TODO: prepare the function to update json path in Config
                self.config.json_path = f"../attributeData/train_font_to_attribute_values_cross_validation_{self.config.trained_model_num}_{i}.json"
                tmp_validation_json_path = f"../attributeData/validation_font_to_attribute_values_cross_validation_{self.config.trained_model_num}_{i}.json"
                tmp_test_json_path = f"../attributeData/test_font_to_attribute_values_cross_validation_{self.config.trained_model_num}_{i}.json"
                tmp_validation_json = json.load(open(tmp_validation_json_path, "r"))
                tmp_test_json = json.load(open(tmp_test_json_path, "r"))
                tmp_validation_font_names = list(tmp_validation_json.keys())
                tmp_test_font_names = list(tmp_test_json.keys())
                print(i, tmp_validation_font_names[0], tmp_test_font_names[0])

            # leave one attribute out
            if self.config.leave_out_attributes:
                leave_out_attribute = self.config.leave_out_attributes[i]
                print(
                    "===================== Leave One Out Attribute ================================"
                )
                print(leave_out_attribute)

                tmp_exclusive_attributes += [leave_out_attribute]
                tmp_inclusive_attributes = [
                    a
                    for a in tmp_inclusive_attributes
                    if a not in [leave_out_attribute]
                ]
                print(len(tmp_inclusive_attributes))
                print(len(tmp_exclusive_attributes))
            # leave one font out
            elif self.config.one_leave_out:
                print(
                    "===================== Leave One Out ================================"
                )
                one_left_out_font_name = list(all_json.keys())[i]
                print(one_left_out_font_name)
                if os.path.exists(
                    retrieve_one_leave_out_model_path(one_left_out_font_name)
                ):
                    continue

                self.config.json_path = self.config.one_leave_out_json_path
                self.config.one_leave_out_json = all_json
                self.config.one_leave_out_json.pop(one_left_out_font_name)

                with open(self.config.one_leave_out_json_path, "w") as f:
                    json.dump(self.config.one_leave_out_json, f)

            print("===================== Load Model ================================")
            print(self.config.checkpoint_path)

            print("Use CoOp", self.config.use_coop_text)
            print("Use VPT", self.config.use_coop_vision)
            self.config.model = load_model(
                self.config.checkpoint_path,
                model_name=self.config.model_name,
                use_oft_vision=self.config.use_oft_vision,
                use_oft_text=self.config.use_oft_text,
                oft_config_vision=self.config.oft_config_vision,
                oft_config_text=self.config.oft_config_text,
                use_lora_vision=self.config.use_lora_vision,
                use_lora_text=self.config.use_lora_text,
                lora_config_vision=self.config.lora_config_vision,
                lora_config_text=self.config.lora_config_text,
                use_coop_vision=self.config.use_coop_vision,
                use_coop_text=self.config.use_coop_text,
                precontext_length_vision=self.config.precontext_length_vision,
                precontext_length_text=self.config.precontext_length_text,
                precontext_dropout_rate=self.config.precontext_dropout_rate,
                pt_applied_layers=self.config.pt_applied_layers,
            )
            if self.config.use_unpretrained_model:
                print("Warning: use unpretrained model")
                self.config.model.initialize_parameters()
            print(self.config.signature)

            dataset = MyDataset(
                font_dir,
                self.config.json_path,
                texts_for_font_image=self.config.texts_for_font_image,
                use_negative=self.config.use_negative,
                use_weight=self.config.use_weight,
                use_score=self.config.use_score,
                use_multiple_attributes=self.config.use_multiple_attributes,
                use_random_attributes=self.config.use_random_attributes,
                random_prompts_num=self.config.random_prompts_num,
                max_sample_num=self.config.max_sample_num,
                rich_prompt=self.config.rich_prompt,
                sample_num_each_epoch=self.config.sample_num_each_epoch,
                image_file_dir=self.config.image_file_dir,
                attribute_threshold=self.config.attribute_threshold,
                attribute_under_threshold=self.config.attribute_under_threshold,
                preprocess=self.config.tmp_preprocess,
                dump_image=self.config.train_dump_image,
                exclusive_attributes=tmp_exclusive_attributes,
                geta=self.config.geta,
                single_character=self.config.single_character,
                use_bce_loss=self.config.use_bce_loss,
                char_size=self.config.char_size,
                context_length=self.config.context_length,
            )
            print(
                "train prompts num: ",
                sum([len(v) for v in dataset.font_to_attributes.values()]),
            )
            print("train dataset font num: ", len(dataset.font_paths))

            # use aug
            if self.config.use_aug or self.config.tmp_preprocess != preprocess:
                set_image_tensors(
                    dataset,
                    preprocess=self.config.tmp_preprocess,
                    sample_num=self.config.sample_num,
                    color_jitter_sample_num=self.config.color_jitter_sample_num if self.config.use_color_jitter else 0
                )
                print(
                    f"{sum([len(image_tensors) for image_tensors in dataset.font_text_to_image_tensors])} images are randomly created."
                )

            val_datasets = None
            test_datasets = None
            if dataset.use_score:
                val_dataset = TestDataset(
                    font_dir,
                    tmp_validation_json_path,
                    [self.config.texts_for_font_image[0]],
                    target_attributes=tmp_inclusive_attributes,
                    preprocess=preprocess,
                    dump_image=self.config.tmp_dump_image,
                    image_file_dir=self.config.image_file_dir_for_validation,
                    single_character=self.config.single_character,
                    use_score=self.config.use_score,
                    char_size=self.config.test_char_size,
                    context_length=self.config.context_length,
                )
                test_dataset = TestDataset(
                    font_dir,
                    tmp_test_json_path,
                    [self.config.texts_for_font_image[0]],
                    target_attributes=tmp_inclusive_attributes,
                    preprocess=preprocess,
                    dump_image=self.config.tmp_dump_image,
                    image_file_dir=self.config.image_file_dir_for_validation,
                    single_character=self.config.single_character,
                    use_score=self.config.use_score,
                    char_size=self.config.test_char_size,
                    context_length=self.config.context_length,
                )
                print("val dataset font num: ", len(val_dataset.font_paths))
                print("test dataset font num: ", len(test_dataset.font_paths))

            elif self.config.use_fast_evaluator:
                val_image_dataset = TestImageDataset(
                    font_dir,
                    tmp_validation_json_path,
                    val_texts_for_font_image[0],
                    dump_image=self.config.tmp_dump_image,
                    image_file_dir=self.config.image_file_dir_for_validation,
                    preprocess=preprocess,
                )
                test_image_dataset = TestImageDataset(
                    font_dir,
                    tmp_test_json_path,
                    test_texts_for_font_image[0],
                    dump_image=self.config.tmp_dump_image,
                    image_file_dir=self.config.image_file_dir_for_validation,
                    preprocess=preprocess,
                )
                text_dataset = TestTextDataset(
                    target_attributes=tmp_inclusive_attributes,
                    context_length=self.config.context_length,
                )

            else:
                val_datasets = [
                    TestDataset(
                        font_dir,
                        tmp_validation_json_path,
                        val_texts_for_font_image,
                        target_attributes=[target_attribute],
                        preprocess=preprocess,
                        dump_image=self.config.tmp_dump_image,
                        image_file_dir=self.config.image_file_dir_for_validation,
                        single_character=self.config.single_character,
                        char_size=self.config.test_char_size,
                        context_length=self.config.context_length,
                    )
                    for target_attribute in tmp_inclusive_attributes
                ]
                test_datasets = [
                    TestDataset(
                        font_dir,
                        tmp_test_json_path,
                        test_texts_for_font_image,
                        target_attributes=[target_attribute],
                        preprocess=preprocess,
                        dump_image=self.config.tmp_dump_image,
                        image_file_dir=self.config.image_file_dir_for_validation,
                        single_character=self.config.single_character,
                        char_size=self.config.test_char_size,
                        context_length=self.config.context_length,
                    )
                    for target_attribute in tmp_inclusive_attributes
                ]
                print("val dataset font num: ", len(val_datasets[0].font_paths))
                print("test dataset font num: ", len(test_datasets[0].font_paths))

            # Define your own dataloader
            train_dataloader = DataLoader(
                dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=True,
                pin_memory=False,
                num_workers=0,
            )

            # https://github.com/openai/CLIP/issues/57
            def convert_models_to_fp32(model):
                for p in model.parameters():
                    p.data = p.data.float()
                    if p.grad is not None:
                        p.grad.data = p.grad.data.float()

            if device == "cpu":
                self.config.model.float()
            else:
                # Actually this line is unnecessary since clip by default already on
                # float16
                convert_weights(self.config.model)

            # freeze the model except the last layer
            for param in self.config.model.parameters():
                param.requires_grad = False
            # model.text_projection.requires_grad = False
            # model.ln_final.requires_grad = False

            # unfreeze the target layers in text encoder
            for name, param in self.config.model.transformer.named_parameters():
                if any(
                    [
                        name.startswith(target_layer)
                        for target_layer in self.config.target_layers_text
                    ]
                ):
                    param.requires_grad = True

            # unfreeze the target layers in vision encoder
            for name, param in self.config.model.visual.named_parameters():
                if any(
                    [
                        name.startswith(target_layer)
                        for target_layer in self.config.target_layers_vision
                    ]
                ):
                    param.requires_grad = True
            for name, param in self.config.model.named_parameters():
                if name == "logit_scale":
                    pass
                    # print('Optimize logit_scale')
                    # param.requires_grad = False

            if self.config.use_oft_vision or self.config.use_oft_text:
                print("Optimize OFT")
                for name, param in self.config.model.named_parameters():
                    if "oft" in name:
                        param.requires_grad = True
            if self.config.use_lora_vision or self.config.use_lora_text:
                print("Optimize LoRA")
                for name, param in self.config.model.named_parameters():
                    if "lora_proj" in name or (
                        (
                            self.config.lora_config_vision.learnable_alpha
                            or self.config.lora_config_text.learnable_alpha
                        )
                        and "lora_alpha" in name
                    ):
                        if "visual" in name:
                            if self.config.use_lora_vision:
                                param.requires_grad = True
                            else:
                                param.requires_grad = False
                        else:
                            if self.config.use_lora_text:
                                param.requires_grad = True
                            else:
                                param.requires_grad = False
                    # else:
                    #     param.requires_grad = False
            if self.config.use_coop_text:
                if self.config.do_coop_text_optimize:
                    print("Optimize prompt")
                for name, param in self.config.model.named_parameters():
                    if name == "precontext":
                        if self.config.do_coop_text_optimize:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
            if self.config.use_coop_vision:
                if self.config.do_coop_vision_optimize:
                    print("Optimize vision prompt")
                for name, param in self.config.model.named_parameters():
                    if name.startswith("visual.precontext_vision"):
                        if self.config.do_coop_vision_optimize:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False

            for name, param in self.config.model.named_parameters():
                if param.requires_grad:
                    print(name)

            if not (False or False):
                if len(self.config.target_layers_vision) != 0:
                    print("Optimize visual proj")
                    self.config.model.visual.proj.requires_grad = True
                if len(self.config.target_layers_text) != 0:
                    print("Optimize ln_final")
                    self.config.model.ln_final.requires_grad = True

            loss_mse = nn.MSELoss()
            if self.config.use_bce_loss:
                loss_img = nn.BCEWithLogitsLoss()
                loss_txt = nn.BCEWithLogitsLoss()
            else:
                loss_img = nn.CrossEntropyLoss()
                loss_txt = nn.CrossEntropyLoss()
            if self.config.use_contrastive_image_loss:
                loss_img_contrastive = nn.CrossEntropyLoss()

            # Params used from paper, the lr is smaller, more safe for fine-tuning to
            # new dataset
            if self.config.use_coop_text or self.config.use_coop_vision:
                # optimizer for config.model.precontext
                coop_optimizer = optim.Adam(
                    self.config.model.coop_parameters(),
                    lr=self.config.coop_lr,
                    betas=(0.9, 0.98),
                    eps=1e-6,
                    weight_decay=0.2,
                )
                coop_schedular = optim.lr_scheduler.LinearLR(
                    coop_optimizer,
                    start_factor=1,
                    end_factor=self.config.lr_schedular_end_factor,
                    total_iters=self.config.EPOCH * len(train_dataloader),
                )
            if self.config.use_oft_vision or self.config.use_oft_text or self.config.use_lora_vision or self.config.use_lora_text:
                print("Optimize OFT or LoRA")
                optimizer = optim.Adam(
                    # self.config.model.lora_parameters(),
                    self.config.model.parameters(),
                    lr=self.config.oft_lr,
                    betas=(0.9, 0.98),
                    eps=1e-6,
                    weight_decay=0.2,
                )
            # elif self.config.use_lora_vision or self.config.use_lora_text:
            #     optimizer = optim.Adam(
            #         # self.config.model.lora_parameters(),
            #         self.config.model.parameters(),
            #         lr=self.config.lora_lr,
            #         betas=(0.9, 0.98),
            #         eps=1e-6,
            #         weight_decay=0.2,
            #     )
            else:
                optimizer = optim.Adam(
                    self.config.model.parameters(),
                    lr=self.config.lr,
                    betas=(0.9, 0.98),
                    eps=1e-6,
                    weight_decay=0.2,
                )
            schedular = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1,
                end_factor=self.config.lr_schedular_end_factor,
                total_iters=self.config.EPOCH * len(train_dataloader),
            )

            # add your own code to track the training progress.
            self.config.model.train()
            best_val_loss = None
            best_epoch = 1
            batch_max_num = 250
            for epoch in range(self.config.EPOCH):
                epoch_loss = 0
                batch_count = 0
                for batch in tqdm(train_dataloader):
                    batch_count += 1
                    if batch_count > batch_max_num:
                        break

                    optimizer.zero_grad()
                    if self.config.use_coop_text or self.config.use_coop_vision:
                        coop_optimizer.zero_grad()

                    weights = None
                    images_contrastive = None
                    if dataset.use_score:
                        images, texts, scores = batch
                        scores = torch.tensor(scores, dtype=torch.float16)
                    elif dataset.use_weight:
                        images, texts, weights = batch
                        # to Half torch tensor
                        weights = torch.tensor(weights, dtype=torch.float16)
                    elif self.config.use_bce_loss:
                        if self.config.use_contrastive_image_loss or self.config.use_triplet_image_loss:
                            (
                                images,
                                images_contrastive,
                                texts,
                                font_indices,
                                attribute_indices,
                            ) = batch
                        else:
                            images, texts, font_indices, attribute_indices = batch
                        font_indices = font_indices.to(device)
                        attribute_indices = attribute_indices.to(device)
                        mask_matrix = dataset.mask_font_idx_signed_attribute_matrix_ground_truth_fast(
                            font_indices, attribute_indices
                        )
                        mask_matrix = mask_matrix.to(device).to(torch.float16)
                    else:
                        images, texts = batch

                    if images_contrastive is not None:
                        images_contrastive = images_contrastive.to(device)
                    images = images.to(device)
                    texts = texts.to(device)

                    if self.config.use_coop_text or self.config.use_coop_vision:
                        image_features = self.config.model.encode_image(images)
                        text_features = self.config.model.encode_text(texts)
                        image_features = image_features / image_features.norm(
                            dim=-1, keepdim=True
                        )
                        text_features = text_features / text_features.norm(
                            dim=-1, keepdim=True
                        )
                        logit_scale = self.config.model.logit_scale.exp()
                        logits_per_image = (
                            logit_scale * image_features @ text_features.t()
                        )
                        logits_per_text = logits_per_image.t()
                        if images_contrastive is not None:
                            image_features_contrastive = (
                                self.config.model.encode_image(images_contrastive)
                            )
                            image_features_contrastive = (
                                image_features_contrastive
                                / image_features_contrastive.norm(
                                    dim=-1, keepdim=True
                                )
                            )
                            logits_per_contrastive_image_1 = (
                                logit_scale
                                * image_features
                                @ image_features_contrastive.t()
                            )
                            logits_per_contrastive_image_2 = (
                                logits_per_contrastive_image_1.t()
                            )

                    elif dataset.use_score:
                        logits = self.config.model(images, texts)
                    else:
                        if images_contrastive is not None:
                            image_features = self.config.model.encode_image(images)
                            text_features = self.config.model.encode_text(texts)
                            image_features = image_features / image_features.norm(
                                dim=-1, keepdim=True
                            )
                            text_features = text_features / text_features.norm(
                                dim=-1, keepdim=True
                            )
                            logit_scale = self.config.model.logit_scale.exp()
                            logits_per_image = (
                                logit_scale * image_features @ text_features.t()
                            )
                            logits_per_text = logits_per_image.t()
                            image_features_contrastive = self.config.model.encode_image(
                                images_contrastive
                            )
                            image_features_contrastive = (
                                image_features_contrastive
                                / image_features_contrastive.norm(dim=-1, keepdim=True)
                            )
                            logits_per_contrastive_image_1 = (
                                logit_scale
                                * image_features
                                @ image_features_contrastive.t()
                            )
                            logits_per_contrastive_image_2 = (
                                logits_per_contrastive_image_1.t()
                            )
                        else:
                            logits_per_image, logits_per_text = self.config.model(
                                images, texts
                            )

                    ground_truth = torch.arange(
                        len(images), dtype=torch.long, device=device
                    )

                    if dataset.use_score:
                        scores = scores.to(device)
                        total_loss = loss_mse(logits, scores)
                    elif dataset.use_weight:
                        weights = weights.to(device)
                        loss_img = nn.CrossEntropyLoss(weight=weights)
                        loss_txt = nn.CrossEntropyLoss(weight=weights)
                        total_loss = (
                            loss_img(logits_per_image, ground_truth)
                            + loss_txt(logits_per_text, ground_truth)
                        ) / 2
                    else:
                        if self.config.use_bce_loss:
                            total_loss = (
                                loss_img(logits_per_image, mask_matrix)
                                + loss_txt(logits_per_text, mask_matrix.T)
                            ) / 2
                        else:
                            total_loss = (
                                loss_img(logits_per_image, ground_truth)
                                + loss_txt(logits_per_text, ground_truth)
                            ) / 2

                    if self.config.use_contrastive_image_loss:
                        total_loss += (
                            (
                                loss_img_contrastive(
                                    logits_per_contrastive_image_1, ground_truth
                                )
                                + loss_img_contrastive(
                                    logits_per_contrastive_image_2, ground_truth
                                )
                            )
                            * self.config.contrastive_image_loss_weight
                            / 2
                        )
                    if self.config.use_triplet_image_loss:
                        d1 = torch.abs(image_features - image_features_contrastive)
                        # slide batch of image_features_contrastive
                        d2 = torch.abs(image_features - torch.roll(
                            image_features_contrastive, 1, dims=0
                        ))
                        # hinge loss
                        triplet_loss = torch.mean(
                                torch.max(
                                    torch.tensor(0.0, device=device),
                                    d1 - d2
                                    + self.config.triplet_image_loss_margin,
                                )
                        ) * self.config.triplet_image_loss_weight
                        total_loss += triplet_loss

                    total_loss.backward()

                    if self.config.do_optimize:
                        if device != "cpu":
                            convert_models_to_fp32(self.config.model)
                        if device == "cpu":
                            optimizer.step()
                            schedular.step()
                        else:
                            optimizer.step()
                            schedular.step()
                        if self.config.use_coop_text or self.config.use_coop_vision:
                            coop_optimizer.step()
                            coop_schedular.step()
                        if device != "cpu":
                            convert_weights(self.config.model)

                    # free images, texts
                    images = images.to("cpu")
                    texts = texts.to("cpu")
                    total_loss = total_loss.to("cpu")
                    epoch_loss += total_loss.item()
                    torch.cuda.empty_cache()

                if (epoch + 1) % 1 == 0:
                    self.config.model.eval()
                    if dataset.use_score:
                        val_loss = self.config.evaluate_used_dumped_image_use_score(
                            self.config.model, val_dataset
                        )
                        test_loss = self.config.evaluate_used_dumped_image_use_score(
                            self.config.model, test_dataset
                        )
                        try:
                            val_corr, _ = evaluate_use_dumped_image(
                                self.config.model,
                                val_datasets,
                                tmp_inclusive_attributes,
                                use_dense_clip=True,
                            )
                        except:
                            print("val_corr error")
                            val_corr = 0
                        try:
                            test_corr, _ = evaluate_use_dumped_image(
                                self.config.model,
                                test_datasets,
                                tmp_inclusive_attributes,
                                use_dense_clip=True,
                            )
                        except:
                            print("test_corr error")
                            test_corr = 0
                        print(
                            f"EPOCH: {epoch+1}, loss: {epoch_loss}, val_loss: {val_loss}, test_loss: {test_loss}, val_corr: {val_corr}, test_corr: {test_corr}"
                        )
                    elif self.config.task_for_validation:
                        val_attr_cr = evaluate_attribute_comparison_task_for_each_comparison(
                            tmp_validation_font_names,
                            self.config.vision_text,
                            model=self.config.model,
                            image_file_dir=self.config.image_file_dir_for_validation,
                            inclusive_attributes=tmp_inclusive_attributes,
                        )
                        test_attr_cr = 0
                        test_attr_cr = evaluate_attribute_comparison_task_for_each_comparison(
                            tmp_test_font_names,
                            self.config.vision_text,
                            model=self.config.model,
                            image_file_dir=self.config.image_file_dir_for_validation,
                            inclusive_attributes=tmp_inclusive_attributes,
                        )

                        val_sim_cr = 0
                        test_sim_cr = 0
                        val_sim_cr = evaluate_similarity_comparison_task(
                            tmp_validation_font_names,
                            self.config.vision_text,
                            model=self.config.model,
                            image_file_dir=self.config.image_file_dir_for_validation,
                        )
                        test_sim_cr = evaluate_similarity_comparison_task(
                            tmp_test_font_names,
                            self.config.vision_text,
                            model=self.config.model,
                            image_file_dir=self.config.image_file_dir_for_validation,
                        )

                        val_corr = evaluate_correlation_coefficient(
                            self.config.model,
                            tmp_validation_json_path,
                            val_image_dataset,
                            text_dataset,
                        )
                        test_corr = evaluate_correlation_coefficient(
                            self.config.model,
                            tmp_test_json_path,
                            test_image_dataset,
                            text_dataset,
                        )
                        val_loss = val_corr
                        test_loss = test_corr
                    elif self.config.use_fast_evaluator:
                        val_corr = evaluate_correlation_coefficient(
                            self.config.model,
                            tmp_validation_json_path,
                            val_image_dataset,
                            text_dataset,
                        )
                        # test_corr = evaluate_correlation_coefficient(
                        #     self.config.model,
                        #     tmp_test_json_path,
                        #     test_image_dataset,
                        #     text_dataset,
                        # )
                        test_corr = 0
                        val_loss = val_corr
                        test_loss = test_corr

                    else:
                        try:
                            val_loss, _, val_variance = evaluate_use_dumped_image(
                                self.config.model,
                                val_datasets,
                                tmp_inclusive_attributes,
                                return_variance=True,
                            )
                            test_loss = 0
                            test_loss, _, test_variance = evaluate_use_dumped_image(
                                self.config.model,
                                test_datasets,
                                tmp_inclusive_attributes,
                                return_variance=True,
                            )
                        except Exception as e:
                            print(e)
                            val_loss = 0
                            val_variance = 0
                            test_loss = 0

                    # if one_leave_out:
                    if self.config.task_for_validation:
                        print(
                            f"EPOCH: {epoch+1}, loss: {epoch_loss}, val_attr_cr: {val_attr_cr}, val_sim_cr: {val_sim_cr}, val_corr: {val_corr}, test_attr_cr: {test_attr_cr}, test_sim_cr: {test_sim_cr}, test_corr: {test_corr}"
                        )
                        if not self.config.do_optimize:
                            return {
                                "loss": epoch_loss,
                                "val_attr_cr": val_attr_cr,
                                "val_sim_cr": val_sim_cr,
                                "val_corr": val_corr,
                                "test_attr_cr": test_attr_cr,
                                "test_sim_cr": test_sim_cr,
                                "test_corr": test_corr,
                            }
                    else:
                        print(
                            f"EPOCH: {epoch+1}, loss: {epoch_loss}, val_loss: {val_loss}, test_loss: {test_loss}"
                        )
                    if best_val_loss is None or val_loss > best_val_loss:
                        best_val_loss = val_loss
                        best_epoch = epoch + 1
                        if self.config.do_optimize:
                            print(f"Saving model...{best_epoch}")
                            torch.save(
                                {
                                    "epoch": epoch,
                                    "model_state_dict": self.config.model.state_dict(),
                                    "optimizer_state_dict": optimizer.state_dict(),
                                    "loss": epoch_loss,
                                },
                                f"model_checkpoints/{self.config.signature}.pt",
                            )
                            print(
                                f"Saved model...{best_epoch}, {self.config.signature}"
                            )

                    if not self.config.do_optimize:
                        break
                    self.config.model.train()
                    # if val_loss >= 1.7:
                    # break

            # rename model_path
            if self.config.one_leave_out:
                model_path = f"model_checkpoints/model_{best_epoch}.pt"
                best_checkpoint_path = retrieve_one_leave_out_model_path(
                    one_left_out_font_name
                )
                os.rename(model_path, best_checkpoint_path)

            if self.config.leave_out_attributes:
                model_path = f"model_checkpoints/{self.config.signature}.pt"
                best_checkpoint_path = f"model_checkpoints/leave_one_out_{self.config.leave_out_attributes[i]}_{self.config.signature}.pt"
                os.rename(model_path, best_checkpoint_path)

            dataset.do_apotosis()
            if val_datasets is not None:
                for val_dataset in val_datasets:
                    val_dataset.do_apotosis()
            if test_datasets is not None:
                for test_dataset in test_datasets:
                    test_dataset.do_apotosis()

            del dataset
            del val_datasets
            del test_datasets