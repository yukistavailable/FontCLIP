from copy import copy
from dataclasses import dataclass
from logging import getLogger
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

from models.init_model import (
    device,
    preprocess,
    convert_weights,
)
from utils.initialize_font_data import (
    font_dir,
    validation_json_path,
    inclusive_attributes,
    exclusive_attributes,
)
from evals.evaluate_tools import (
    evaluate_correlation_coefficient,
)
from dataset.dataset import (
    MyDataset,
    set_image_tensors,
    TestImageDataset,
    TestTextDataset,
)
from trains.config_fine_tune import Config

@dataclass
class Trainer:
    config: Config = None

    def __post_init__(self):
        pass

    def train(self):
        for i in range(
            self.config.start_index_for_train_model, self.config.trained_model_num
        ):
            tmp_exclusive_attributes = copy(exclusive_attributes)
            tmp_inclusive_attributes = copy(inclusive_attributes)
            tmp_validation_json_path = validation_json_path
            val_texts_for_font_image = self.config.val_texts_for_font_image
            print("val_texts_for_font_image", val_texts_for_font_image)
            if self.config.use_single_character:
                self.config.image_file_dir_for_validation = None

            print(self.config.checkpoint_path)

            print("Use CoOp", self.config.use_coop_text)
            print("Use VPT", self.config.use_coop_vision)
            if self.config.use_unpretrained_model:
                print("Warning: use unpretrained model")
                self.config.model.initialize_parameters()
            print(self.config.signature)

            dataset = MyDataset(
                font_dir,
                self.config.json_path,
                texts_for_font_image=self.config.texts_for_font_image,
                use_negative=self.config.use_negative,
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
            val_image_dataset = TestImageDataset(
                font_dir,
                tmp_validation_json_path,
                val_texts_for_font_image[0],
                dump_image=self.config.tmp_dump_image,
                image_file_dir=self.config.image_file_dir_for_validation,
                preprocess=preprocess,
            )
            text_dataset = TestTextDataset(
                target_attributes=tmp_inclusive_attributes,
                context_length=self.config.context_length,
            )


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

            # freeze the model first
            for param in self.config.model.parameters():
                param.requires_grad = False

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

            if self.config.use_bce_loss:
                loss_img = nn.BCEWithLogitsLoss()
                loss_txt = nn.BCEWithLogitsLoss()
            else:
                loss_img = nn.CrossEntropyLoss()
                loss_txt = nn.CrossEntropyLoss()

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

                    if self.config.use_bce_loss:
                        images, texts, font_indices, attribute_indices = batch
                        font_indices = font_indices.to(device)
                        attribute_indices = attribute_indices.to(device)
                        mask_matrix = dataset.mask_font_idx_signed_attribute_matrix_ground_truth_fast(
                            font_indices, attribute_indices
                        )
                        mask_matrix = mask_matrix.to(device).to(torch.float16)
                    else:
                        images, texts = batch

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

                    else:
                        logits_per_image, logits_per_text = self.config.model(
                            images, texts
                        )

                    ground_truth = torch.arange(
                        len(images), dtype=torch.long, device=device
                    )

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
                    val_corr = evaluate_correlation_coefficient(
                        self.config.model,
                        tmp_validation_json_path,
                        val_image_dataset,
                        text_dataset,
                    )
                    val_loss = val_corr

                    print(
                        f"EPOCH: {epoch+1}, loss: {epoch_loss}, val_correlation: {val_loss}"
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