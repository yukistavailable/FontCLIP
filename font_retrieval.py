import argparse
import os

import gradio as gr
import matplotlib.font_manager as font_manager
import numpy as np
import torch
from PIL import Image, ImageFont
from PIL.Image import Image as PILImage
from typing import Optional

from models.init_model import device, load_model, my_preprocess, preprocess
from models.lora import LoRAConfig
from utils.initialize_font_data import (
    fox_text_four_lines,
)
from utils.tokenizer import tokenize
from utils.transform_image import (
    draw_text_with_new_lines,
    generate_all_fonts_embedded_images,
    my_transform,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_file_dir", type=str, default="gwfonts_images/")
    parser.add_argument("--aug_num", type=int, default=8)
    parser.add_argument("--column_num", type=int, default=1)
    parser.add_argument("--row_num", type=int, default=5)
    parser.add_argument("--font_dir", type=str, default="gwfonts/")
    parser.add_argument("--cached_font_db_path", type=str, default=None)
    args = parser.parse_args()
    # set your checkpoint path
    checkpoint_path = "model_checkpoints/model.pt"
    # Load the model
    font_dir = args.font_dir
    image_file_dir = args.image_file_dir
    char_size = 150
    aug_num = args.aug_num
    cached_font_db_path = args.cached_font_db_path
    text = fox_text_four_lines

    # add font
    for font in font_manager.findSystemFonts(font_dir):
        font_manager.fontManager.addfont(font)
    all_gwfont_paths = sorted(
        [
            os.path.join(font_dir, file_name)
            for file_name in os.listdir(font_dir)
            if file_name not in [".DS_Store"]
        ]
    )
    target_font_paths = all_gwfont_paths

    preprocess_for_aug = my_transform(lower_bound_of_scale=0.3)

    # LoRA
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
        checkpoint_path,
        model_name="ViT-B/32",
        use_oft_vision=False,
        use_oft_text=False,
        oft_config_vision=None,
        oft_config_text=None,
        use_lora_text=True,
        use_lora_vision=False,
        lora_config_vision=None,
        lora_config_text=lora_config_text,
        use_coop_text=False,
        use_coop_vision=False,
        precontext_length_vision=10,
        precontext_length_text=77,
        precontext_dropout_rate=0,
        pt_applied_layers=None,
    )
    print("Prepare font_db...")
    font_db = None
    if cached_font_db_path is not None and os.path.exists(cached_font_db_path):
        print("Loading font_db...")
        with open(cached_font_db_path, "rb") as f:
            font_db = np.load(f)
    else:
        print("There is no font_db, generating font_db...")
        font_db_path = f"output/font_db_aug{aug_num}.npy"
        embedded_images = generate_all_fonts_embedded_images(
            target_font_paths,
            text,
            image_file_dir=image_file_dir,
            model=model,
            preprocess=preprocess_for_aug,
            aug_num=aug_num,
        )
        embedded_images_numpy = torch.cat(list(embedded_images.values())).cpu().numpy()
        font_db = embedded_images_numpy
        # save font_db
        if not os.path.exists(os.path.dirname(font_db_path)):
            os.makedirs(os.path.dirname(font_db_path))
        with open(font_db_path, "wb") as f:
            np.save(f, font_db)
    
    def create_image_from_font_path(font_path: str, sample_text: Optional[str]) -> PILImage:
        if font_path.endswith(".png") or font_path.endswith(".jpg"):
            image = Image.open(font_path)
            return image
        font = ImageFont.truetype(font_path, char_size)
        image = create_image(sample_text, font)
        return image

    def create_image(text, font, char_size=char_size):
        line_num = text.count("\n") + 1
        width = int(char_size * len(text) * 1.8 / line_num)
        height = int(char_size * 1.5) * line_num
        image = draw_text_with_new_lines(text, font, width, height)
        return image

    def calc_cos_sim(a, b):
        dot_product = np.dot(b, a.T)
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b, axis=1)
        cos_sim = dot_product / np.outer(b_norm, a_norm)
        # sim = dot_product / (b_norm[:, np.newaxis] * a_norm)
        return cos_sim[:, 0]

    def query_text(text, font_db, model):
        model.eval()
        input_texts = text
        if isinstance(text, str):
            input_texts = [text]
        tokenized_text = tokenize(input_texts).to(device)
        embedded_text = model.encode_text(tokenized_text).cpu().numpy()
        # retrieve most similar font
        cos_sim = calc_cos_sim(embedded_text, font_db)
        sorted_index = np.argsort(-cos_sim)
        return sorted_index, cos_sim

    def query_image(image, font_db, model, preprocess=preprocess, aug_num=64):
        model.eval()
        if isinstance(image, str):
            image = Image.open(image)

        if aug_num == 1:
            preprocessed_image = preprocess(image).unsqueeze(0).to(device)
            embedded_image = model.encode_image(preprocessed_image).cpu().numpy()
        else:
            preprocessed_images = torch.cat(
                [preprocess(image).unsqueeze(0).to(device) for _ in range(aug_num)]
            )
            embedded_images = model.encode_image(preprocessed_images).cpu()
            embedded_image = (
                torch.mean(embedded_images, axis=0).unsqueeze(0).cpu().numpy()
            )
        # retrieve most similar font
        cos_sim = calc_cos_sim(embedded_image, font_db)
        sorted_index = np.argsort(-cos_sim)
        return sorted_index, cos_sim

    def query_image_and_text(
        image,
        text,
        alpha=0.5,
        font_db=font_db,
        model=model,
        preprocess=preprocess,
        aug_num=64,
    ):
        model.eval()
        input_texts = text
        if isinstance(text, str):
            input_texts = [text]
        tokenized_text = tokenize(input_texts).to(device)
        embedded_text = model.encode_text(tokenized_text).cpu().numpy()
        if aug_num == 1:
            preprocessed_image = preprocess(image).unsqueeze(0).to(device)
            embedded_image = model.encode_image(preprocessed_image).cpu().numpy()
        else:
            preprocessed_images = torch.cat(
                [preprocess(image).unsqueeze(0).to(device) for _ in range(aug_num)]
            )
            embedded_images = model.encode_image(preprocessed_images).cpu()
            embedded_image = (
                torch.mean(embedded_images, axis=0).unsqueeze(0).cpu().numpy()
            )
        sum_embedded = alpha * embedded_image + (1 - alpha) * embedded_text
        cos_sim = calc_cos_sim(sum_embedded, font_db)
        sorted_index = np.argsort(-cos_sim)
        return sorted_index, cos_sim

    def save_output_buidler(*images):
        for i, image in enumerate(images):
            if image is None:
                continue
            image.save(f"output_images/{i}.png")

    default_text_value = "Eurographics"
    sorted_index = None
    current_index = 0
    column_num = args.column_num
    row_num = args.row_num

    def builder_query(
        prompt,
        image,
        alpha=1.0,
        sample_text="hand write",
        char_size=char_size,
        column_num=column_num,
        row_num=row_num,
    ):
        global sorted_index
        global current_index
        current_index = 0

        is_prompt = True
        is_image = True
        if prompt is None or prompt.replace(" ", "") == "":
            is_prompt = False
        if image is None:
            is_image = False

        if not prompt.endswith("font"):
            prompt += " font"

        if (not is_prompt) and (not is_image):
            return [None] * (column_num * row_num)

        result_images = []
        if is_prompt and is_image:
            sorted_index, _ = query_image_and_text(
                image,
                prompt,
                alpha,
                font_db=font_db,
                model=model,
                aug_num=32,
                preprocess=preprocess_for_aug,
            )
            for i in range(column_num):
                for j in range(row_num):
                    font_path = target_font_paths[sorted_index[i * row_num + j]]
                    print(font_path)
                    image = create_image_from_font_path(font_path, sample_text)
                    result_images.append(image)

            return result_images

        if is_prompt and (not is_image):
            sorted_index, _ = query_text(prompt, font_db, model)
            for i in range(column_num):
                for j in range(row_num):
                    font_path = target_font_paths[sorted_index[i * row_num + j]]
                    print(font_path)
                    image = create_image_from_font_path(font_path, sample_text)
                    result_images.append(image)
            return result_images

        if (not is_prompt) and is_image:
            sorted_index, _ = query_image(
                image, font_db, model, aug_num=32, preprocess=my_preprocess
            )
            for i in range(column_num):
                for j in range(row_num):
                    font_path = target_font_paths[sorted_index[i * row_num + j]]
                    print(font_path)
                    image = create_image_from_font_path(font_path, sample_text)
                    result_images.append(image)
            return result_images

        return [None] * (column_num * row_num)

    def builder_next_query(
        command="next",
        sample_text="hand write",
        char_size=char_size,
        column_num=column_num,
        row_num=row_num,
    ):
        assert command in ["next", "previous"]
        global current_index
        if command == "next":
            current_index += 1
            if current_index * (column_num * row_num) >= len(sorted_index):
                current_index = 0
        else:
            current_index -= 1
            if current_index < 0:
                current_index = len(sorted_index) // (column_num * row_num) - 1

        result_images = []
        for i in range(column_num):
            for j in range(row_num):
                font_path = target_font_paths[
                    sorted_index[current_index + i * row_num + j]
                ]
                font = ImageFont.truetype(font_path, char_size)
                image = create_image(sample_text, font)
                result_images.append(image)
        return result_images

    def builder_next_query_slider(
        slider_value,
        sample_text="hand write",
        char_size=char_size,
        column_num=column_num,
        row_num=row_num,
    ):
        if sorted_index is None:
            return [None] * (column_num * row_num)

        assert 0 <= slider_value <= len(target_font_paths) - 1
        global current_index
        current_index = int(slider_value)

        result_images = []
        for i in range(column_num):
            for j in range(row_num):
                font_path = target_font_paths[
                    sorted_index[current_index + i * row_num + j]
                ]
                font = ImageFont.truetype(font_path, char_size)
                image = create_image(sample_text, font)
                result_images.append(image)
        return result_images

    css = """
    .input textarea {font-size: 50px !important}
    """

    gr_images = []
    with gr.Blocks(css=css) as demo:
        with gr.Row():
            with gr.Column(scale=2):
                text2 = gr.Text(
                    label="Type here to preview text",
                    value=default_text_value,
                    interactive=True,
                    visible=False,
                )
                text1 = gr.Text(
                    label="Text Prompt", interactive=True, elem_classes="input"
                )
                slider1 = gr.Slider(
                    0,
                    1.0,
                    value=0.5,
                    step=0.01,
                    label="balance between text prompt and image queries (0: only attribute, 1: only image)",
                    interactive=True,
                    visible=False,
                )
                image1 = gr.Image(label="Image query", type="pil", interactive=True)
                button1 = gr.Button(value="Refresh", visible=True)
            with gr.Column(scale=3):
                with gr.Row():
                    for i in range(column_num):
                        with gr.Column():
                            for j in range(row_num):
                                image = gr.Image(
                                    label=f"best {j+i*row_num}",
                                    type="pil",
                                    interactive=True,
                                    show_label=False,
                                )
                                gr_images.append(image)

                """
                with gr.Row():
                    button4 = gr.Button(value='previous', interactive=True)
                    button3 = gr.Button(value='next', interactive=True)
                """
                slider2 = gr.Slider(
                    0,
                    len(target_font_paths) - 1,
                    value=0,
                    step=1,
                    label="Browse more fonts",
                    interactive=True,
                    visible=False,
                )
                button2 = gr.Button(
                    value="save outputs", interactive=True, visible=False
                )

        slider1.change(
            fn=builder_query,
            inputs=[text1, image1, slider1, text2],
            outputs=gr_images,
            show_progress=False,
        )
        text1.change(
            fn=builder_query,
            inputs=[text1, image1, slider1, text2],
            outputs=gr_images,
            show_progress=False,
        )
        image1.change(
            fn=builder_query,
            inputs=[text1, image1, slider1, text2],
            outputs=gr_images,
            show_progress=False,
        )
        text2.change(
            fn=builder_query,
            inputs=[text1, image1, slider1, text2],
            outputs=gr_images,
            show_progress=False,
        )
        button1.click(
            fn=builder_query,
            inputs=[text1, image1, slider1, text2],
            outputs=gr_images,
            show_progress=False,
        )
        button2.click(fn=save_output_buidler, inputs=gr_images, show_progress=False)
        slider2.change(
            fn=builder_next_query_slider,
            inputs=[slider2, text2],
            outputs=gr_images,
            show_progress=False,
        )
        # button3.click(fn=builder_next_query, inputs=[button3, text2], outputs=[image2, image3], show_progress=False)
        # button4.click(fn=builder_next_query, inputs=[button4, text2], outputs=[image2, image3], show_progress=False)

    demo.launch(debug=True, share=True)
