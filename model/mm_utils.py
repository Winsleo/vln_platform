from PIL import Image
from io import BytesIO
import base64
import math
import ast
import re
import torch
from transformers import StoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX


def resize_and_center_crop(image, shortest_edge_length):
    """
    Resize and center crop an image to a square with side length of shortest_edge_length.

    Args:
        image (PIL.Image.Image): The input image.
        shortest_edge_length (int): The target shortest edge length.

    Returns:
        PIL.Image.Image: The resized and cropped image.
    """
    # Calculate new dimensions and resize
    aspect_ratio = float(image.width) / float(image.height)
    if aspect_ratio > 1:
        new_width = int(shortest_edge_length * aspect_ratio)
        new_height = shortest_edge_length
    else:
        new_width = shortest_edge_length
        new_height = int(shortest_edge_length / aspect_ratio)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    # Calculate the position and perform the center crop
    left = (new_width - shortest_edge_length) / 2
    top = (new_height - shortest_edge_length) / 2
    right = (new_width + shortest_edge_length) / 2
    bottom = (new_height + shortest_edge_length) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))

    return cropped_image


def center_pad_after_resize(image, new_width, new_height, target_width, target_height, resample=Image.LANCZOS, fill=(0, 0, 0)):
    """
    Resize to (resize_width, resize_height), then center-pad into (target_width, target_height).

    Note: resample is forwarded to PIL.Image.resize. If None, PIL default is used (keeps original behavior).
    """
    resized_image = image.resize((new_width, new_height), resample)

    canvas = Image.new("RGB", (target_width, target_height), fill)
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    canvas.paste(resized_image, (paste_x, paste_y))
    return canvas


def compute_fit_size_keep_ar(original_width, original_height, target_width, target_height):
    """
    Compute new size that fits inside target while preserving aspect ratio, by min(scale_w, scale_h).
    """
    scale_w = target_width / original_width
    scale_h = target_height / original_height
    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)
    return new_width, new_height


def choose_target_resolution(input_width, input_height, grid_params):
    """
    Choose a target resolution from grid_params that best matches the input aspect ratio
    and magnitude (closest max dimension), replicating auto_pad_images' selection.
    Returns (target_width, target_height).
    """
    input_aspect_ratio = input_width / input_height
    candidate_resolutions = [(w / h, w, h) for w in grid_params for h in grid_params]
    closest_aspect_ratio = min(candidate_resolutions, key=lambda x: abs(input_aspect_ratio - x[0]))
    candidate_resolutions = [(x[1], x[2]) for x in candidate_resolutions if abs(x[0] - closest_aspect_ratio[0]) < 1e-3]
    target_resolution = min(candidate_resolutions, key=lambda res: abs(max(input_width, input_height) / max(res) - 1))
    return target_resolution


def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        # Calculate the downscaled size to keep the aspect ratio
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)

        # Calculate effective and wasted resolutions
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def preprocess_to_tensor_list(images, processor):
    """
    Preprocess a list of images to a list of tensors.

    Args:
        images (list): A list of PIL.Image.Image objects.
        processor (Processor): The image processor.
    """
    return [processor.preprocess(img, return_tensors="pt")["pixel_values"][0] for img in images]


def preprocess_and_stack(images, processor):
    tensors = preprocess_to_tensor_list(images, processor)
    return torch.stack(tensors, dim=0)


def stack_list_if_same_shape(tensors):
    if len(tensors) > 0 and all(x.shape == tensors[0].shape for x in tensors):
        return torch.stack(tensors, dim=0)
    return tensors


def get_patch_size_from_processor(processor):
    """
    Best-effort to extract patch size from a processor, compatible with dict/sequence forms.
    Falls back to crop_size["height"].
    """
    try:
        # Some processors expose size as a sequence
        return processor.size[0]
    except Exception:
        try:
            # Dict form with shortest_edge
            return processor.size["shortest_edge"]
        except Exception:
            # Fallback to crop_size height
            return processor.crop_size["height"]


def grid_pinpoints_to_resolutions(grid_pinpoints, patch_size):
    """
    Normalize grid_pinpoints into a list of (width, height) pixel resolutions.
    Supports string interval form like "(1x1)-(2x3)" by expanding to all multiples of patch_size.
    """
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        pairs = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
        return [(i * patch_size, j * patch_size) for (i, j) in pairs]
    if type(grid_pinpoints) is list:
        return [tuple(x) for x in grid_pinpoints]
    # literal-eval string form
    return [tuple(x) for x in ast.literal_eval(grid_pinpoints)]


def auto_pad_images(image, grid_params):
    """
    Pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        grid_params (list): The candidate image height or width.

    Returns:
        PIL.Image.Image: The padded image.
    """
    assert isinstance(image, Image.Image), "Input should be a Pillow Image"
    assert len(grid_params) > 0, "Grid parameters should not be empty"

    # Step 1: Calculate and find the closest aspect ratio
    input_width, input_height = image.size
    input_aspect_ratio = input_width / input_height
    target_resolution = choose_target_resolution(input_width, input_height, grid_params)

    resize_width, resize_height = target_resolution
    if input_width > input_height:
        resize_height = int(resize_width / input_aspect_ratio)
    else:
        resize_width = int(resize_height * input_aspect_ratio)

    return center_pad_after_resize(
        image,
        resize_width,
        resize_height,
        target_resolution[0],
        target_resolution[1],
        resample=Image.LANCZOS,
        fill=(0, 0, 0),
    )


def extract_patches(image, patch_size, overlap_ratio=0, align="center"):
    """
    Extract patches from an image.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.
        overlap_ratio (float): The overlap ratio in [0,1).
        align (str): Grid alignment strategy: 'center' (default) or 'top_left'.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    assert isinstance(image, Image.Image), "Input should be a Pillow Image"
    assert patch_size > 0, "Patch size should be greater than 0"
    assert 0 <= overlap_ratio < 1, "Overlap ratio should be between 0 and 1"
    assert align in ("center", "top_left"), "align must be 'center' or 'top_left'"

    W, H = image.size
    patches = []

    stride = int(patch_size * (1 - overlap_ratio))
    stride = max(1, stride)

    if align == "center":
        num_patches_y = (H - patch_size) // stride + 1
        num_patches_x = (W - patch_size) // stride + 1
        y_start = (H - (num_patches_y - 1) * stride - patch_size) // 2
        x_start = (W - (num_patches_x - 1) * stride - patch_size) // 2
        for y in range(y_start, y_start + num_patches_y * stride, stride):
            for x in range(x_start, x_start + num_patches_x * stride, stride):
                patch = image.crop((x, y, x + patch_size, y + patch_size))
                patches.append(patch)
    else:  # top_left
        for y in range(0, max(0, H - patch_size + 1), stride):
            for x in range(0, max(0, W - patch_size + 1), stride):
                patch = image.crop((x, y, x + patch_size, y + patch_size))
                patches.append(patch)

    return patches


def process_highres_image_crop_split(image, processor, crop_resolution, split_resolution):
    image_crop = resize_and_center_crop(image, crop_resolution)
    image_patches = extract_patches(image_crop, patch_size=split_resolution, overlap_ratio=0)
    return preprocess_and_stack(image_patches, processor)


def process_highres_image(image, processor, grid_pinpoints):
    grid_params = [int(x) for x in grid_pinpoints.split(",")]
    width_height = max(image.size)
    fit_grid_params = [x for x in grid_params if x >= width_height]
    if len(fit_grid_params) == 0:
        select_size = max(grid_params)
    else:
        select_size = min(fit_grid_params)
    # FIXME: always select the 448
    select_size = max(grid_params)
    image_padded = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))

    # FIXME: this seems to be a bug that it always resizes instead of padding
    image_original_resize = image.resize((processor.size["shortest_edge"], processor.size["shortest_edge"]))
    image_padded = image_padded.resize((select_size, select_size))
    image_patches = extract_patches(image_padded, patch_size=processor.size["shortest_edge"], overlap_ratio=0)
    image_patches = [image_original_resize] + image_patches
    return preprocess_and_stack(image_patches, processor)


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    # Unified interpolation: LANCZOS
    new_width, new_height = compute_fit_size_keep_ar(original_width, original_height, target_width, target_height)
    return center_pad_after_resize(
        image,
        new_width,
        new_height,
        target_width,
        target_height,
        resample=Image.LANCZOS,
        fill=(0, 0, 0),
    )


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    possible_resolutions = grid_pinpoints_to_resolutions(grid_pinpoints, patch_size)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def process_anyres_image(image, processor, grid_pinpoints):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    # Convert grid_pinpoints from string to list
    patch_size = get_patch_size_from_processor(processor)
    possible_resolutions = grid_pinpoints_to_resolutions(grid_pinpoints, patch_size)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    patches = extract_patches(image_padded, processor.crop_size["height"], overlap_ratio=0, align="top_left")

    # FIXME: this seems to be a bug that it resizes instead of pad.
    # but to keep it consistent with previous, i will keep it as it is
    # TODO: uncomment below to ablate with the padding
    if isinstance(processor.size, dict):
        shortest_edge = processor.size["shortest_edge"]
    else:
        shortest_edge = min(processor.size)
    image_original_resize = image.resize((shortest_edge, shortest_edge))
    # image_padded_square = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
    # image_original_resize = image_padded_square.resize((processor.size['shortest_edge'], processor.size['shortest_edge']))

    image_patches = [image_original_resize] + patches
    return preprocess_and_stack(image_patches, processor)


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == "highres":
        for image in images:
            image = process_highres_image(image, image_processor, model_cfg.image_grid_pinpoints)
            new_images.append(image)
    elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
        for image in images:
            image = process_anyres_image(image, image_processor, model_cfg.image_grid_pinpoints)
            new_images.append(image)
    elif image_aspect_ratio == "crop_split":
        for image in images:
            crop_resolution = model_cfg.image_crop_resolution
            split_resolution = model_cfg.image_split_resolution
            image = process_highres_image_crop_split(image, image_processor, crop_resolution, split_resolution)
            new_images.append(image)
    elif image_aspect_ratio == "pad":
        for image in images:
            image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            new_images.append(image)
    else:
        return image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
    new_images = stack_list_if_same_shape(new_images)
    return new_images


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if output_ids[0, -keyword_id.shape[0] :] == keyword_id:
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
