import numpy as np
import torch
import torch.nn.functional as F
import kornia as K
from CLIP.tokenizer import tokenize
# from pytorch_grad_cam.utils.image import show_cam_on_image
# import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def encode_text_with_prompt_ensemble_equall(model, obj, device):
    prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect',
                     '{} without damage']
    prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage', ' disease {}',
                       'abnormal {}']
    prompt_state = [prompt_normal, prompt_abnormal]
    prompt_templates = ['a bad photo of a {}.', 'a low resolution photo of the {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a bright photo of a {}.', 'a dark photo of the {}.', 'a photo of my {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a photo of one {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'a low resolution photo of a {}.', 'a photo of a large {}.', 'a blurry photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a photo of the small {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'a dark photo of a {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.']
    text_features = []
    with torch.no_grad():
        for i in range(len(prompt_state)):
            prompted_state = [state.format(obj) for state in prompt_state[i]]
            prompted_sentence = []
            for s in prompted_state:
                for template in prompt_templates:
                    prompted_sentence.append(template.format(s))
            prompted_sentence = tokenize(prompted_sentence).to(device)
            class_embeddings = model.encode_text(prompted_sentence)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            text_features.append(class_embedding)

    text_features = torch.stack(text_features, dim=1).to(device)
    return text_features

def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])

def get_translation_mat(a, b):
    return torch.tensor([[1, 0, a],
                         [0, 1, b]])

def rot_img(x, theta):
    dtype =  torch.FloatTensor
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid, padding_mode="reflection")
    return x

def translation_img(x, a, b):
    dtype =  torch.FloatTensor
    rot_mat = get_translation_mat(a, b)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid, padding_mode="reflection")
    return x

def hflip_img(x):
    x = K.geometry.transform.hflip(x)
    return x

def vflip_img(x):
    x = K.geometry.transform.vflip(x)
    return x

def zoom_img(x, scale_factor):
    """
    Zooms in on an image tensor.
    :param x: Input image tensor of shape (B, C, H, W)
    :param scale_factor: Scaling factor for zoom. Values >1 zoom in, values <1 zoom out.
    :return: Zoomed image tensor.
    """
    _, _, h, w = x.shape
    new_h, new_w = int(h / scale_factor), int(w / scale_factor)
    cropped = torch.nn.functional.interpolate(
        x[:, :, (h - new_h) // 2 : (h + new_h) // 2, (w - new_w) // 2 : (w + new_w) // 2],
        size=(h, w),
        mode="bilinear",
        align_corners=False
    )
    return cropped




def color_jittering(x, brightness=0.2, contrast=0.2, saturation=0.2):
    """
    Apply color jittering in PyTorch

    Args:
        x (torch.Tensor): Input image tensor of shape [C, H, W]
        brightness (float): Brightness jittering factor
        contrast (float): Contrast jittering factor
        saturation (float): Saturation jittering factor

    Returns:
        torch.Tensor: Augmented image tensor
    """
    # Brightness augmentation
    brightness_factor = 1.0 + torch.rand(1).item() * brightness * 2 - brightness
    x = x * brightness_factor

    # Contrast augmentation
    contrast_factor = 1.0 + torch.rand(1).item() * contrast * 2 - contrast
    mean = x.mean(dim=[1, 2], keepdim=True)
    x = (x - mean) * contrast_factor + mean

    # Clip values to valid range
    x = torch.clamp(x, 0, 1)

    return x

def augment(fewshot_img, fewshot_mask=None):

    augment_fewshot_img = fewshot_img

    if fewshot_mask is not None:
        augment_fewshot_mask = fewshot_mask

        # rotate img
        for angle in [-np.pi/8, -np.pi/16, np.pi/16, np.pi/8]:
            rotate_img = rot_img(fewshot_img, angle)
            augment_fewshot_img = torch.cat([augment_fewshot_img, rotate_img], dim=0)

            rotate_mask = rot_img(fewshot_mask, angle)
            augment_fewshot_mask = torch.cat([augment_fewshot_mask, rotate_mask], dim=0)
        # translate img
        for a,b in [(0.1,0.1), (-0.1,0.1), (-0.1,-0.1), (0.1,-0.1)]:
            trans_img = translation_img(fewshot_img, a, b)
            augment_fewshot_img = torch.cat([augment_fewshot_img, trans_img], dim=0)

            trans_mask = translation_img(fewshot_mask, a, b)
            augment_fewshot_mask = torch.cat([augment_fewshot_mask, trans_mask], dim=0)

        for scale_factor in [1.1, 1.15, 1.2, 1.3]:  # Slight zoom-in
            zoomed_img = zoom_img(fewshot_img, scale_factor)
            augment_fewshot_img = torch.cat([augment_fewshot_img, zoomed_img], dim=0)

            zoomed_mask = zoom_img(fewshot_mask, scale_factor)
            augment_fewshot_mask = torch.cat([augment_fewshot_mask, zoomed_mask], dim=0)



        # color jittering
        jittered_img = color_jittering(fewshot_img)
        augment_fewshot_img = torch.cat([augment_fewshot_img, jittered_img], dim=0)
        augment_fewshot_mask = torch.cat([augment_fewshot_mask, fewshot_mask], dim=0)


        # hflip img
        flipped_img = hflip_img(fewshot_img)
        augment_fewshot_img = torch.cat([augment_fewshot_img, flipped_img], dim=0)
        flipped_mask = hflip_img(fewshot_mask)
        augment_fewshot_mask = torch.cat([augment_fewshot_mask, flipped_mask], dim=0)

        # vflip img
        flipped_img = vflip_img(fewshot_img)
        augment_fewshot_img = torch.cat([augment_fewshot_img, flipped_img], dim=0)

        flipped_mask = vflip_img(fewshot_mask)
        augment_fewshot_mask = torch.cat([augment_fewshot_mask, flipped_mask], dim=0)

    else:
        # rotate img
        for angle in [-np.pi/8, -np.pi/16, np.pi/16, np.pi/8]:
            rotate_img = rot_img(fewshot_img, angle)
            augment_fewshot_img = torch.cat([augment_fewshot_img, rotate_img], dim=0)

        # translate img
        for a,b in [(0.1,0.1), (-0.1,0.1), (-0.1,-0.1), (0.1,-0.1)]:
            trans_img = translation_img(fewshot_img, a, b)
            augment_fewshot_img = torch.cat([augment_fewshot_img, trans_img], dim=0)

        for scale_factor in [1.1, 1.15, 1.2, 1.3]:  # Slight zoom-in
            zoomed_img = zoom_img(fewshot_img, scale_factor)
            augment_fewshot_img = torch.cat([augment_fewshot_img, zoomed_img], dim=0)

        # for scale_factor in [0.9, 0.95, 1.05, 1.1]:  # Slight adjustments to intensity
        #     scaled_img = scale_intensity(fewshot_img, scale_factor)
        #     augment_fewshot_img = torch.cat([augment_fewshot_img, scaled_img], dim=0)


        # color jittering
        jittered_img = color_jittering(fewshot_img)
        augment_fewshot_img = torch.cat([augment_fewshot_img, jittered_img], dim=0)

        # hflip img
        flipped_img = hflip_img(fewshot_img)
        augment_fewshot_img = torch.cat([augment_fewshot_img, flipped_img], dim=0)

        # vflip img
        flipped_img = vflip_img(fewshot_img)
        augment_fewshot_img = torch.cat([augment_fewshot_img, flipped_img], dim=0)

        B, _, H, W = augment_fewshot_img.shape
        augment_fewshot_mask = torch.zeros([B, 1, H, W])
    
    return augment_fewshot_img, augment_fewshot_mask