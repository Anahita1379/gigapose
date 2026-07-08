import os
import cv2
import matplotlib
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import transforms
import torch

from torchvision.utils import save_image
from src.libVis.numpy import (
    create_edge_from_mask,
    plot_keypoints,
)

os.environ["MPLCONFIGDIR"] = os.getcwd() + "./tmp/"
np.random.seed(2022)
COLORS_SPACE = np.random.randint(0, 255, size=(1000, 3))
inv_rgb_transform = T.Compose(
    [
        T.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        ),
    ]
)


def convert_cmap(tensor):
    b, h, w = tensor.shape
    ndarr = tensor.to("cpu", torch.uint8).numpy()
    output = torch.zeros((b, 3, h, w), device=tensor.device)
    for i in range(b):
        cmap = matplotlib.cm.get_cmap("magma")
        tmp = cmap(ndarr[i])[..., :3]
        data = transforms.ToTensor()(np.array(tmp)).to(tensor.device)
        output[i] = data
    return output


def save_tensor_to_image(tensor, path, image_size=None, nrow=4):
    if image_size is not None:
        tensor = F.interpolate(tensor, image_size, mode="bilinear", align_corners=False)
    save_image(tensor, path, nrow=nrow)


def add_border_to_tensor(tensor, border_color, border_size=5):
    if border_color == "green":
        border_color = torch.tensor([0, 1, 0], dtype=tensor.dtype, device=tensor.device)
    elif border_color == "red":
        border_color = torch.tensor([1, 0, 0], dtype=tensor.dtype, device=tensor.device)
    else:
        raise NotImplementedError
    # Add the top border
    size = tensor[:, :, :border_size, :].shape
    tensor[:, :, :border_size, :] = border_color.view(1, 3, 1, 1).expand(size)
    # Add the bottom border
    size = tensor[:, :, -border_size:, :].shape
    tensor[:, :, -border_size:, :] = border_color.view(1, 3, 1, 1).expand(size)
    # Add the left border
    size = tensor[:, :, :, :border_size].shape
    tensor[:, :, :, :border_size] = border_color.view(1, 3, 1, 1).expand(size)
    # Add the right border
    size = tensor[:, :, :, -border_size:].shape
    tensor[:, :, :, -border_size:] = border_color.view(1, 3, 1, 1).expand(size)
    return tensor


def put_image_to_grid(list_imgs, adding_margin=True):
    num_col = len(list_imgs)
    b, c, h, w = list_imgs[0].shape
    device = list_imgs[0].device
    if adding_margin:
        num_all_col = num_col + 1
    else:
        num_all_col = num_col
    grid = torch.zeros((b * num_all_col, 3, h, w), device=device).to(list_imgs[0].dtype)
    idx_grid = torch.arange(0, grid.shape[0], num_all_col, device=device).to(
        torch.int64
    )
    for i in range(num_col):
        grid[idx_grid + i] = list_imgs[i].to(list_imgs[0].dtype)
    return grid, num_col + 1


def resize_tensor(tensor, size):
    return F.interpolate(tensor, size, mode="bilinear", align_corners=True)


def convert_tensor_to_image(tensor, type="rgb", unnormalize=True):
    if unnormalize and type == "rgb":
        tensor = inv_rgb_transform(tensor)
    if type == "rgb":
        tmp = tensor.permute(0, 2, 3, 1) * 255
    elif type == "mask":
        tmp = tensor * 255
    return np.uint8(tmp.cpu().numpy())


def merge_image(images):
    if len(images) == 1:
        return images[0]
    else:
        np.concatenate(images, axis=0)


def plot_keypoints_batch(
    data,
    type_data="gt",
    unnormalize=True,
    patch_size=14,
    num_samples=16,
    concate_input_in_pred=True,
    write_num_matches=True,
):
    batch_size = data.src_img.shape[0]
    batch_size = min(batch_size, num_samples)

    # convert tensor to numpy
    src_img = convert_tensor_to_image(data.src_img, unnormalize=unnormalize)
    tar_img = convert_tensor_to_image(data.tar_img, unnormalize=unnormalize)

    if type_data == "gt":
        src_pts = data.src_pts.cpu().numpy()
        tar_pts = data.tar_pts.cpu().numpy()
    elif type_data == "pred":
        src_pts = data.pred_src_pts.cpu().numpy()
        tar_pts = data.pred_tar_pts.cpu().numpy()

    matching_imgs = []
    for idx in range(batch_size):
        mask = src_pts[idx, :, 0] != -1
        # mask = np.logical_and(mask, tar_pts[idx, :, 0] != -1)
        border_color = None  # [255, 0, 0]
        concate_input = concate_input_in_pred
        keypoint_img = plot_keypoints(
            src_img=src_img[idx],
            src_pts=src_pts[idx][mask],
            tar_img=tar_img[idx],
            tar_pts=tar_pts[idx][mask],
            border_color=border_color,
            concate_input=concate_input,
            write_num_matches=write_num_matches,
            patch_size=patch_size,
        )
        matching_imgs.append(torch.from_numpy(keypoint_img / 255.0).permute(2, 0, 1))
    matching_imgs = torch.stack(matching_imgs)
    return matching_imgs


def plot_ist_alignment_batch(data, num_samples=16, alpha=0.42):
    """Overlay the IST-warped template silhouette on real image crops.

    IST predicts relative in-plane rotation and scale per valid patch pair.
    We robustly aggregate those predictions per instance, align the template
    and target mask centroids, and blend the transformed template mask onto
    the unmasked target crop. This visualizes exactly what IST predicts
    without presenting it as a complete 6D CAD-pose estimate.
    """
    if not hasattr(data, "pred_relScale") or not hasattr(data, "pred_relInplane"):
        raise ValueError("Batch is missing IST predictions for visualization.")

    batch_size = min(data.src_img.shape[0], num_samples)
    src_masks = convert_tensor_to_image(data.src_mask, type="mask")
    tar_masks = convert_tensor_to_image(data.tar_mask, type="mask")
    if hasattr(data, "tar_actual_img"):
        tar_imgs = np.uint8(
            np.clip(data.tar_actual_img[:batch_size].detach().cpu().numpy(), 0, 1)
            .transpose(0, 2, 3, 1)
            * 255
        )
    else:
        tar_imgs = convert_tensor_to_image(
            data.tar_img[:batch_size], unnormalize=True
        )

    scales = data.pred_relScale[:batch_size].detach().cpu().numpy()
    rotations = data.pred_relInplane[:batch_size].detach().cpu().numpy()
    overlays = []
    for idx in range(batch_size):
        valid = (
            np.isfinite(scales[idx])
            & np.isfinite(rotations[idx, :, 0])
            & np.isfinite(rotations[idx, :, 1])
            & (scales[idx] > 0)
        )
        image = tar_imgs[idx].copy()
        if not valid.any():
            overlays.append(torch.from_numpy(image / 255.0).permute(2, 0, 1))
            continue

        scale = float(np.median(scales[idx][valid]))
        mean_vector = rotations[idx][valid].mean(axis=0)
        angle_deg = float(np.degrees(np.arctan2(mean_vector[1], mean_vector[0])))

        source_mask = src_masks[idx] > 127
        target_mask = tar_masks[idx] > 127
        source_grid = data.src_pts[idx].detach().cpu().numpy()
        target_grid = data.tar_pts[idx].detach().cpu().numpy()
        valid_points = valid & (source_grid[:, 0] != -1)
        valid_points &= target_grid[:, 0] != -1
        if not valid_points.any():
            overlays.append(torch.from_numpy(image / 255.0).permute(2, 0, 1))
            continue

        angle_rad = np.radians(angle_deg)
        linear = scale * np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)],
            ],
            dtype=np.float32,
        )
        # GigaPose patch coordinates refer to a 14-pixel feature grid.
        source_points = source_grid[valid_points].astype(np.float32) * 14.0
        target_points = target_grid[valid_points].astype(np.float32) * 14.0
        translations = target_points - source_points @ linear.T
        translation = np.median(translations, axis=0)
        affine = np.concatenate([linear, translation[:, None]], axis=1)
        height, width = image.shape[:2]
        warped_mask = cv2.warpAffine(
            source_mask.astype(np.uint8),
            affine,
            (width, height),
            flags=cv2.INTER_NEAREST,
        ).astype(bool)

        color = np.zeros_like(image)
        color[:, :] = (0, 255, 70)
        image[warped_mask] = np.uint8(
            (1.0 - alpha) * image[warped_mask] + alpha * color[warped_mask]
        )
        predicted_edge = create_edge_from_mask(
            image_size=(width, height),
            mask=Image.fromarray(np.uint8(warped_mask) * 255),
        )
        target_edge = create_edge_from_mask(
            image_size=(width, height),
            mask=Image.fromarray(np.uint8(target_mask) * 255),
        )
        image[predicted_edge] = (255, 40, 40)
        image[target_edge] = (0, 255, 70)
        cv2.putText(
            image,
            f"pred scale={scale:.2f} angle={angle_deg:.1f}",
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        overlays.append(torch.from_numpy(image / 255.0).permute(2, 0, 1))
    return torch.stack(overlays)


def plot_Kabsch(batch, affine_transforms, unnormalize=True, num_img_plots=16):
    """
    Plot the mapping includes: rotation, translation, scale
    """
    batch_size = batch.src_img.shape[0]
    batch_size = min(batch_size, num_img_plots)
    affined_imgs = []

    # convert tensor to numpy
    src_imgs = convert_tensor_to_image(batch.src_img, unnormalize=unnormalize)
    src_masks = convert_tensor_to_image(batch.src_mask, type="mask")
    tar_imgs = convert_tensor_to_image(batch.tar_img, unnormalize=unnormalize)
    tar_masks = convert_tensor_to_image(batch.tar_mask, type="mask")
    Ms = affine_transforms.clone().cpu().numpy()

    for idx in range(batch_size):
        src_img, tar_img = src_imgs[idx], tar_imgs[idx]
        src_mask, tar_mask = src_masks[idx], tar_masks[idx]
        M = Ms[idx]
        src_rgba = np.concatenate([src_img, src_mask[:, :, np.newaxis]], axis=2)
        tar_mask = Image.fromarray(np.uint8(tar_mask))

        # convert tar_img to gray scale
        rows, cols, _ = tar_img.shape
        tar_img = cv2.cvtColor(tar_img, cv2.COLOR_RGB2GRAY)
        tar_img = cv2.cvtColor(tar_img, cv2.COLOR_GRAY2RGB)
        tar_img = Image.fromarray(np.uint8(tar_img))

        M = M[:2, :]
        wrap_src_rgba = cv2.warpAffine(np.uint8(src_rgba), M, (cols, rows))
        wrap_src_PIL = Image.fromarray(wrap_src_rgba)
        wrap_src_mask = wrap_src_PIL.getchannel("A")
        tar_img.paste(wrap_src_PIL.convert("RGB"), (0, 0), wrap_src_mask)

        wrap_src_egde = create_edge_from_mask(
            image_size=tar_img.size, mask=wrap_src_mask
        )
        query_edge = create_edge_from_mask(image_size=tar_img.size, mask=tar_mask)

        tar_img = np.array(tar_img.convert("RGB"))
        tar_img[wrap_src_egde, :] = [255, 0, 0]
        tar_img[query_edge, :] = [0, 255, 0]
        # overlay
        # query = cv2.addWeighted(
        #     np.array(query), 0.5, np.array(reproj_ref_PIL.convert("RGB")), 0.5, 0
        # )
        affined_imgs.append(torch.from_numpy(tar_img / 255.0).permute(2, 0, 1))
    affined_imgs = torch.stack(affined_imgs)
    return affined_imgs
