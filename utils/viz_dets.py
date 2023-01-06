# Visualize the MaskRCNN detections


import matplotlib.colors as mcolors
import numpy as np
import torch
from torchvision.utils import draw_bounding_boxes
from PIL import ImageColor


def drawDetections(
    image, detections, score_th=0.3, draw_bbox=True, width=5, fontsize=50,
    draw_mask=True, alpha=0.5
):
    """
    Draw 2D bbox and/or mask detections on a SINGLE image
    @param image (CxWxH): Image
    @param detections (PandasTensor): MRCNN detections
    @param score_th (float): Detection score threshold, below which no bbox
    @param draw_bbox (bool): Whether to draw bboxes
    @param width (float):  Bbox line width
    @param fontsize (int): The requested font size in points
    @param draw_mask (bool): Whether to draw masks
    @param alpha (float): Transparency of the mask
    """
    labels = detections.infos["label"]
    scores = detections.infos["score"]
    # Only Viz bboxes beyond score threshold
    keep = np.where(scores > score_th)[0]
    labels = labels[keep]
    # Squeeze the batch dimension if any
    if len(image.shape) == 4:
        assert image.shape[0] == 1, "Error: only support single img"
        image = image.squeeze(0)
    # Convert image to 0~255 if it's not 
    if image.dtype in [torch.float, torch.float32]:
        image  = (image * 255).to(torch.uint8)
    # Generate object colors based on object label IDs
    colors = list(mcolors.TABLEAU_COLORS.values()) * 10
    colors_obj, colors_obj_rgb = [], []
    for l in labels:
        # Extract color based on label ID if any, otherwise black
        color = colors[int(l[-3:])] if l[-3:].isnumeric() else '#000000'
        color_rgb = ImageColor.getrgb(color)
        color_rgb = torch.tensor(color_rgb).to(torch.uint8)
        colors_obj.append(color)
        colors_obj_rgb.append(color_rgb)
    if draw_bbox: 
        bboxes = detections.bboxes
        bboxes = bboxes[keep, ...]
        image = draw_bounding_boxes(
            image.cpu(), bboxes, labels,
            colors=colors_obj, width=width, font_size=fontsize
        )
    if draw_mask:
        masks = detections.masks
        masks = masks[keep, ...]
        img_to_draw = image.detach().clone()
        for mask, color in zip(masks, colors_obj_rgb):
            img_to_draw[:, mask] = color[:, None]
        image = image * (1 - alpha) + img_to_draw * alpha
        image = image.to(torch.uint8)
    # Convert image to HxWxC format
    image = image.permute(1, 2, 0).detach().cpu().numpy()
    return image


