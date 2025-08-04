import random
import json
import cv2
import requests
import itertools
import torch
import logging
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from PIL import Image
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline


# Use this if you run out of memory during inference
#   This moves SAM to the CPU, which is slower but might enable
#   inference on GPUs with low RAM
segmentor_to_cpu = False

# Set general logging config
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d [%(levelname)s] -- %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None
    track_id: Optional[int] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))
    
# New Helper function to get a color for a track ID
def get_color_for_track(track_id: int) -> tuple:
    # Use a hash function to generate a color for each track ID
    random.seed(track_id)
    color = [random.randint(0, 255) for _ in range(3)]
    return tuple(color)



# Functions given to us by Nicolas and the Robotics Lab team from Hochschule Trier
# Define Grounded SAM model
def detect(
        image: Image.Image,
        labels: List[str],
        threshold: float,
        detector_id: str
) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    """

    object_detector = pipeline(model=detector_id,
                               task='zero-shot-object-detection',
                               device=device)

    labels = [label if label.endswith(".") else label + "." for label in
              labels]

    results = object_detector(image, candidate_labels=labels,
                              threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]

    return results


def segment(
        image: Image.Image,
        detection_results: List[Dict[str, Any]],
        segmenter_id: str,
        polygon_refinement: bool = False
) -> List[DetectionResult]:
    """
    Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
    """

    if segmentor_to_cpu:
        segmentor_device = 'cpu'
    else:
        segmentor_device = device

    # Early exit if no detections
    if len(detection_results) == 0:
        return detection_results  # No masks to compute

    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id,
                                                             device_map=segmentor_device)

    processor = AutoProcessor.from_pretrained(segmenter_id, use_fast=True)

    boxes = get_boxes(detection_results)
    if not boxes or not boxes[0]:  # still check structure
        return detection_results

    inputs = processor(images=image, input_boxes=boxes,
                       return_tensors='pt').to(segmentor_device)

    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results



def grounded_segmentation(
        image: Union[Image.Image, str],
        labels: List[str],
        detector_id: str,
        segmenter_id: str,
        threshold: float,
        polygon_refinement: bool = False
) -> Tuple[np.ndarray, List[DetectionResult]]:
    if isinstance(image, str):
        image = load_image(image)

    detections = detect(image, labels, threshold, detector_id)
    detections = segment(image, detections, segmenter_id, polygon_refinement)

    return np.array(image), detections


def annotate(image: Union[Image.Image, np.ndarray],
             detection_results: List[DetectionResult]) -> np.ndarray:
    # Convert PIL Image to OpenCV format
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    # Iterate over detections and add bounding boxes and masks
    for detection in detection_results:
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        # Sample a random color for each detection
        # Original line, this for some reason resulted in different colors for the same detection for each frame
        # color = np.random.randint(0, 256, size=3) 

        # Assign color for each track ID
        if detection.track_id is not None:
            color = get_color_for_track(detection.track_id)
        else:
            color = (0, 255, 255)  # Or some default color for untracked boxes


        label_text = f'{label}: {score:.2f}'
        # If track_id is available, append it to the label
        if detection.track_id is not None:
            label_text += f' (ID: {detection.track_id})'

        # Draw bounding box
        cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax),
                      color, 2)
        cv2.putText(image_cv2, label_text,
                    (box.xmin, box.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 2)

        # If mask is available, apply it
        if mask is not None:
            # Convert mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_cv2, contours, -1, color, 2)

    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)


def plot_detections(
        image: Union[Image.Image, np.ndarray],
        detections: List[DetectionResult],
        save_name: Optional[str] = None,
        show: bool = False  # False for batch processing, True for interactive display
) -> None:
    annotated_image = annotate(image, detections)
    plt.imshow(annotated_image)
    plt.axis('off')
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()  # Close the plot to free memory


def random_named_css_colors(num_colors: int) -> List[str]:
    """
    Returns a list of randomly selected named CSS colors.

    Args:
    - num_colors (int): Number of random colors to generate.

    Returns:
    - list: List of randomly selected named CSS colors.
    """

    # List of named CSS colors
    named_css_colors = [
        'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige',
        'bisque', 'black', 'blanchedalmond',
        'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse',
        'chocolate', 'coral', 'cornflowerblue',
        'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod',
        'darkgray', 'darkgreen', 'darkgrey',
        'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
        'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
        'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise',
        'darkviolet', 'deeppink', 'deepskyblue',
        'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite',
        'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite',
        'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey',
        'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory',
        'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon',
        'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow',
        'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon',
        'lightseagreen', 'lightskyblue', 'lightslategray',
        'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen',
        'linen', 'magenta', 'maroon', 'mediumaquamarine',
        'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen',
        'mediumslateblue', 'mediumspringgreen', 'mediumturquoise',
        'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose',
        'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive',
        'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod',
        'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip',
        'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple',
        'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown',
        'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver',
        'skyblue', 'slateblue', 'slategray', 'slategrey',
        'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato',
        'turquoise', 'violet', 'wheat', 'white',
        'whitesmoke', 'yellow', 'yellowgreen'
    ]

    # Sample random named CSS colors
    return random.sample(named_css_colors,
                         min(num_colors, len(named_css_colors)))


def plot_detections_plotly(
        image: np.ndarray,
        detections: List[DetectionResult],
        class_colors: Optional[Dict[str, str]] = None
) -> None:
    # If class_colors is not provided, generate random colors for each class
    if class_colors is None:
        num_detections = len(detections)
        colors = random_named_css_colors(num_detections)
        class_colors = {}
        for i in range(num_detections):
            class_colors[i] = colors[i]

    fig = px.imshow(image)
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
    )

    # Add bounding boxes
    shapes = []
    annotations = []
    for idx, detection in enumerate(detections):
        label = detection.label
        box = detection.box
        score = detection.score
        mask = detection.mask

        polygon = mask_to_polygon(mask)

        fig.add_trace(go.Scatter(
            x=[point[0] for point in polygon] + [polygon[0][0]],
            y=[point[1] for point in polygon] + [polygon[0][1]],
            mode='lines',
            line=dict(color=class_colors[idx], width=2),
            fill='toself',
            name=f'{label}: {score:.2f}'
        ))

        xmin, ymin, xmax, ymax = box.xyxy
        shape = [
            dict(
                type='rect',
                xref='x', yref='y',
                x0=xmin, y0=ymin,
                x1=xmax, y1=ymax,
                line=dict(color=class_colors[idx])
            )
        ]
        annotation = [
            dict(
                x=(xmin + xmax) // 2, y=(ymin + ymax) // 2,
                xref='x"', yref='y',
                text=f'{label}: {score:.2f}',
            )
        ]

        shapes.append(shape)
        annotations.append(annotation)

    # Update layout
    button_shapes = [
        dict(label='None', method='relayout', args=['shapes', []])]
    button_shapes = button_shapes + [
        dict(label=f'Detection {idx + 1}', method='relayout',
             args=['shapes', shape]) for idx, shape in enumerate(shapes)
    ]
    button_shapes = button_shapes + [
        dict(label='All', method='relayout', args=['shapes', sum(shapes, [])])]

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        # margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        updatemenus=[
            dict(
                type='buttons',
                direction='up',
                buttons=button_shapes
            )
        ],
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    # Show plot
    fig.show(renderer='browser')


def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon


def polygon_to_mask(polygon: List[Tuple[int, int]],
                    image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """

    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask


def load_image(image_str: str) -> Image.Image:
    if image_str.startswith('http'):
        image = Image.open(requests.get(image_str, stream=True).raw).convert(
            'RGB')
    else:
        image = Image.open(image_str).convert('RGB')

    return image


def get_boxes(results: List[DetectionResult]) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]


def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> \
        List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks


def export_yolo_masks(detections_dino_sam, labels_map, image_name,
                      export_folder, img_width, img_height):
    masks = [x.mask for x in detections_dino_sam]
    labels = [labels_map[x.label] for x in detections_dino_sam]

    contours = [cv2.findContours(x.astype(np.uint8),
                                 cv2.RETR_TREE,
                                 cv2.CHAIN_APPROX_SIMPLE) for x in
                masks]
    labels = list(itertools.chain.from_iterable(
        [[y[0]] * y[1] for y in zip(labels, [len(x[0]) for x in contours])]))
    contours = [np.squeeze(y).T for y in
                itertools.chain.from_iterable(x[0] for x in contours)]
    contours = [[y[0] / img_width, y[1] / img_height] for y in
                contours]
    contours = [np.vstack((x[0], x[1])).T.tolist() for x in
                contours]

    # Overwrite the file for it to be empty to start with
    with open(export_folder.joinpath(f'{image_name}.txt'), 'w') as _:
        pass

    with open(export_folder.joinpath(f'{image_name}.txt'), 'a') as f:
        for i in range(len(contours)):
            f.write(f'{labels[i]} ' +
                    ' '.join([str(x) for x in itertools.chain.from_iterable(
                        contours[i])]) + '\n')
            



# Slightly modified version of the above function to export annotiation to JSON in one signle file

def export_yolo_masks_json(
    detections_dino_sam,
    labels_map,
    image_name,
    export_folder,
    img_width: int,
    img_height: int,
    annotation_file: str = 'annotations.json'
):
    """
    Appends polygon annotations to a JSON file per folder.
    Each entry is a dict with image name, label index, and normalized polygon.

    Args:
        detections_dino_sam: List of DetectionResult
        labels_map: Dict[str, int]
        image_name: e.g., "000025"
        export_folder: output path"
        img_width, img_height: for normalization
        annotation_file: JSON filename (default: 'annotations.json')
    """

    masks = [x.mask for x in detections_dino_sam]
    labels = [labels_map[x.label] for x in detections_dino_sam]

    contours_raw = [cv2.findContours(x.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) for x in masks]

    labels = list(itertools.chain.from_iterable(
        [[label] * len(contour[0]) for label, contour in zip(labels, contours_raw)]
    ))

    contours = [np.squeeze(c).T for c in itertools.chain.from_iterable(c[0] for c in contours_raw)]
    contours = [[pt[0] / img_width, pt[1] / img_height] for pt in contours]
    contours = [np.vstack((x[0], x[1])).T.tolist() for x in contours]

    new_annotations = []
    for label, polygon in zip(labels, contours):
        annotation = {
            "image": f"{image_name}.png",
            "label": label,
            "polygon": [[round(x, 6), round(y, 6)] for x, y in polygon]
        }
        new_annotations.append(annotation)

    # Load existing annotations if the file exists
    annotations_path = export_folder / annotation_file
    if annotations_path.exists():
        with open(annotations_path, 'r') as f:
            all_annotations = json.load(f)
    else:
        all_annotations = []

    # Append new entries and save
    all_annotations.extend(new_annotations)
    with open(annotations_path, 'w') as f:
        json.dump(all_annotations, f, indent=2)

            

if __name__ == '__main__':
    print("This file only contains helper functions and classes. Run DynamicObjectFiltering.py instead.")