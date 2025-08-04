import cv2
import logging
import torch
import os
import pathlib
import numpy as np

from PIL import Image
from typing import List
from ByteTrack_wrapper import Tracker

from Helpers import (
    annotate,
    plot_detections,
    grounded_segmentation,
    export_yolo_masks_json
)

# Set general logging config
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d [%(levelname)s] -- %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

logger.info(f'CUDA is available: {torch.cuda.is_available()}')

# Model size
dino_model = 'base'  # base or tiny
sam_model = 'base'  # base, large or huge

# Paths and Hyperparameters
WORKSPACE_DIR = pathlib.Path(__file__).parent.resolve()
DATASETS_DIR = WORKSPACE_DIR.joinpath('Datasets')

IMAGES_DIR = DATASETS_DIR / 'Images'
IMAGE_POST_PROCESSING_DIR = DATASETS_DIR / 'Post_Processing' / 'Images'
IMAGE_POST_PROCESSING_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_DIR = DATASETS_DIR / 'Videos'
VIDEO_POST_PROCESSING_DIR = DATASETS_DIR / 'Post_Processing' / 'VideoFrames'
VIDEO_POST_PROCESSING_DIR.mkdir(parents=True, exist_ok=True)

GROUNDED_SAM_MODEL = WORKSPACE_DIR.joinpath('Grounded_Sam')
RESULTS_FOLDER = GROUNDED_SAM_MODEL.joinpath('results')
YOLO_LABELS_FOLDER = RESULTS_FOLDER.joinpath('yolo_labels')
MODELS_FOLDER = GROUNDED_SAM_MODEL.joinpath('models')

os.makedirs(YOLO_LABELS_FOLDER, exist_ok=True)

def process_images(
    categories,
    cameras,
    image_dir: pathlib.Path,
    image_post_processing_dir: pathlib.Path,
    labels: List[str],
    threshold: float,
    detector_id: str,
    segmenter_id: str,
    labels_map,
    logger
):
    """
    Process images in all categories/cameras, detect and segment, save annotated results and YOLO masks.
    """
    for category in categories:
        for camera in cameras:
            input_folder = image_dir / category / camera
            output_folder = image_post_processing_dir / category / camera
            output_folder.mkdir(parents=True, exist_ok=True)

            image_paths = sorted(input_folder.glob('*.png'))

            logger.info(f'Processing {len(image_paths)} images in {category}/{camera}...')

            for image_path in image_paths:
                image_name = image_path.stem
                logger.info(f'Processing {category}/{camera}/{image_name}')

                image = Image.open(image_path).convert('RGB')

                image_array, detections = grounded_segmentation(
                    image=image,
                    labels=labels,
                    threshold=threshold,
                    polygon_refinement=True,
                    detector_id=detector_id,
                    segmenter_id=segmenter_id
                )

                # Save annotated image
                annotated_path = output_folder / f'{image_name}_annotated.png'
                plot_detections(image_array, detections, save_name=annotated_path, show=False)

                # Save YOLO labels (and polygons, etc)
                yolo_output_folder = output_folder  # change to change where .txts are saved
                export_yolo_masks_json(
                    detections_dino_sam=detections,
                    labels_map=labels_map,
                    image_name=image_name,
                    export_folder=yolo_output_folder,
                    img_width=image.width,
                    img_height=image.height
                )

    logger.info("Image processing complete.")


def compute_iou(boxA, boxB):
    '''Compute Intersection over Union (IoU) between two bounding boxes.'''
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def compute_center_distance(boxA, boxB):
    '''Compute Euclidean distance between centers of two bounding boxes. Legacy function, not used in current code.'''
    cxA = (boxA[0] + boxA[2]) / 2
    cyA = (boxA[1] + boxA[3]) / 2
    cxB = (boxB[0] + boxB[2]) / 2
    cyB = (boxB[1] + boxB[3]) / 2
    return np.sqrt((cxA - cxB)**2 + (cyA - cyB)**2)

def find_background_frame(video_frames, all_detections):
    """
    Find a frame in the video that has no detections. This function assumes that the background frame is static. 
    """
    for idx, detections in enumerate(all_detections):
        if len(detections) == 0:
            logger.info(f"Background frame found at index {idx} with zero detections.")
            return idx, video_frames[idx], np.zeros((video_frames[idx].shape[0], video_frames[idx].shape[1]), dtype=np.uint8)

    # Fallback: if frame has zero detections, do as before
    min_detections = float('inf')
    background_idx = 0
    masks = []

    for idx, (frame, detections) in enumerate(zip(video_frames, all_detections)):
        combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        count = 0

        for det in detections:
            if det.mask is not None:
                combined_mask = np.logical_or(combined_mask, det.mask)
                count += 1
        masks.append(combined_mask)

        if count < min_detections:
            min_detections = count
            background_idx = idx

    logger.info(f"Background frame found at index {background_idx} with {min_detections} detections.")
    return background_idx, video_frames[background_idx], masks[background_idx]

def erase_ids_from_frame(frame, detections, ids_to_erase, background_frame, background_mask):
    """
    Erase specified IDs from the frame by replacing them with the background.
    """
    new_frame = frame.copy()
    for det in detections:
        if det.track_id in ids_to_erase and det.mask is not None:
            mask = (det.mask > 0)
            # Only use background pixels where the background_mask is 0 (not a detection)            
            safe_bg = np.logical_and(mask, background_mask == 0)
            for c in range(3): # RGB
                new_frame[:, :, c][safe_bg] = background_frame[:, :, c][safe_bg]
    
    return new_frame

def process_and_erase_ids(video_frames, all_detections, ids_to_erase):
    """
    Process video frames and erase specified IDs from images.
    """
    # First pass, get background frame
    bg_idx, background_frame, background_mask = find_background_frame(video_frames, all_detections)

    out_frames = []
    for idx, (frame, detections) in enumerate(zip(video_frames, all_detections)):
        new_frame = erase_ids_from_frame(frame, detections, ids_to_erase, background_frame, background_mask)
        out_frames.append(new_frame)
    return out_frames

def videoProcessing(
    video_path: pathlib.Path,
    output_dir: pathlib.Path,
    detector_id: str,
    segmenter_id: str,
    labels: List[str],
    threshold: float,
    polygon_refinement: bool = True,
    erase_ids: List[int] = None
):
    """
    Process a video frame-by-frame using Grounded SAM and save annotated video output.

    Args:
        video_path: Path to the input .avi file
        output_dir: Folder where processed frames and video will be saved
        detector_id: HF model ID for object detection
        segmenter_id: HF model ID for segmentation
        labels: List of labels to detect
        threshold: Detection confidence threshold
        polygon_refinement: Whether to refine masks into polygons
        erase_ids: List of track IDs to erase from the video (optional)
    """

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    video_capture = cv2.VideoCapture(str(video_path))

    # Check if video opened successfully
    if not video_capture.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    # Get video properties
    frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Collect all frames and detection results
    video_frames = []
    all_detections = []

    # Initialize ByteTrack tracker
    tracker = Tracker(args={
        "track_thresh": 0.5,          # Detection threshold for tracking
        "track_buffer": 100,          # Buffer size for tracking
        "match_thresh": 0.8,          # Threshold for matching detections
        "frame_rate": frame_rate,     # Frame rate of the video
        "mot20": False                # Whether to use MOT20 dataset settings (MOT17 otherwise, MNT20 is stricter whith filtering)
    })

    # IoU threshold for matching detections to tracks
    iou_threshold = 0.5  

    frame_idx = 0
    while video_capture.isOpened():   
        ret, frame = video_capture.read()
        if not ret:
            break

        # Progression logging
        logger.info(f"Processing frame {frame_idx + 1}/{total_frames}")

        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Perform grounded segmentation
        image_array, detections = grounded_segmentation(
            image=image_pil,
            labels=labels,
            threshold=threshold,
            polygon_refinement=polygon_refinement,
            detector_id=detector_id,
            segmenter_id=segmenter_id
        )

        # Convert detections to a format suitable for ByteTrack
        height, width = frame.shape[0], frame.shape[1]
        frame_detections = []

        for det in detections:
            if det.label.lower().startswith("person"):
                x1, y1, x2, y2 = det.box.xyxy
                score = det.score
                # Clamp coordinates to image bounds (meant to prevent out-of-bounds errors)
                x1 = max(0, min(x1, width-1))
                y1 = max(0, min(y1, height-1))
                x2 = max(0, min(x2, width-1))
                y2 = max(0, min(y2, height-1))
                # Makes sure x1 < x2 and y1 < y2
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                # **NO CLASS!** (The version of ByteTrack used can only handle 5 values per detection)
                frame_detections.append([x1, y1, x2, y2, score])  

        detection_array = np.array(frame_detections, dtype=np.float32)

        if len(detection_array) > 0:
            img_info = [frame.shape[0], frame.shape[1]]  # [height, width]
            img_size = (frame.shape[0], frame.shape[1])

            # Warning if any NaN or inf values are detected in the detection array (used for debugging)
            if not np.all(np.isfinite(detection_array)):
                print(f"Frame {frame_idx}: WARNING - NaN or inf detected in input array!", detection_array)

            # Logging detections for debugging
            # print(f"FRAME {frame_idx} detections for ByteTrack:")
            # for d in detection_array:
            #    print(d)

            # ByteTrack expects (N, 5) [x1, y1, x2, y2, score]
            tracks = tracker.update(detection_array, img_info, img_size)

            # Assign track IDs to detection objects
            for track in tracks:
                tid, tlwh, score = track
                tx, ty, tw, th = tlwh
                x1, y1, x2, y2 = tx, ty, tx + tw, ty + th
                track_box = [int(x1), int(y1), int(x2), int(y2)]

                # Assign track_id by IoU matching
                for det in detections:
                    if det.label.lower().startswith("person"):
                        det_box = det.box.xyxy
                        iou = compute_iou(det_box, track_box)
                        if iou > iou_threshold:  
                            det.track_id = int(tid)
                            break

        # Store frame and detection results
        video_frames.append(frame.copy())
        all_detections.append(detections)

        # Logging skipped frames (in case of no detections)
        if len(detections) == 0:
            logger.warning(f"Frame {frame_idx} skipped: no detections.")
            continue  # Skip to next frame

        frame_idx +=1

    video_capture.release()
    logger.info(f"Number of frames processed: {len(video_frames)}")  # Log the number of frames processed at end
    logger.info(f"Number of total unique IDs: {len(set(det.track_id for detections in all_detections for det in detections if det.track_id is not None))}")

    # Check if erase_ids is provided and handle special case for -1 (erasing all IDs)
    if erase_ids is not None and len(erase_ids) == 1 and erase_ids[0] == -1:
        all_ids = {det.track_id for detections in all_detections for det in detections if det.track_id is not None}
        erase_ids = list(all_ids)  # Erase all IDs if -1 is specified
        logger.info(f"erase_ids=[-1] detected; erasing ALL IDs: {erase_ids}")

    # Erase specified IDs if provided
    if erase_ids:
        ids_present = any(det.track_id in erase_ids for detections in all_detections for det in detections if det.track_id is not None)
        logger.info(f"Erasing IDs: {erase_ids} from video frames. Present in video: {ids_present}")
        out_frames = process_and_erase_ids(video_frames, all_detections, erase_ids)
    else:
        out_frames = video_frames

    # Write final video
    output_video_path_str = str(output_dir / 'processed_video.avi')

    out = cv2.VideoWriter(
        output_video_path_str,
        cv2.VideoWriter_fourcc(*'XVID'),
        frame_rate,
        (frame_width, frame_height)
    )

    for idx, (frame, detections) in enumerate(zip(out_frames, all_detections)):
        # Remove erased IDs from detections for annotation
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for annotation
        filtered_detections = [d for d in detections if d.track_id not in erase_ids]
        annotated_np = annotate(frame_rgb, filtered_detections)
        frame_bgr = cv2.cvtColor(annotated_np, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    # for frame in out_frames:
    #    out.write(frame)
    out.release() 
    
    # Final logging message
    logger.info("Video processing complete.")


def main():
    # Settings
    labels = ['car.', 'person.', 'bicycle.']
    threshold = 0.3
    labels_map = {x: i for i, x in enumerate(labels)}

    detector_id = f'IDEA-Research/grounding-dino-{dino_model}'
    segmenter_id = f'facebook/sam-vit-{sam_model}'

    # Iterate over categories and cameras (for image processing)
    categories = ['Bicycle', 'Cars']
    cameras = ['Left', 'Right']

    '''
    # Image processing pipeline
    process_images(
        categories=categories, 
        cameras=cameras, 
        image_dir=IMAGES_DIR, 
        image_post_processing_dir=IMAGE_POST_PROCESSING_DIR, 
        labels=labels, 
        threshold=threshold, 
        detector_id=detector_id, 
        segmenter_id=segmenter_id, 
        labels_map=labels_map, 
        logger=logger
    )
    '''
    
    # Process videos
    video_files = sorted(VIDEO_DIR.glob('*.avi'))

    # Check if any video files were found and iterate over them
    for video_path in video_files:
        logger.info(f"Processing video file: {video_path.name}")
        video_output_dir = VIDEO_POST_PROCESSING_DIR / video_path.stem
        video_output_dir.mkdir(parents=True, exist_ok=True)

        # Call the video processing function
        videoProcessing(
            video_path=video_path,
            output_dir=video_output_dir,
            detector_id=detector_id,
            segmenter_id=segmenter_id,
            labels=labels,
            threshold=threshold,
            polygon_refinement=True,
            erase_ids=[-1]  # Example IDs to erase; change as needed; [-1] to erase all IDs
        )
    

if __name__ == '__main__':
    main()