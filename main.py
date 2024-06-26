import os
import numpy as np
import cv2
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch
import torchvision
import deepgaze_pytorch
from deepgaze_pytorch.modules import encode_scanpath_features
import collections
from collections.abc import Sequence, MutableMapping
collections.Sequence = Sequence
collections.MutableMapping = MutableMapping

import pysaliency.models
from pysaliency.models import sample_from_logdensity
import h5py
from tqdm import tqdm
from foveal_transform import FovealTransform
import logging
from typing import List, Tuple, Set

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FOVEA_SIZE = 0.1  # Fovea size as a fraction of image size
IMG_TARGET_SIZE = 175  # Retina Resolution of the retina warp output
RETINA_SIZE = 128 # Size of the Output Retina Warps (128x128)
JITTER_TYPE = "gaussian"
JITTER_AMOUNT = 0.0
NUM_FIXATIONS_TRAIN = 10
OUTPUT_DIR = 'retina_warps'
OUTPUT_FILE = 'retina_warps.h5'
COCO_DATASET_DIR = '/share/klab/datasets/avs/input/NSD_scenes_MEG_size_adjusted_925'
RANDOM_SEED = 42
INCLUDE_INITIAL_FIXATION = True # Flag to control whether to include the initial fixation at the center point
CENTERBIAS_FILE = 'centerbias_mit1003.npy'
PROCESSED_FILES_FILE = 'processed_files.txt'

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

# Initialize the DeepGaze model
model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(DEVICE)

# Load the centerbias log density
centerbias_template = np.load(CENTERBIAS_FILE)

# Load the MS COCO dataset
coco_train = COCO(os.path.join('/share/klab/datasets/avs/input/annotations', 'instances_train2017.json'))
coco_val = COCO(os.path.join('/share/klab/datasets/avs/input/annotations', 'instances_val2017.json'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_fixations(fixations: List[Tuple[float, float]], width: int, height: int) -> List[Tuple[float, float]]:
    """
    Normalize fixation coordinates to the range [-1, 1].

    Args:
        fixations: List of fixation coordinates (x, y).
        width: Width of the image.
        height: Height of the image.

    Returns:
        List of normalized fixation coordinates (x, y).
    """
    normalized_fixations = []
    for x, y in fixations:
        normalized_x = (x - width / 2.0) / (width / 2.0)
        normalized_y = (y - height / 2.0) / (height / 2.0)
        normalized_fixations.append((normalized_x, normalized_y))
    return normalized_fixations

def unnormalize_fixations(fixations: List[Tuple[float, float]], width: int, height: int) -> List[Tuple[float, float]]:
    """
    Unnormalize fixation coordinates from the range [-1, 1] to the original image coordinates.

    Args:
        fixations: List of normalized fixation coordinates (x, y).
        width: Width of the image.
        height: Height of the image.

    Returns:
        List of unnormalized fixation coordinates (x, y).
    """
    unnormalized_fixations = []
    for x, y in fixations:
        unnormalized_x = (x * (width / 2.0)) + (width / 2.0)
        unnormalized_y = (y * (height / 2.0)) + (height / 2.0)
        unnormalized_fixations.append((unnormalized_x, unnormalized_y))
    return unnormalized_fixations

def get_fixation_history(fixation_coordinates: List[float], model: deepgaze_pytorch.DeepGazeIII) -> List[float]:
    """
    Get the fixation history based on the model's included fixations.

    Args:
        fixation_coordinates: List of fixation coordinates.
        model: DeepGaze model.

    Returns:
        List of fixation coordinates based on the model's included fixations.
    """
    history = []
    for index in model.included_fixations:
        try:
            history.append(fixation_coordinates[index])
        except IndexError:
            history.append(np.nan)
    return history

def rescale_centerbias(centerbias_template: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Rescale the centerbias template to match the image size.

    Args:
        centerbias_template: Centerbias template.
        height: Height of the image.
        width: Width of the image.

    Returns:
        Rescaled centerbias template.
    """
    centerbias = zoom(centerbias_template, (height / centerbias_template.shape[0], width / centerbias_template.shape[1]), order=0, mode='nearest')
    centerbias -= logsumexp(centerbias)
    assert centerbias.shape == (height, width), f"Centerbias shape {centerbias.shape} does not match image shape ({height}, {width})"
    return centerbias

def predict_fixation(model: deepgaze_pytorch.DeepGazeIII, image_tensor: torch.Tensor, centerbias_tensor: torch.Tensor,
                     x_hist_tensor: torch.Tensor, y_hist_tensor: torch.Tensor, random_state: np.random.RandomState) -> Tuple[float, float]:
    """
    Predict the next fixation point using the model.

    Args:
        model: DeepGaze model.
        image_tensor: Image tensor.
        centerbias_tensor: Centerbias tensor.
        x_hist_tensor: Tensor of x-coordinates of fixation history.
        y_hist_tensor: Tensor of y-coordinates of fixation history.
        random_state: Random state.

    Returns:
        Tuple of predicted fixation coordinates (x, y).
    """
    log_density_prediction = model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)
    logD = log_density_prediction.detach().cpu().numpy()[0, 0]
    next_x, next_y = sample_from_logdensity(logD, rst=random_state)

    # Get the height and width of the image tensor
    _, _, height, width = image_tensor.shape

    # Check if the predicted fixation is within the image bounds
    assert 0 <= next_x < width, f"Predicted fixation x-coordinate {next_x} is outside the image bounds (width: {width})"
    assert 0 <= next_y < height, f"Predicted fixation y-coordinate {next_y} is outside the image bounds (height: {height})"

    return next_x, next_y

def generate_retina_warps(image: np.ndarray, num_fixations: int, model: deepgaze_pytorch.DeepGazeIII) -> Tuple[np.ndarray, List[np.ndarray], List[float], List[float]]:
    """
    Generate retina warps for the given image.

    Args:
        image: Input image.
        num_fixations: Number of fixations to generate.
        model: DeepGaze model.

    Returns:
        Tuple containing the original image, list of retina warps, list of x-coordinates of fixation history,
        and list of y-coordinates of fixation history.
    """
    height, width = image.shape[:2]
    centerbias = rescale_centerbias(centerbias_template, height, width)
    image_tensor = torch.tensor([image.transpose(2, 0, 1)]).to(DEVICE)
    image_tensor = image_tensor / 255.0

    centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)

    fixation_history_x = [width // 2]
    fixation_history_y = [height // 2]
    normalized_fixation_history = [(0, 0)]  # Store normalized fixations

    random_state = np.random.RandomState(seed=RANDOM_SEED)

    foveal_transform = FovealTransform(fovea_size=FOVEA_SIZE,
                                       img_target_size=IMG_TARGET_SIZE,
                                       img_size=(height, width),
                                       jitter_type=JITTER_TYPE,
                                       jitter_amount=JITTER_AMOUNT,
                                       device=DEVICE,
                                       random_seed=RANDOM_SEED,
                                       retina_size=RETINA_SIZE)

    retina_warps = []

    if INCLUDE_INITIAL_FIXATION:
        # Initial fixation at the center point (0, 0)
        initial_fixation = torch.tensor([[0.0, 0.0]]).to(DEVICE)
        initial_retina_img = foveal_transform(image_tensor.float(), initial_fixation)[0].permute(1, 2, 0).cpu().numpy()
        retina_warps.append(initial_retina_img)

    for _ in range(num_fixations - 1):
        x_hist = get_fixation_history(fixation_history_x, model)
        y_hist = get_fixation_history(fixation_history_y, model)
        x_hist_tensor = torch.tensor([x_hist]).to(DEVICE)
        y_hist_tensor = torch.tensor([y_hist]).to(DEVICE)
        next_x, next_y = predict_fixation(model, image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor, random_state)

        normalized_next_fixation = normalize_fixations([(next_x, next_y)], width, height)
        normalized_next_x, normalized_next_y = normalized_next_fixation[0]

        fixation_history_x.append(next_x)
        fixation_history_y.append(next_y)
        normalized_fixation_history.append((normalized_next_x, normalized_next_y))

        fixations = torch.tensor([[normalized_next_x, normalized_next_y]]).to(DEVICE)

        retina_img = foveal_transform(image_tensor.float(), fixations)[0].permute(1, 2, 0).cpu().numpy()

        retina_warps.append(retina_img)

    return image, retina_warps, fixation_history_x, fixation_history_y

def save_to_h5(original_image: np.ndarray, retina_warps: List[np.ndarray], fixation_history_x: List[float],
               fixation_history_y: List[float], output_path: str, file_name: str) -> None:
    """
    Save the original image, retina warps, and fixation history to an HDF5 file.

    Args:
        original_image: Original input image.
        retina_warps: List of retina warp images.
        fixation_history_x: List of x-coordinates of fixation history.
        fixation_history_y: List of y-coordinates of fixation history.
        output_path: Path to the output HDF5 file.
        file_name: Name of the file.
    """
    with h5py.File(output_path, 'a') as f:  # 'a' mode for appending data
        grp = f.create_group(file_name)
        grp.create_dataset('original_image', data=original_image)
        grp.create_dataset('retina_warps', data=np.array(retina_warps))
        grp.create_dataset('fixation_history_x', data=np.array(fixation_history_x))
        grp.create_dataset('fixation_history_y', data=np.array(fixation_history_y))
        grp.attrs['file_name'] = file_name

def process_image(img_path: str, num_fixations: int, model: deepgaze_pytorch.DeepGazeIII) -> None:
    """
    Process a single image by generating retina warps and saving the results.

    Args:
        img_path: Path to the input image.
        num_fixations: Number of fixations to generate.
        model: DeepGaze model.
    """
    image = cv2.imread(img_path)

    if image is None:
        logger.error(f"Failed to load image: {img_path}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    original_image, retina_warps, fixation_history_x, fixation_history_y = generate_retina_warps(image, num_fixations, model)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    file_name = os.path.basename(img_path)
    save_to_h5(original_image, retina_warps, fixation_history_x, fixation_history_y, output_path, file_name)

def save_processed_files(processed_files: Set[str], processed_files_file: str) -> None:
    """
    Save the list of processed file names to a file.

    Args:
        processed_files: Set of processed file names.
        processed_files_file: Path to the file to save the processed file names.
    """
    with open(processed_files_file, 'w') as f:
        f.write('\n'.join(processed_files))

def load_processed_files(processed_files_file: str) -> Set[str]:
    """
    Load the list of processed file names from a file.

    Args:
        processed_files_file: Path to the file containing the processed file names.

    Returns:
        Set of processed file names.
    """
    if os.path.exists(processed_files_file):
        with open(processed_files_file, 'r') as f:
            return set(f.read().splitlines())
    return set()

def main() -> None:
    """
    Main function to process images and generate retina warps.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load the processed file names
    processed_files = load_processed_files(PROCESSED_FILES_FILE)

    # Generate retina warps for the images in the dataset
    img_files = [f for f in os.listdir(COCO_DATASET_DIR) if f.lower().endswith('.jpg')]
    logger.info(f"Number of images: {len(img_files)}")
    for img_file in tqdm(img_files, desc="Processing images"):
        if img_file in processed_files:
            continue
        img_path = os.path.join(COCO_DATASET_DIR, img_file)
        process_image(img_path, NUM_FIXATIONS_TRAIN, model)
        processed_files.add(img_file)
        save_processed_files(processed_files, PROCESSED_FILES_FILE)

if __name__ == '__main__':
    main()