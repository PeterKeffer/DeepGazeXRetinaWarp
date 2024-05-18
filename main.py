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

# Hyperparameters
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FOVEA_SIZE = 0.1  # Fovea size as a fraction of image size
IMG_TARGET_SIZE = 100  # Target size of the retina warp output
JITTER_TYPE = "gaussian"
JITTER_AMOUNT = 0.1
NUM_FIXATIONS_TRAIN = 10
NUM_FIXATIONS_VAL = 5
OUTPUT_DIR = 'retina_warps'
OUTPUT_FILE = 'retina_warps.h5'
COCO_DATASET_DIR = '/share/klab/datasets/avs/input/NSD_scenes_MEG_size_adjusted_925'
RANDOM_SEED = 42

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# Initialize the DeepGaze model
model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(DEVICE)

# Load the centerbias log density
centerbias_file = 'centerbias_mit1003.npy'
centerbias_template = np.load(centerbias_file)

# Load the MS COCO dataset
# Load the MS COCO dataset
coco_train = COCO(os.path.join('/share/klab/datasets/avs/input/annotations', 'instances_train2017.json'))
coco_val = COCO(os.path.join('/share/klab/datasets/avs/input/annotations', 'instances_val2017.json'))

class FovealTransform(torch.nn.Module):
    def __init__(self, fovea_size, img_target_size, img_size, jitter_type, jitter_amount, device, random_seed, retina_size=90):
        super().__init__()
        self.device = device
        self.roh_0 = int(fovea_size * (img_size[1] // 2))
        self.roh_max = int(img_size[1] // 2)
        retina_resolution = img_target_size  # img_target_size if img_size[0] == 256 else img_target_size*2
        self.N_r, self.resulting_resolution = self.find_number_of_rings(retina_resolution, self.roh_max, self.roh_0)
        self.N_r = int(self.N_r)
        self.x_0 = int(img_size[0] // 2)
        self.y_0 = int(img_size[1] // 2)
        self.random_seed = random_seed

        self.retina_coordinates, self.fovea_mask = self.create_sampling_coordinates(self.N_r,
                                                                                    self.roh_0,
                                                                                    self.roh_max,
                                                                                    self.x_0,
                                                                                    self.y_0)

        # postprocessing of coordinates to get rid of artifacts
        count = 0
        while self.retina_coordinates[0, -1][0] <= self.retina_coordinates[0, -2][0] or self.retina_coordinates[0, -1][1] > self.retina_coordinates[0, -2][1]:
            count += 1
            self.retina_coordinates = self.retina_coordinates[1:-1, 1:-1, :]
            self.fovea_mask = self.fovea_mask[1:-1, 1:-1, :]
        self.retina_coordinates = self.retina_coordinates[:-count, :-count]
        self.fovea_mask = self.fovea_mask[:-count, :-count]

        self.retina_coordinates = (self.retina_coordinates * 2 - img_size[0]) / img_size[0]  # between -1 and 1 because gridsampler requires that
        self.retina_coordinates = self.retina_coordinates.contiguous()
        self.img_target_size = retina_size
        self.img_size = img_size
        self.jitter_type = jitter_type
        self.jitter_amount = int(jitter_amount)

    def get_fcg_coordinates(self, x, y, N_r, roh_0, roh_max, x_0, y_0, inverse=True):
        """
        Given a coordinate pair (x,y) in the original image,
        this function computes the foveal cartesian geometry coordinates mapped back to
        cartesian coordinates on the image, given a number of rings N_r, the half width of the fovea roh_0,
        the half-size of the image roh_max, and the image center x_0, y_0).

        """
        self.x_0 = x_0
        self.y_0 = y_0
        # Preparation
        self.a = np.exp((np.log(roh_max / roh_0) / N_r))
        self.r = 0
        self.Gamma = np.empty((self.N_r))
        self.Gamma[0] = self.roh_0

        self.roh = np.empty((self.N_r))
        self.roh[0] = self.roh_0
        self.Ring = np.empty((self.N_r))
        self.Ring[0] = self.r
        for i in range(1, self.N_r):
            self.roh[i] = np.floor(self.roh_0 * self.a ** i)
            if (self.roh[i] < self.Gamma[self.r]) or (self.roh[i] > self.Gamma[self.r]):
                self.r = self.r + 1
                self.Gamma[self.r] = self.roh[i]
            self.Ring[i] = self.r
        self.foveaHeight = self.foveaWidth = 2 * (self.roh_0 + self.N_r - 1) + 1
        self.fcx = self.fcy = np.floor(self.foveaWidth / 2)

        def mapping(x, y):
            x = x - self.x_0
            y = y - self.y_0
            r = max(np.abs(x), np.abs(y))
            if r <= roh_0:
                x_prime = x + self.fcx
                y_prime = y + self.fcy
            else:
                xi = int(np.floor(np.log(r / roh_0) / np.log(self.a)))
                xi = min(xi, len(self.Ring) - 1)
                delta_roh_xi = (roh_0 + self.Ring[xi]) / (np.floor(roh_0 * self.a ** xi))
                x_prime = np.floor(x * delta_roh_xi + self.fcx)
                y_prime = np.floor(y * delta_roh_xi + self.fcy)
            return x_prime, y_prime

        def inverse_mapping(x_prime, y_prime):
            x_prime = x_prime - self.fcx
            y_prime = y_prime - self.fcy
            roh = max(np.abs(x_prime), np.abs(y_prime))
            if roh <= self.roh_0:
                x = x_prime + self.x_0
                y = y_prime + self.y_0
            else:
                index = int(roh - self.roh_0)
                index = min(index, len(self.Gamma) - 1)
                delta_roh_Gamma = self.Gamma[index] / roh
                x = np.floor(x_prime * delta_roh_Gamma + self.x_0)
                y = np.floor(y_prime * delta_roh_Gamma + self.y_0)
            return x, y

        if inverse:
            x, y = inverse_mapping(x, y)

            return (x, y)
        else:
            x_prime, y_prime = mapping(x, y)
            return (x_prime, y_prime)

    def create_sampling_coordinates(self, N_r, roh_0, roh_max, x_0, y_0):
        """
        Given a number of rings N_r, a fovea radius roh_0, an image radius roh_max, and central fixation coordinates
        x_0 and y_0 (==roh_max)
        """
        assert y_0 == roh_max, "fixation coordinates are expected to be in the center"

        self.a = np.exp((np.log(self.roh_max / self.roh_0) / self.N_r))
        self.max_delta_roh = (self.roh_0 + self.N_r) / np.floor((self.a ** (self.N_r - 1)) * self.roh_0)
        self.fcx = self.fcy = np.floor((2 * (self.roh_0 + self.N_r) + 1) / 2)
        self.max_x_prime = int(np.floor(self.roh_max * self.max_delta_roh + self.fcx))
        self.max_y_prime = int(np.floor(self.roh_max * self.max_delta_roh + self.fcy))

        new_coordinates = np.empty((self.max_x_prime, self.max_y_prime, 2))
        for i in range(0, self.max_x_prime):
            for j in range(0, self.max_y_prime):
                x, y = self.get_fcg_coordinates(i, j, N_r, roh_0, roh_max, x_0, y_0, inverse=True)
                new_coordinates[i, j, 0] = x
                new_coordinates[i, j, 1] = y

        # create a fovea mask (with ones where the fovea is not), used to add irregularity to peripheral cone locations
        fovea_mask = np.ones((self.max_x_prime, self.max_y_prime, 2))
        fovea_mask[self.max_x_prime // 2 - self.roh_0:self.max_x_prime // 2 + self.roh_0, self.max_y_prime // 2 - self.roh_0:self.max_y_prime // 2 + self.roh_0, :] = 0.
        fovea_mask = torch.Tensor(fovea_mask)
        return torch.Tensor(new_coordinates[:, :, ::-1].astype(np.float32)), fovea_mask

    def add_jitter(self, retina_warp_coordinates, fovea_mask, jitter_amount=0.0, jitter_type="gaussian"):
        """
        adds uniform jitter to peripheral area to imitate the irregular sampling properties of the retina.

        Expects the unshifted warping coordinates and a fovea mask with ones everywhere except in the central fovea.
        """
        keep_identical_mask = fovea_mask.int()
        if jitter_type == "uniform":
            jitter_amount = int(jitter_amount)
            torch.manual_seed(RANDOM_SEED)  # Set the random seed for reproducibility
            jitter = torch.round((torch.rand(retina_warp_coordinates.shape) * 2 - 1) * jitter_amount) * keep_identical_mask
            jitter = jitter * 2 / self.img_size[0]
        elif jitter_type == "gaussian":
            torch.manual_seed(RANDOM_SEED)  # Set the random seed for reproducibility
            jitter = (torch.randn(retina_warp_coordinates.shape) * jitter_amount * keep_identical_mask.float()).int()
            jitter = jitter * 2 / self.img_size[0]

        return retina_warp_coordinates + jitter

    def warp_images(self, images, fixations, retina_warp_coordinates, fovea_mask,
                    jitter_amount=0.0, jitter_type="gaussian"):
        """
        Takes a batch of (a time series of) images and a batch of (a sequence of) fixations, as well as the
        retina warp coordinates, pointing to locations on the original image (or beyond, in which case zeros are sampled),
        and samples a foveal transformed batch of image(s) from the original image(s).

        retina_warp_coordinates is expected to not have a batch dimension, so (x_dim, y_dim, 2)
        """
        fixations = fixations.float().cpu()
        batch_size = images.shape[0]
        # merge batch and time dimension
        if len(fixations.shape) > 2:
            # if there is only one image but a sequence of fixations, repeat the image
            if not len(images.shape) > 4:
                images = torch.unsqueeze(images, axis=1).repeat(1, 1, fixations.shape[1], 1, 1)  # B, C, T, H, W
            time_steps = images.shape[1]
            height = images.shape[2]
            width = images.shape[3]
            images = torch.reshape(images, (batch_size * time_steps, images.shape[4], height, width))
            no_time_dim = False
        else:
            height = images.shape[1]
            width = images.shape[2]
            time_steps = 1
            no_time_dim = True

        retina_warp_coordinates = retina_warp_coordinates.repeat(batch_size * time_steps, 1, 1, 1)  # B*T, h,w,2

        # add jitter while still assuming central fixation
        if jitter_amount > 0.0:
            retina_warp_coordinates = self.add_jitter(retina_warp_coordinates,
                                                      self.fovea_mask,
                                                      self.jitter_amount,
                                                      self.jitter_type)
        if not no_time_dim:
            fixations = torch.reshape(fixations, (batch_size * time_steps, 2))

        x = fixations[:, 0].unsqueeze(1).unsqueeze(1)
        y = fixations[:, 1].unsqueeze(1).unsqueeze(1)
        shifted_retina_warp_coordinates = retina_warp_coordinates
        # shift the retina warp coordinates to the fixation location
        shifted_retina_warp_coordinates[:, :, :, 0] = retina_warp_coordinates[:, :, :, 0] + x
        shifted_retina_warp_coordinates[:, :, :, 1] = retina_warp_coordinates[:, :, :, 1] + y
        # warp the images
        images = images.to("cpu")
        shifted_retina_warp_coordinates = shifted_retina_warp_coordinates.to("cpu")

        warped_images = torch.nn.functional.grid_sample(input=images, grid=shifted_retina_warp_coordinates).to(self.device).contiguous()  # tfa.image.resampler(data=images, warp=shifted_retina_warp_coordinates)

        shifted_retina_coordinates_batch = shifted_retina_warp_coordinates.permute(0, 3, 1, 2).contiguous().to(self.device)

        # which pixels are outside the image? (everything bigger +-1)
        outside_image = (shifted_retina_coordinates_batch[:, 0, :, :] < -1) | (shifted_retina_coordinates_batch[:, 0, :, :] > 1) | (shifted_retina_coordinates_batch[:, 1, :, :] < -1) | (shifted_retina_coordinates_batch[:, 1, :, :] > 1)

        # set the outside pixels to gray # check the image value range to kow how to colour gray
        warped_images[outside_image.repeat(1, 3, 1, 1)] = 0.5
        # set the outside pixels to gray

        # import pdb; pdb.set_trace()
        # add the shifted retina coordinates to the image
        warped_images = torch.concat([warped_images, shifted_retina_coordinates_batch], dim=1)
        if no_time_dim:
            return warped_images

        else:
            # restore time dimension from merged batch*time dimension
            return torch.reshape(warped_images, (batch_size,
                                                 warped_images.shape[-3],
                                                 time_steps,
                                                 warped_images.shape[-2],
                                                 warped_images.shape[-1]))

    def find_number_of_rings(self, desired_resolution, roh_max, roh_0):
        """
        Computes the closest achievable resolution for the FCG sampling, given an image size
        and a chosen foveal radius.
        """
        max_x_prime = 0
        N_r = roh_max - roh_0  # max number of rings is half number of pixels minus half fovea
        distance_shrinking = True
        distance = np.inf
        while distance_shrinking:
            N_r -= 1
            a = np.exp((np.log(roh_max / roh_0) / N_r))
            max_delta_roh = (roh_0 + N_r) / np.floor(a ** (N_r - 1) * roh_0)
            fcx = fcy = np.floor((2 * (roh_0 + N_r) + 1) / 2)
            max_x_prime = int(np.floor(roh_max * max_delta_roh + fcx))
            max_y_prime = int(np.floor(roh_max * max_delta_roh + fcy))

            new_distance = np.abs(max_x_prime - desired_resolution)

            distance_shrinking = new_distance < distance

            distance = new_distance

        return N_r, max_x_prime

    def forward(self, img_x, fixations):
        transformed_image = self.warp_images(img_x, fixations,
                                             self.retina_coordinates, self.fovea_mask,
                                             jitter_amount=self.jitter_amount,
                                             jitter_type=self.jitter_type)

        scaled_image = torchvision.transforms.functional.resize(transformed_image, (self.img_target_size, self.img_target_size))

        return scaled_image


def normalize_fixations(fixations, width, height):
    normalized_fixations = []
    for x, y in fixations:
        normalized_x = (x / width) * 2 - 1
        normalized_y = (y / height) * 2 - 1
        normalized_fixations.append((normalized_x, normalized_y))
    return normalized_fixations


def unnormalize_fixations(fixations, width, height):
    unnormalized_fixations = []
    for x, y in fixations:
        unnormalized_x = ((x + 1) / 2) * width
        unnormalized_y = ((y + 1) / 2) * height
        unnormalized_fixations.append((unnormalized_x, unnormalized_y))
    return unnormalized_fixations


def get_fixation_history(fixation_coordinates, model):
    """
    Get the fixation history based on the model's included fixations.
    """
    history = []
    for index in model.included_fixations:
        try:
            history.append(fixation_coordinates[index])
        except IndexError:
            history.append(np.nan)
    return history


def rescale_centerbias(centerbias_template, height, width):
    """
    Rescale the centerbias template to match the image size.
    """
    centerbias = zoom(centerbias_template, (height / centerbias_template.shape[0], width / centerbias_template.shape[1]), order=0, mode='nearest')
    centerbias -= logsumexp(centerbias)
    assert centerbias.shape == (height, width), f"Centerbias shape {centerbias.shape} does not match image shape ({height}, {width})"
    return centerbias


def predict_fixation(model, image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor, rst):
    """
    Predict the next fixation point using the model.
    """
    log_density_prediction = model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)
    logD = log_density_prediction.detach().cpu().numpy()[0, 0]
    next_x, next_y = sample_from_logdensity(logD, rst=rst)

    # Get the height and width of the image tensor
    _, _, height, width = image_tensor.shape

    # Check if the predicted fixation is within the image bounds
    assert 0 <= next_x < width, f"Predicted fixation x-coordinate {next_x} is outside the image bounds (width: {width})"
    assert 0 <= next_y < height, f"Predicted fixation y-coordinate {next_y} is outside the image bounds (height: {height})"

    return next_x, next_y


def generate_retina_warps(image, num_fixations, model):
    height, width = image.shape[:2]
    centerbias = rescale_centerbias(centerbias_template, height, width)
    image_tensor = torch.tensor([image.transpose(2, 0, 1)]).to(DEVICE)
    image_tensor = image_tensor / 255.0

    centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)

    fixation_history_x = [width // 2]
    fixation_history_y = [height // 2]
    normalized_fixation_history = [(0, 0)]  # Store normalized fixations

    rst = np.random.RandomState(seed=RANDOM_SEED)

    foveal_transform = FovealTransform(fovea_size=FOVEA_SIZE,
                                       img_target_size=IMG_TARGET_SIZE,
                                       img_size=(height, width),
                                       jitter_type=JITTER_TYPE,
                                       jitter_amount=JITTER_AMOUNT,
                                       device=DEVICE,
                                       random_seed=RANDOM_SEED)

    retina_warps = []
    for _ in range(num_fixations - 1):
        x_hist = get_fixation_history(fixation_history_x, model)
        y_hist = get_fixation_history(fixation_history_y, model)
        x_hist_tensor = torch.tensor([x_hist]).to(DEVICE)
        y_hist_tensor = torch.tensor([y_hist]).to(DEVICE)
        next_x, next_y = predict_fixation(model, image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor, rst)

        normalized_next_fixation = normalize_fixations([(next_x, next_y)], width, height)
        normalized_next_x, normalized_next_y = normalized_next_fixation[0]

        fixation_history_x.append(next_x)
        fixation_history_y.append(next_y)
        normalized_fixation_history.append((normalized_next_x, normalized_next_y))

        fixations = torch.tensor([[normalized_next_x, normalized_next_y]]).to(DEVICE)

        retina_img = foveal_transform(image_tensor.float(), fixations)[0].permute(1, 2, 0).cpu().numpy()

        retina_warps.append(retina_img)

    return image, retina_warps, fixation_history_x, fixation_history_y


def save_to_h5(original_image, retina_warps, fixation_history_x, fixation_history_y, output_path, img_id):
    with h5py.File(output_path, 'a') as f:  # 'a' mode for appending data
        grp = f.create_group(str(img_id))
        grp.create_dataset('original_image', data=original_image)
        grp.create_dataset('retina_warps', data=np.array(retina_warps))
        grp.create_dataset('fixation_history_x', data=np.array(fixation_history_x))
        grp.create_dataset('fixation_history_y', data=np.array(fixation_history_y))


def process_image(img_info, num_fixations, model):
    img_path = os.path.join(COCO_DATASET_DIR, img_info['file_name'])
    image = cv2.imread(img_path)

    if image is None:
        print(f"Failed to load image: {img_path}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    original_image, retina_warps, fixation_history_x, fixation_history_y = generate_retina_warps(image, num_fixations, model)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    save_to_h5(original_image, retina_warps, fixation_history_x, fixation_history_y, output_path, img_info["id"])

def save_processed_ids(processed_ids, processed_ids_file):
    with open(processed_ids_file, 'w') as f:
        f.write(' '.join(map(str, processed_ids)))


def load_processed_ids(processed_ids_file):
    if os.path.exists(processed_ids_file):
        with open(processed_ids_file, 'r') as f:
            return set(map(int, f.read().split()))
    return set()


# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Constants
PROCESSED_TRAIN_IDS_FILE = 'processed_train_ids.txt'
PROCESSED_VAL_IDS_FILE = 'processed_val_ids.txt'

# Load the processed image IDs
processed_train_ids = load_processed_ids(PROCESSED_TRAIN_IDS_FILE)
processed_val_ids = load_processed_ids(PROCESSED_VAL_IDS_FILE)

# Generate retina warps for the training set
# Generate retina warps for the training set
train_img_ids = coco_train.getImgIds()
print("Number of training images:", len(train_img_ids))
for img_id in tqdm(train_img_ids, desc="Processing training images"):
    if img_id in processed_train_ids:
        tqdm.write(f"Skipping already processed training image: {img_id}")
        continue
    img_info = coco_train.loadImgs(img_id)[0]
    process_image(img_info, NUM_FIXATIONS_TRAIN, model)
    processed_train_ids.add(img_id)
    save_processed_ids(processed_train_ids, PROCESSED_TRAIN_IDS_FILE)
    tqdm.write(f"Processed training image: {img_id}")