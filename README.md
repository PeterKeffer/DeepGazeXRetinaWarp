# DeepGaze RetinaWarps

This project generates retina warps for the MS COCO dataset using the DeepGaze model.

## Setup

1. Clone the repository:
git clone https://github.com/peterkeffer/retina-warps.git
cd retina-warps
Copy code
2. Create a Conda environment using the provided `environment.yml` file:
conda env create -f environment.yml
conda activate DeepGazeXRetinaWarp
Copy code
Alternatively, you can install the required packages using pip:
pip install -r requirements.txt
Copy code
3. Make sure you have the MS COCO dataset available at `/share/klab/datasets/ms_coco` on your HPC.

## Usage

1. Modify the hyperparameters in the `main.py` script if needed.

2. Run the script:
python main.py
Copy code
The retina warps will be saved in the `DeepGazeXRetinaWarp` directory in H5 format.

## Output

The script generates an H5 file for each image in the MS COCO dataset, containing:
- `original_image`: The original image.
- `retina_warps`: The generated retina warps.
- `fixation_history_x`: The x-coordinates of the fixation history.
- `fixation_history_y`: The y-coordinates of the fixation history.