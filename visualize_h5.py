import h5py
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random
import argparse


def visualize_h5_data(h5_file, output_dir=None, plot=False, save_format='jpg'):
    with h5py.File(h5_file, 'r') as f:
        original_image = f['original_image'][:]
        retina_warps = f['retina_warps'][:]
        fixation_history_x = f['fixation_history_x'][:]
        fixation_history_y = f['fixation_history_y'][:]

        height, width = original_image.shape[:2]
        num_warps = len(retina_warps)

        if plot:
            fig, axes = plt.subplots(1, num_warps + 1, figsize=(20, 5))
            axes[0].imshow(original_image)
            axes[0].plot(fixation_history_x, fixation_history_y, 'ro-', markersize=5)
            axes[0].set_title('Original Image with Fixations')
            axes[0].axis('off')

            for i in range(num_warps):
                warp_image = retina_warps[i, :, :, :3]  # Take only the first 3 dimensions as the image
                warp_coordinates = retina_warps[i, :, :, 3:]  # Take the last 2 dimensions as coordinates
                axes[i + 1].imshow(warp_image)
                axes[i + 1].plot(warp_coordinates[:, :, 1], warp_coordinates[:, :, 0], 'r.', markersize=1)
                axes[i + 1].set_title(f'Retina Warp {i + 1}')
                axes[i + 1].axis('off')

            plt.tight_layout()
            plt.show()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # Save the original image with fixations
            original_image_with_fixations = original_image.copy()
            for x, y in zip(fixation_history_x, fixation_history_y):
                cv2.circle(original_image_with_fixations, (int(x), int(y)), 5, (255, 0, 0), -1)
            original_image_path = os.path.join(output_dir, f'original_image.{save_format}')
            if save_format == 'jpg':
                cv2.imwrite(original_image_path, cv2.cvtColor(original_image_with_fixations, cv2.COLOR_RGB2BGR))
            else:
                plt.imsave(original_image_path, original_image_with_fixations)

            # Save the retina warps with fixation points
            for i in range(num_warps):
                warp_image = retina_warps[i, :, :, :3]  # Take only the first 3 dimensions as the image
                warp_image_with_fixations = (warp_image * 255).astype(np.uint8).copy()
                warp_coordinates = retina_warps[i, :, :, 3:]  # Take the last 2 dimensions as coordinates
                for y, x in zip(warp_coordinates[:, :, 0].flatten(), warp_coordinates[:, :, 1].flatten()):
                    cv2.circle(warp_image_with_fixations, (int(x), int(y)), 1, (255, 0, 0), -1)
                warp_image_path = os.path.join(output_dir, f'retina_warp_{i + 1}.{save_format}')
                if save_format == 'jpg':
                    cv2.imwrite(warp_image_path, cv2.cvtColor(warp_image_with_fixations, cv2.COLOR_RGB2BGR))
                else:
                    plt.imsave(warp_image_path, warp_image_with_fixations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize H5 data with retina warps and fixation history.")
    parser.add_argument("h5_directory", type=str, help="Directory containing the h5 files.")
    parser.add_argument("--num_examples", type=int, default=10, help="Number of random examples to visualize.")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Directory to save the visualizations.")
    parser.add_argument("--plot", action="store_true", help="Whether to plot the visualizations.")
    parser.add_argument("--save_format", type=str, default="jpg", choices=["jpg", "png"], help="Format to save the images.")

    args = parser.parse_args()

    # Get a list of all h5 files in the directory
    h5_files = [file for file in os.listdir(args.h5_directory) if file.endswith('.h5')]

    # Randomly select num_examples files
    random_files = random.sample(h5_files, args.num_examples)

    for file in random_files:
        h5_file_path = os.path.join(args.h5_directory, file)
        visualize_h5_data(h5_file_path, plot=args.plot, output_dir=args.output_dir, save_format=args.save_format)
