import h5py
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random
import argparse


def unnormalize_fixations(fixations, width, height):
    unnormalized_fixations = []
    for x, y in fixations:
        unnormalized_x = (x * (width / 2)) + (width / 2)
        unnormalized_y = (y * (height / 2)) + (height / 2)
        unnormalized_fixations.append((unnormalized_x, unnormalized_y))
    return unnormalized_fixations


def visualize_h5_data(h5_file, img_id, output_dir=None, plot=False, save_format='jpg'):
    with h5py.File(h5_file, 'r') as f:
        grp = f[img_id]
        original_image = grp['original_image'][:]
        retina_warps = grp['retina_warps'][:]
        fixation_history_x = grp['fixation_history_x'][:]
        fixation_history_y = grp['fixation_history_y'][:]

        height, width = original_image.shape[:2]
        num_warps = len(retina_warps)

        # Fixation history (already in pixel coordinates)
        fixation_history = list(zip(fixation_history_x, fixation_history_y))

        if plot:
            fig, axes = plt.subplots(1, num_warps + 1, figsize=(20, 5))
            axes[0].imshow(original_image)
            axes[0].plot(fixation_history_x, fixation_history_y, 'ro-', markersize=5, label='Fixation History')
            axes[0].scatter([fixation_history_x[0]], [fixation_history_y[0]], color='orange', s=100, label='First Fixation')

            for i in range(num_warps):
                warp_image = retina_warps[i, :, :, :3]  # Take only the first 3 dimensions as the image
                warp_coordinates = retina_warps[i, :, :, 3:]  # Take the last 2 dimensions as coordinates

                # Extract the fixation coordinates from the warp
                warp_coords = warp_coordinates[0, 0, :]  # Take the (0, 0, :) entry for coordinates
                unnormalized_warp_coords = unnormalize_fixations([warp_coords], width, height)
                unnormalized_warp_x, unnormalized_warp_y = zip(*unnormalized_warp_coords)

                # Plot fixation from retina warp
                axes[0].scatter(unnormalized_warp_x, unnormalized_warp_y, color='blue', s=100, marker='x', label=f'Retina Warp {i + 1} Fixation')

                axes[i + 1].imshow(warp_image)
                axes[i + 1].scatter([warp_coords[1]], [warp_coords[0]], color='red', s=100, marker='o')
                axes[i + 1].set_title(f'Retina Warp {i + 1}')
                axes[i + 1].axis('off')

            # Connect the fixations with lines
            for j in range(1, len(fixation_history)):
                axes[0].plot([fixation_history[j - 1][0], fixation_history[j][0]],
                             [fixation_history[j - 1][1], fixation_history[j][1]],
                             color='red')

            axes[0].set_title('Original Image with Fixations')
            axes[0].axis('off')
            plt.legend()
            plt.tight_layout()
            plt.show()

        if output_dir:
            img_output_dir = os.path.join(output_dir, img_id)
            os.makedirs(img_output_dir, exist_ok=True)

            # Save the original image with fixations and connected lines
            original_image_with_fixations = original_image.copy()
            for i, (x, y) in enumerate(fixation_history):
                color = (255, 165, 0) if i == 0 else (255, 0, 0)  # Orange for the first fixation, red for others
                size = 10 if i == 0 else 5  # Larger size for the first fixation
                cv2.circle(original_image_with_fixations, (int(x), int(y)), size, color, -1)
                if i > 0:
                    cv2.line(original_image_with_fixations, (int(fixation_history[i - 1][0]), int(fixation_history[i - 1][1])),
                             (int(x), int(y)), (255, 0, 0), 1)  # Red line to connect fixations

            # Connect the fixations from retina warps with lines and circles
            prev_x, prev_y = None, None
            for i in range(num_warps):
                color = (0, 165, 255) if i == 0 else (0, 0, 255)  # Orange for the first fixation, red for others
                warp_coordinates = retina_warps[i, :, :, 3:]  # Take the last 2 dimensions as coordinates
                warp_coords = warp_coordinates[0, 0, :]  # Take the (0, 0, :) entry for coordinates
                unnormalized_warp_coords = unnormalize_fixations([warp_coords], width, height)
                for x, y in unnormalized_warp_coords:
                    cv2.circle(original_image_with_fixations, (int(x), int(y)), 10, color, -1)  # Blue for retina warp fixations
                    if prev_x is not None and prev_y is not None:
                        cv2.line(original_image_with_fixations, (int(prev_x), int(prev_y)),
                                 (int(x), int(y)), (0, 0, 255), 2)  # Blue line to connect retina warp fixations
                    prev_x, prev_y = x, y

            original_image_path = os.path.join(img_output_dir, f'original_image.{save_format}')
            if save_format == 'jpg':
                cv2.imwrite(original_image_path, cv2.cvtColor(original_image_with_fixations, cv2.COLOR_RGB2BGR))
            else:
                plt.imsave(original_image_path, original_image_with_fixations)

            # Save the retina warps with fixation points
            for i in range(num_warps):
                warp_image = retina_warps[i, :, :, :3]  # Take only the first 3 dimensions as the image
                warp_image_with_fixations = (warp_image * 255).astype(np.uint8).copy()
                warp_coordinates = retina_warps[i, :, :, 3:]  # Take the last 2 dimensions as coordinates

                # Extract and plot the fixation coordinates
                warp_coords = warp_coordinates[0, 0, :]  # Take the (0, 0, :) entry for coordinates
                for y, x in [warp_coords]:
                    cv2.circle(warp_image_with_fixations, (int(x), int(y)), 5, (255, 0, 0), -1)
                warp_image_path = os.path.join(img_output_dir, f'retina_warp_{i + 1}.{save_format}')
                if save_format == 'jpg':
                    cv2.imwrite(warp_image_path, cv2.cvtColor(warp_image_with_fixations, cv2.COLOR_RGB2BGR))
                else:
                    plt.imsave(warp_image_path, warp_image_with_fixations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize H5 data with retina warps and fixation history.")
    parser.add_argument("--h5_file", type=str, default="retina_warps/retina_warps.h5", help="Path to the single h5 file containing all retina warps.")
    parser.add_argument("--num_examples", type=int, default=10, help="Number of random examples to visualize.")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Directory to save the visualizations.")
    parser.add_argument("--plot", action="store_true", help="Whether to plot the visualizations.")
    parser.add_argument("--save_format", type=str, default="jpg", choices=["jpg", "png"], help="Format to save the images.")

    args = parser.parse_args()

    with h5py.File(args.h5_file, 'r') as f:
        img_ids = list(f.keys())

    # Randomly select num_examples image IDs
    random_img_ids = random.sample(img_ids, args.num_examples)

    for img_id in random_img_ids:
        visualize_h5_data(args.h5_file, img_id, plot=args.plot, output_dir=args.output_dir, save_format=args.save_format)