import h5py
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def visualize_h5_data(h5_file, output_dir=None, plot=False, save_format='jpg'):
    with h5py.File(h5_file, 'r') as f:
        original_image = f['original_image'][:]
        retina_warps = f['retina_warps'][:]
        fixation_history_x = f['fixation_history_x'][:]
        fixation_history_y = f['fixation_history_y'][:]

    num_warps = len(retina_warps)

    if plot:
        fig, axes = plt.subplots(1, num_warps + 1, figsize=(20, 5))
        axes[0].imshow(original_image)
        axes[0].plot(fixation_history_y, fixation_history_x, 'ro-', markersize=5)
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

        original_image_path = os.path.join(output_dir, f'original_image.{save_format}')
        if save_format == 'jpg':
            cv2.imwrite(original_image_path, cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
        else:
            plt.imsave(original_image_path, original_image)

        for i in range(num_warps):
            warp_image = retina_warps[i, :, :, :3]  # Take only the first 3 dimensions as the image

            warp_image_path = os.path.join(output_dir, f'retina_warp_{i + 1}.{save_format}')
            if save_format == 'jpg':
                cv2.imwrite(warp_image_path, cv2.cvtColor((warp_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            else:
                plt.imsave(warp_image_path, warp_image)

visualize_h5_data("retina_warps/574769.h5", plot=True)