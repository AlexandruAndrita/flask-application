from group_dataset_images import group_images
from group_dataset_images import stacking_method
from prep_dataset import image_validation
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader


if __name__ == '__main__':

    """
    testing the dataset selection -- no stacking, no padding
    """
    # ds = group_images.RandomImagePixelationDataset(
    #     r"D:\an III\bachelor's thesis\thesis\output_test_images_validation\test_batch_2",
    #     width_range=(50, 300),
    #     height_range=(50, 300),
    #     size_range=(10, 50)
    # )
    #
    # for pixelated_image, known_array, target_array, image_file in ds:
    #     fig, axes = plt.subplots(ncols=3)
    #     axes[0].imshow(pixelated_image[0], cmap="gray", vmin=0, vmax=255)
    #     axes[0].set_title("pixelated_image")
    #     axes[1].imshow(known_array[0], cmap="gray", vmin=0, vmax=1)
    #     axes[1].set_title("known_array")
    #     axes[2].imshow(target_array[0], cmap="gray", vmin=0, vmax=255)
    #     axes[2].set_title("target_array")
    #     fig.suptitle(os.path.basename(image_file))
    #     fig.tight_layout()
    #     plt.show()

    """
    testing the dataset selection -- with stacking, with padding
    """
    ds = group_images.RandomImagePixelationDataset(
        r"D:\an III\bachelor's thesis\thesis\output_test_images_validation\test_batch_2",
        width_range=(50, 300),
        height_range=(50, 300),
        size_range=(10, 50)
    )

    output_dir = r"D:\an III\bachelor's thesis\thesis\stacked_images_example"
    index=0
    image_validation.delete_directory_content(output_dir)

    dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=stacking_method.stack_with_padding)
    for (stacked_pixelated_images, stacked_known_arrays, target_arrays, image_files) in dl:
        fig, axes = plt.subplots(nrows=dl.batch_size, ncols=3)
        for i in range(dl.batch_size):
            axes[i, 0].imshow(stacked_pixelated_images[i][0], cmap="gray", vmin=0, vmax=255)
            axes[i, 1].imshow(stacked_known_arrays[i][0], cmap="gray", vmin=0, vmax=1)
            axes[i, 2].imshow(target_arrays[i][0], cmap="gray", vmin=0, vmax=255)
        fig.tight_layout()

        figure_name = f"figure_stacked_{index}.jpg"
        path = os.path.join(output_dir, figure_name)
        fig.suptitle(figure_name)
        fig.savefig(path)

        index += 1
        plt.show()
