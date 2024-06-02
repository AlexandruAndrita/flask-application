import torch
from prep_dataset.helpers import *
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


def test(cnn_model, test_dataset, criterion, device):
    cnn_model.eval()
    total_loss = 0
    folder_path = "D:\\an III\\bachelor's thesis\\pixelated_images"
    with torch.no_grad():
        for i,batch in enumerate(test_dataset.dataset):
            """
            for i in batch:
                input, mask, target, path
                input = batch[0][i]
                mask = batch[1][i]
                target = batch[2][i]
                path = batch[3][i]
            """
            pixelated_image = torch.from_numpy(batch[0])
            known_array = torch.from_numpy(batch[1])
            target_array = torch.from_numpy(batch[2])
            image_path = batch[3]

            pixelated_image = pixelated_image.to(device)
            target_array = target_array.to(device)

            # normalizing targets
            target_array_normalized = normalize_targets(target_array)
            pixelated_image = pixelated_image.reshape(1, pixelated_image.shape[0], pixelated_image.shape[1],pixelated_image.shape[2])

            # normalizing inputs (pixelated image)
            pixelated_image_normalized = normalize_targets(pixelated_image)
            pixelated_image_normalized = pixelated_image_normalized.reshape(1, pixelated_image_normalized.shape[2],pixelated_image_normalized.shape[3])

            output = cnn_model(pixelated_image_normalized)
            output=output.reshape(1,output.shape[1],output.shape[2])
            # normalizing output of the model
            output_normalized = normalize_targets(output)

            known_array = known_array.to(dtype=torch.bool)
            crop = output_normalized[~known_array]
            crop_reshaped = crop.reshape(target_array_normalized.shape)

            loss = criterion(crop_reshaped, target_array_normalized)
            total_loss += loss.item()

            # Comparing the original picture with the one provided by the model
            pil_before_model = transforms.ToPILImage()(pixelated_image_normalized.cpu().detach())
            pixelated_image_normalized[~known_array] = crop
            pil_image_model = transforms.ToPILImage()(pixelated_image_normalized.cpu().detach())
            image_path = Image.open(image_path)

            fig, axs = plt.subplots(1, 3, figsize=(10, 5))

            axs[0].imshow(image_path)
            axs[0].set_title("Original Image")
            axs[0].axis("off")

            axs[1].imshow(pil_before_model,cmap="gray")
            axs[1].set_title("Pixelated Image")
            axs[1].axis("off")

            axs[2].imshow(pil_image_model,cmap="gray")
            axs[2].set_title("Model Output")
            axs[2].axis("off")

            plt.tight_layout()
            plt.show()

    return total_loss / len(test_dataset)
