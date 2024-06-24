import torch
from prep_dataset.helpers import *
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def test(cnn_model, test_dataset, criterion, device):
    cnn_model.eval()
    total_loss = 0
    with torch.no_grad():
        for i,batch in enumerate(test_dataset):
            pixelated_image_batch, known_array_batch, target_array_list, image_paths = batch

            pixelated_image_batch = pixelated_image_batch.to(device)
            known_array_batch = known_array_batch.to(device, dtype=torch.bool)

            batch_loss = 0
            for j in range(pixelated_image_batch.size(0)):
                pixelated_image = pixelated_image_batch[j].unsqueeze(0)
                known_array = known_array_batch[j].unsqueeze(0)
                target_array = target_array_list[j].to(device)
                image_path = image_paths[j]

                max_value = pixelated_image.max().item()
                min_value = pixelated_image.min().item()

                # normalization of inputs and targets
                pixelated_image_normalized = apply_normalization(pixelated_image, max_value, min_value)
                target_array_normalized = apply_normalization(target_array, max_value, min_value)

                output = cnn_model(pixelated_image_normalized)

                mask = ~known_array
                masked_output = output[mask]

                masked_output = masked_output.view(target_array_normalized.shape)

                loss = criterion(masked_output, target_array_normalized)
                total_loss += loss.item()

                total_loss += batch_loss / pixelated_image_batch.size(0)

                denormalized_output = denormalize(output.cpu().detach().numpy(), max_value, min_value)

                if np.isnan(denormalized_output).any() or np.isinf(denormalized_output).any():
                    print(f"NaN or Inf detected in denormalized output at batch {i}, item {j}")
                    continue

                denormalized_output = np.clip(denormalized_output, 0, 255).astype(np.uint8)

                pil_before_model = transforms.ToPILImage()(pixelated_image_normalized.squeeze(0).squeeze(0).cpu().detach())
                reconstructed_image = pixelated_image_normalized.clone()
                denormalized_output = torch.tensor(denormalized_output, dtype=reconstructed_image.dtype, device=reconstructed_image.device)
                denormalized_output_masked = denormalized_output[~mask].view(reconstructed_image[~mask].shape)
                reconstructed_image[~mask] = denormalized_output_masked

                pil_image_model = transforms.ToPILImage()(reconstructed_image.squeeze(0).squeeze(0).cpu().detach())

                output_no_mask = transforms.ToPILImage()(output.squeeze(0).squeeze(0).cpu().detach())

                try:
                    image_path = Image.open(image_path)
                except Exception as e:
                    raise Exception(f"Image {image_path} is already opened or {image_path} does not exist. Exception: {e}")

                # useful plot for general comparison between input and outputs
                fig, axs = plt.subplots(2, 2, figsize=(10, 5))

                axs[0,0].imshow(image_path)
                axs[0,0].set_title("Original Image")
                axs[0,0].axis("off")

                axs[0,1].imshow(pil_before_model,cmap="gray")
                axs[0,1].set_title("Pixelated Image")
                axs[0,1].axis("off")

                axs[1,0].imshow(pil_image_model,cmap="gray")
                axs[1,0].set_title("Model Output with Mask")
                axs[1,0].axis("off")

                axs[1, 1].imshow(output_no_mask, cmap="gray")
                axs[1, 1].set_title("Model Output no Mask")
                axs[1, 1].axis("off")

                plt.tight_layout()
                plt.suptitle("Original vs. Pixelated vs. ML Model")
                plt.show()

    return total_loss / len(test_dataset)
