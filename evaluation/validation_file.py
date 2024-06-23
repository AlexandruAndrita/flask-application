import torch
from prep_dataset.helpers import *


def validation(cnn_model, validation_dataset, criterion, device):
    cnn_model.eval()
    total_loss = 0
    with torch.no_grad():
        for i,batch in enumerate(validation_dataset):
            pixelated_image_batch, known_array_batch, target_array_list, _ = batch

            pixelated_image_batch = pixelated_image_batch.to(device)
            known_array_batch = known_array_batch.to(device, dtype=torch.bool)

            batch_loss = 0
            for j in range(pixelated_image_batch.size(0)):
                pixelated_image = pixelated_image_batch[j].unsqueeze(0)
                known_array = known_array_batch[j].unsqueeze(0)
                target_array = target_array_list[j].to(device)

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

                if torch.isnan(output).any() or torch.isinf(output).any():
                    print(f"Batch {i}, Item {j}, output contains NaN or Inf values")

            total_loss += batch_loss / pixelated_image_batch.size(0)

    return total_loss / len(validation_dataset)
