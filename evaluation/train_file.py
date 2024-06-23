import torch
from prep_dataset.helpers import *
from torchvision import transforms
import os
import numpy as np
from PIL import Image


def train(cnn_model, train_dataset, optimizer, criterion, device):
    cnn_model.train()
    total_loss = 0
    # used with the purpose of saving images in order to visually analyze the results from the training process
    folder_path = "D:\\an III\\bachelor's thesis\\pixelated_images"
    for i,batch in enumerate(train_dataset):
        pixelated_image_batch, known_array_batch, target_array_list, _ = batch

        pixelated_image_batch = pixelated_image_batch.to(device)
        known_array_batch = known_array_batch.to(device, dtype=torch.bool)

        if torch.isnan(pixelated_image_batch).any() or torch.isinf(pixelated_image_batch).any():
            print(f"NaN or Inf detected in pixelated_image_batch at batch {i}")
            continue
        if torch.isnan(known_array_batch).any() or torch.isinf(known_array_batch).any():
            print(f"NaN or Inf detected in known_array_batch at batch {i}")
            continue

        batch_loss = 0
        for j in range(pixelated_image_batch.size(0)):
            pixelated_image = pixelated_image_batch[j].unsqueeze(0)  # (1, 1, 128, 170)
            known_array = known_array_batch[j].unsqueeze(0)  # (1, 1, 128, 170)
            target_array = target_array_list[j].to(device)

            max_value = pixelated_image.max().item()
            min_value = pixelated_image.min().item()

            # normalization of inputs and targets
            pixelated_image_normalized = apply_normalization(pixelated_image,max_value,min_value)
            target_array_normalized = apply_normalization(target_array, max_value, min_value)

            optimizer.zero_grad()  # resetting accumulated gradients
            output = cnn_model(pixelated_image_normalized)

            mask = ~known_array
            masked_output = output[mask]
            masked_output = masked_output.view(target_array_normalized.shape)

            loss = criterion(masked_output, target_array_normalized)
            batch_loss += loss.item()

            loss.backward()  # compute gradients
            optimizer.step()  # weight update

            print(f"Batch {i}, Item {j}: mean {output.mean().item()}")
            print(f"Batch {i}, Item {j}: masked_output min value {masked_output.min().item()}, max value {masked_output.max().item()}, mean {masked_output.mean().item()}")

            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"Batch {i}, Item {j}, output contains NaN or Inf values")
                continue
            else:
                pixelated_image_saved = transforms.ToPILImage()(pixelated_image_normalized.squeeze(0).squeeze(0).cpu())
                pixelated_image_saved.save(os.path.join(folder_path, f"pix_image_{i}_{j}.jpg"))

                target_array_saved = transforms.ToPILImage()(target_array_normalized.squeeze(0).cpu())
                target_array_saved.save(os.path.join(folder_path, f"target_array_{i}_{j}.jpg"))

                max_value = pixelated_image.max().item()
                min_value = pixelated_image.min().item()
                denormalized_output = denormalize(output.cpu().detach().numpy(), max_value, min_value)

                if np.isnan(denormalized_output).any() or np.isinf(denormalized_output).any():
                    print(f"NaN or Inf detected in denormalized output at batch {i}, item {j}")
                    continue

                denormalized_output = np.clip(denormalized_output, 0, 255).astype(np.uint8)

                print(f"Batch {i}, Item {j}: denormalized_output min value {denormalized_output.min()}, max value {denormalized_output.max()}, mean {denormalized_output.mean()}")

                output_normalized_saved = Image.fromarray(denormalized_output.squeeze().squeeze()).convert("L")
                output_normalized_saved.save(os.path.join(folder_path, f"output_denormalized_{i}_{j}.jpg"))

        total_loss += batch_loss / pixelated_image_batch.size(0)

    return total_loss / len(train_dataset)


