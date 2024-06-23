import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt

from group_dataset_images.prepare_datasets import get_images
from evaluation.model_file import CNNModel, CNNEncDecModel
from evaluation.train_file import train
from evaluation.validation_file import validation
from evaluation.test_file import test


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn_enc_dec_model = CNNEncDecModel()
    cnn_model = CNNModel()

    try:
        cnn_model.load_state_dict(torch.load("CNNModel.pth"))
    except Exception as e:
        # as we use this for training purposes only, it is safe to ignore the exception
        # if the .pth file does not exist, then one will be created once the model is trained
        pass

    try:
        cnn_enc_dec_model.load_state_dict(torch.load("CNNDecEndModel.pth"))
    except Exception as e:
        # as we use this for training purposes only, it is safe to ignore the exception
        # if the .pth file does not exist, then one will be created once the model is trained
        pass


    cnn_enc_dec_model.to(dtype=torch.float64, device=device)
    optimizer = torch.optim.Adam(cnn_enc_dec_model.parameters(),lr=0.001)
    criterion = nn.MSELoss()

    input_directory_path = "D:\\an III\\bachelor's thesis\\resized_images_folders\\folder8"
    train_dataset,validation_dataset,test_dataset = get_images(input_directory_path)

    print(f"Train dataset size: {len(train_dataset.sampler)}")
    print(f"Validation dataset size: {len(validation_dataset.sampler)}")
    print(f"Test dataset size: {len(test_dataset.sampler)}")

    num_epochs = 5
    losses_train = []
    losses_validation = []
    for epoch in range(num_epochs):
        loss_train = train(cnn_enc_dec_model, train_dataset, optimizer, criterion, device)
        loss_validation = validation(cnn_enc_dec_model, validation_dataset, criterion, device)

        losses_train.append(loss_train)
        losses_validation.append(loss_validation)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {loss_train} - Validation Loss: {loss_validation}")

    plt.plot(range(1,num_epochs+1),losses_train,label="Train Loss")
    plt.plot(range(1,num_epochs+1),losses_validation, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    loss_test = test(cnn_enc_dec_model, test_dataset, criterion, device)
    print(f"Test Loss: {loss_test}")

    torch.save(cnn_enc_dec_model.state_dict(),"CNNDecEndModel.pth")
    print("Model saved successfully under the name: CNNDecEndModel.pth")
