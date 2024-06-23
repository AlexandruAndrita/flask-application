from group_dataset_images.stacking_method import stack_with_padding
from group_dataset_images import group_images
from torch.utils.data import DataLoader


def stack_as_collate(batch):
    max_height = float("-inf")
    max_width = float("-inf")
    for item in batch:
        _,height,width = item[0].shape
        max_height = max(max_height, height)
        max_width = max(max_width, width)

    return stack_with_padding(batch, 128, 170)


def get_images(input_directory):
    images = group_images.RandomImagePixelationDataset(input_directory,
                                            width_range=(4, 32),
                                            height_range=(4, 32),
                                            size_range=(4, 16))

    """
    up until this point, images have the maximum height and width in each batch, they are grayscale and prepared for training
    """

    images_with_path = list()
    image_without_path = list()

    for i in range(len(images)):
        pixelated_image, known_array,target_array,image_path = images[i]
        images_with_path.append((pixelated_image,known_array,target_array,image_path))
        image_without_path.append((pixelated_image,known_array,target_array))

    test_set_length = int(len(images)*0.1)
    validation_set_length = int(len(images)*0.2)

    test_data = group_images.RandomImagePixelationDataset(input_directory,
                                            width_range=(4, 32),
                                            height_range=(4, 32),
                                            size_range=(4, 16))

    validation_data = group_images.RandomImagePixelationDataset(input_directory,
                                                                width_range=(4, 32),
                                                                height_range=(4, 32),
                                                                size_range=(4,16))

    training_data = group_images.RandomImagePixelationDataset(input_directory,
                                                            width_range=(4, 32),
                                                            height_range=(4, 32),
                                                            size_range=(4,16))

    test_data.filenames = [item[-1] for item in images_with_path[len(images)-test_set_length:len(images)]]
    validation_data.filenames = [item[-1] for item in images_with_path[len(images)-test_set_length-validation_set_length:
                                                len(images)-test_set_length]]
    training_data.filenames = [item[-1] for item in images_with_path[:len(images)-test_set_length-validation_set_length]]

    # max_height = max(test_data.max_height, validation_data.max_height, training_data.max_height)
    # max_width = max(test_data.max_width, validation_data.max_width, training_data.max_width)

    train_dataset = DataLoader(dataset=training_data,shuffle=True,batch_size=32,collate_fn=stack_as_collate)
    validation_dataset = DataLoader(dataset=validation_data,shuffle=False,batch_size=32,collate_fn=stack_as_collate)
    test_dataset = DataLoader(dataset=test_data,shuffle=False,batch_size=32,collate_fn=stack_as_collate)

    return train_dataset,validation_dataset,test_dataset


if __name__ == "__main__":
    input_directory_path = "D:\\an III\\bachelor's thesis\\thesis\dataset\\test_first_stage"
    _, _, _ = get_images(input_directory_path)
