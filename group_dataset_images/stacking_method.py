import numpy as np
import torch


def get_max_height_width(batch_as_list: list):
    max_width = float("-inf")
    max_height = float("-inf")

    # (1,H,W)
    for image in batch_as_list:
        width=image[0].shape[2]
        height=image[0].shape[1]

        max_width=max(max_width,width)
        max_height=max(max_height,height)

    return max_height,max_width


def apply_padding(array,max_height,max_width,c_value):
    if array.shape[0]<max_height:
        padding_dimension_height=max_height-array.shape[0]
        array=np.pad(array,((0,padding_dimension_height),(0,0)),mode='constant',constant_values=c_value)

    if array.shape[1]<max_width:
        padding_dimension_width=max_width-array.shape[1]
        array=np.pad(array,((0,0),(0,padding_dimension_width)),mode='constant',constant_values=c_value)

    return array


def stack_with_padding(batch_as_list: list, max_height: int, max_width: int):
    # (pixelated_image, known_array, target_array, image_file)

    for i,image in enumerate(batch_as_list):
        pixelated_image,known_array,target_array,image_file=image

        original_height_pix_image=pixelated_image.shape[1]
        original_width_pix_image=pixelated_image.shape[2]

        original_height_known_array=known_array.shape[1]
        original_width_known_array=known_array.shape[2]

        pixelated_image=pixelated_image.reshape(original_height_pix_image,original_width_pix_image)
        known_array=known_array.reshape(original_height_known_array,original_width_known_array)

        pixelated_image=apply_padding(pixelated_image,max_height,max_width,0)
        known_array=apply_padding(known_array,max_height,max_width,1)

        pixelated_image=pixelated_image.reshape(1,max_height,max_width)
        known_array=known_array.reshape(1,max_height,max_width)

        batch_as_list[i]=(pixelated_image,known_array,target_array,image_file)

    stacked_pixelated_images=np.stack([image[0] for image in batch_as_list],axis=0)
    stacked_pixelated_images=torch.tensor(stacked_pixelated_images)

    stacked_known_arrays=np.stack([image[1] for image in batch_as_list],axis=0)
    stacked_known_arrays=torch.tensor(stacked_known_arrays)

    target_arrays=[torch.tensor(image[2]) for image in batch_as_list]

    image_files=[image[3] for image in batch_as_list]

    return stacked_pixelated_images,stacked_known_arrays,target_arrays,image_files
    #return stacked_pixelated_images, stacked_known_arrays, stacked_target_arrays, image_files

