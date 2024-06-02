import numpy as np


def data_validation(image: np.ndarray,x:int,y:int,width:int,height:int,size:int):
    if len(image.shape)!=3:
        raise ValueError("Image is not 3D.")
    if image.shape[0]!=1:
        raise ValueError(f"Channel size is {image.shape[0]}, instead of 1.")
    if width<2 or height<2 or size<2:
        raise ValueError("Width, height or size are smaller than 2.")
    if x<0 or x+width>image.shape[2]: # x + width > width
        raise ValueError("Starting point - x - is invalid. The pixelated area would exceed the image width.")
    if y<0 or y+height>image.shape[1]: # y + height > height
        raise ValueError("Starting point - y - is invalid. The pixelated area would exceed the image height.")


def change_pixels(i,j,f_pixelated_image,f_known_array,size_i,size_j):
    values = f_pixelated_image[i:i + size_i, j:j + size_j]
    # if len(values) != 0:
    if values.shape[0] != 0 and values.shape[1] != 0:
        mean = np.mean(values)
        f_pixelated_image[i:i + size_i, j:j + size_j] = mean
        f_known_array[i:i + size_i, j:j + size_j] = False
    return f_pixelated_image,f_known_array


"""
pixelated_image -> returns (1,H,W)
known_array -> returns (1,H,W)
target_array -> returns (1,height,width)
"""
def prepare_image(image: np.ndarray,
                  x: int,
                  y: int,
                  width: int,
                  height: int,
                  size: int)-> tuple[np.ndarray,np.ndarray,np.ndarray]:
    data_validation(image,x,y,width,height,size)
    original_height=image.shape[1]
    original_width=image.shape[2]

    pixelated_image=np.copy(image).reshape(original_height,original_width)
    known_array=np.ones_like(image).reshape(original_height,original_width)
    target_array=np.copy(image).reshape(original_height,original_width)

    i=y
    input_size=size
    while i<=y+height:
        j=x
        while j<=x+width:
            if i+size>y+height and j+size>x+width:
                size_i=y+height-i
                size_j=x+width-j
                pixelated_image,known_array=change_pixels(i,j,pixelated_image,known_array,size_i,size_j)
            elif i+size>y+height:
                size_i=y+height-i
                pixelated_image, known_array = change_pixels(i, j, pixelated_image, known_array, size_i, size)
            elif j+size>x+width:
                size_j=x+width-j
                pixelated_image, known_array = change_pixels(i, j, pixelated_image, known_array, size, size_j)
            else:
                pixelated_image, known_array = change_pixels(i, j, pixelated_image, known_array, size, size)

            j+=input_size
        i+=input_size

    pixelated_image=pixelated_image.reshape(1,original_height,original_width)
    known_array=known_array.reshape(1,original_height,original_width)
    target_array=target_array[y:y+height,x:x+width]

    target_array=target_array.reshape(1,target_array.shape[0],target_array.shape[1])

    return pixelated_image,known_array,target_array
