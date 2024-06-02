import os
from torch.utils.data import Dataset
import glob
from PIL import Image
import numpy as np
from typing import Optional

from prep_dataset import to_grayscale
from prep_dataset import prepare_image


class RandomImagePixelationDataset(Dataset):

    def __init__(
            self,
            image_dir,
            width_range: tuple[int,int],
            height_range: tuple[int,int],
            size_range: tuple[int,int],
            dtype: Optional[type] = None
    ):
        self.validate_input_directory(image_dir)
        self.image_dir = image_dir

        self.validate_inputs(width_range, height_range, size_range)
        self.width_range = width_range
        self.height_range = height_range
        self.size_range = size_range

        self.filenames = self.get_files()
        self.dtype = dtype

        self.max_height,self.max_width = self.get_max_height_width()

    def get_files(self):
        self.image_dir = os.path.abspath(self.image_dir)
        files = glob.glob(os.path.join(self.image_dir, "**", "*"), recursive=True)
        filenames = list()
        for f in files:
            if os.path.isfile(f):
                img = Image.open(f)
                width, height = img.size
                if not ((height == 128 and width == 170) or (height == 170 and width == 128)):
                    print("Error: image '" + f + "' does not have the right size")
                    # raise ValueError("Error: image '" + f + "' does not have the right size")
                else:
                    name, extension = os.path.splitext(f)
                    if extension in [".jpg", ".jpeg", ".JPG", ".JPEG"]:
                        filenames.append(f)
        filenames.sort()

        return filenames

    def get_max_height_width(self):
        max_height = float("-inf")
        max_width = float("-inf")

        for image in self.filenames:
            img = Image.open(image)
            img_array = np.array(img)
            height, width,_ = img_array.shape

            max_height = max(max_height,height)
            max_width = max(max_width,width)

        return max_height,max_width

    @staticmethod
    def validate_input_directory(directory):
        if os.path.exists(directory) is False or os.path.isdir(directory) is False:
            raise ValueError("Image directory is not a valid directory")

    @staticmethod
    def validate_inputs(width,height,size):
        if width[0]<2:
            raise ValueError("Minimum width cannot be smaller than 2")
        if height[0]<2:
            raise ValueError("Minimum height cannot be smaller than 2")
        if size[0]<2:
            raise ValueError("Minimum size cannot be smaller than 2")
        if width[0]>width[1]:
            raise ValueError("Minimum width cannot be greater than maximum width")
        if height[0]>height[1]:
            raise ValueError("Minimum height cannot be greater than maximum height")
        if size[0]>size[1]:
            raise ValueError("Minimum size cannot be greater than maximum size")

    def __getitem__(self, index):
        image_file=self.filenames[index]
        image_loaded=Image.open(image_file)
        image_array=np.array(image_loaded)
        image_array=image_array.astype(dtype=self.dtype)
        grayscale_image=to_grayscale.to_grayscale(image_array)

        rng=np.random.default_rng(seed=index)
        # [0] in order to extract the value from the array
        width=rng.integers(low=self.width_range[0],high=self.width_range[1]+1,size=1)[0]
        height=rng.integers(low=self.height_range[0],high=self.height_range[1]+1,size=1)[0]

        # (1,H,W)
        if width>grayscale_image.shape[2]:
            width=grayscale_image.shape[2]
        if height>grayscale_image.shape[1]:
            height=grayscale_image.shape[1]

        width = min(width,self.max_width)
        height = min(height,self.max_height)

        x=rng.integers(low=0,high=grayscale_image.shape[2]-width+1,size=1)[0]
        y=rng.integers(low=0,high=grayscale_image.shape[1]-height+1,size=1)[0]

        size=rng.integers(low=self.size_range[0],high=self.size_range[1]+1,size=1)[0]

        pixelated_image,known_array,target_array=prepare_image.prepare_image(grayscale_image,x,y,width,height,size)

        return pixelated_image,known_array,target_array,image_file

    def __len__(self):
        return len(self.filenames)


