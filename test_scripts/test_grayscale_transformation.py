from prep_dataset import image_validation
from prep_dataset import prepare_image
from prep_dataset import to_grayscale
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    # TEST_SIZE = 2
    #
    # for i in range(1,TEST_SIZE+1):
    #     input_dir_name, _ = image_validation.create_file_name(i)
    #     output_dir_name, _ = image_validation.create_file_name(i)
    #     _, log_file_name = image_validation.create_file_name(i)
    #
    #     input_dir = f"D:\\an III\\bachelor's thesis\\thesis\\input_test_images_validation\\{input_dir_name}"
    #     output_dir = f"D:\\an III\\bachelor's thesis\\thesis\\output_test_images_validation\\{output_dir_name}"
    #     log_file = f"D:\\an III\\bachelor's thesis\\thesis\\log_files\\{log_file_name}"
    #     image_validation.validate_images(input_dir=input_dir,output_dir=output_dir,log_file=log_file)
    #
    #     print(f"Test set with index {i} evaluated")

    input_dir = "D:\\an III\\bachelor's thesis\\thesis\\input_test_images_validation\\test_batch_3"
    output_dir = "D:\\an III\\bachelor's thesis\\thesis\\output_test_images_validation\\test_batch_3"
    log_file = "D:\\an III\\bachelor's thesis\\thesis\\log_files\\log_file_batch_3.txt"
    image_validation.validate_images(input_dir=input_dir, output_dir=output_dir, log_file=log_file)

    images = os.listdir(output_dir)
    for image in images:
        abs_path = os.path.join(output_dir, image)
        img = Image.open(abs_path)
        img_array = np.array(img)
        grayscale_img_array = to_grayscale.to_grayscale(img_array)

        x, y, width, height, size = 500, 600, 200, 400, 30

        pixelated_image,known_array,target_array = prepare_image.prepare_image(grayscale_img_array,x,y,width,height,size)
        fig = plt.figure(figsize=(10,7))
        rows, columns = 2,2,

        # original image
        fig.add_subplot(rows,columns,1)
        plt.imshow(img)
        plt.axis('off')
        plt.title("Original Image")

        # printing grayscale image
        grayscale_image = Image.fromarray(grayscale_img_array[0])
        fig.add_subplot(rows,columns,2)
        plt.imshow(grayscale_image,cmap='gray')
        plt.axis('off')
        plt.title("Grayscale Image")

        # pixelated image
        pix_image = Image.fromarray(pixelated_image[0])
        fig.add_subplot(rows,columns,3)
        plt.imshow(pix_image,cmap='gray')
        plt.axis('off')
        plt.title("Pixelated Image")

        # target array
        target_image = Image.fromarray(target_array[0])
        fig.add_subplot(rows,columns,4)
        plt.imshow(target_image,cmap='gray')
        plt.axis('off')
        plt.title("Target Image")

        plt.show()