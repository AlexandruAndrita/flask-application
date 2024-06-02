import os
from PIL import Image
import shutil
import hashlib
from pathlib import Path
import numpy as np


def get_image_hash(file):
    h=hashlib.sha1()
    with open(file,"rb") as f:
        chunk=0
        # reading the file in 1024-byte chunks until an empty chunk is read (i.e. we read all the content)
        while chunk!=b'':
            chunk=f.read(1024)
            # updates the sha1 object with the bytes passed as argument
            h.update(chunk)
    # returning hexadecimal string representation
    return h.hexdigest()


def check_image(file,root):
    if file.endswith('.jpeg') is False and file.endswith('.jpg') is False and file.endswith('.JPG') is False and file.endswith('.JPEG') is False:
        return 1
    if os.path.getsize(os.path.join(root, file)) > 250000:  # getting the size of the file
        return 2
    try:
        img = Image.open(os.path.join(root, file))
        image_width = img.width
        image_height = img.height
        image_mode = img.mode  # should be "RGB" or "L"
        if image_height < 100 or image_width < 100:
            return 4
        if image_mode != "RGB" and image_mode != "L":
            return 4

        img_aray = np.array(img)
        img_variance = np.var(img_aray)
        if img_variance<=0:
            return 5

    except IOError:
        return 3


def create_log_file(path):
    log_file_dirname=os.path.dirname(path)

    # parents = True - every missing directory is created
    # exist_ok = True - if file already exists, the error is ignored
    Path(log_file_dirname).mkdir(parents=True, exist_ok=True)


# deleting all the files from a specific directory
def delete_directory_content(directory_path):
    try:
        files=os.listdir(directory_path)
        if len(files)!=0:
            for file in files:
                file_path=os.path.join(directory_path,file)
                os.remove(file_path)
    except FileNotFoundError:
        # the directory does not exist at the moment
        # still, no problem bypassing the exception since the directory will be created later
        pass


def validate_images(input_dir: str, output_dir: str, log_file: str, formatter: str = "07d"):
    input_dir=os.path.abspath(input_dir)
    absolute_paths=[]

    create_log_file(log_file)
    log_file_f=open(log_file,"w")

    """
    output_dir should be empty because otherwise we would encounter errors when renaming files
    this preparation phase will be called once anyway, so normally, it would not need this method
    """
    delete_directory_content(output_dir)

    if os.path.exists(input_dir):
        for root, dir, files in os.walk(input_dir):
            for file in files:
                absolute_paths.append(os.path.abspath(os.path.join(root,file)))
        absolute_paths.sort() # sorting the files according to their absolute paths

        if os.path.exists(output_dir) is False:
            os.mkdir(output_dir)

        valid_images=0
        hash=list()
        for image_path in absolute_paths:
            if check_image(image_path, input_dir) is None:

                file_name,file_extension=os.path.splitext(image_path)
                new_name = ("{:" + formatter + "}").format(valid_images)+file_extension

                shutil.copy(image_path,output_dir)
                image_basename=os.path.basename(image_path)
                image_destination=os.path.join(output_dir,image_basename)
                new_image_destination=os.path.join(output_dir,new_name)
                hash_new_image=get_image_hash(image_destination)
                if hash_new_image not in hash:
                    hash.append(hash_new_image)
                    os.rename(image_destination,new_image_destination)
                    valid_images += 1
                else:
                    file_error=image_basename+",6\n"
                    log_file_f.write(file_error)

                    os.remove(image_destination)
            else:
                # create error message and add them in the log file
                error_key=check_image(image_path,input_dir)
                image_basename=os.path.basename(image_path)

                file_error=image_basename+","+str(error_key)+"\n"
                log_file_f.write(file_error)

    else:
        log_file_f.close()
        raise FileNotFoundError(f"The path '{input_dir}' does not exist")

    log_file_f.close()

    return valid_images


def create_file_name(batch_name):
    folder_file_name = "test_batch_" + str(batch_name)
    log_file_name = "log_file_batch_" + str(batch_name) + ".txt"
    return folder_file_name,log_file_name

