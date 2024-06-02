import numpy as np
import prep_dataset.helpers as mathematical_formulas


def apply_formulas(array: np.ndarray):
    array=np.divide(array,255)
    array=np.where(array<=0.04045,mathematical_formulas.less_equal_than(array),mathematical_formulas.greater_than(array))

    r_linear=array[:,:,0]
    g_linear=array[:,:,1]
    b_linear=array[:,:,2]
    red_coef, green_coef, blue_coef=0.2126,0.7152,0.0722

    y_linear=mathematical_formulas.get_y_linear(r_linear,g_linear,b_linear,red_coef,green_coef,blue_coef)
    y=np.where(y_linear<=0.003130,y_linear*12.92,1.055*np.power(y_linear,1/2.4)-0.055)
    y=mathematical_formulas.normalize_output(y)
    return y


def to_grayscale(pil_image: np.ndarray) -> np.ndarray:
    # 2D image - grayscale
    if len(pil_image.shape)==2:
        array_returned=np.copy(pil_image)

        height=pil_image.shape[0]
        width=pil_image.shape[1]
        array_returned=array_returned.reshape(1,height,width) # (1,H,W)
        return array_returned

    # 3D image - RGB
    elif len(pil_image.shape)==3:
        if pil_image.shape[2]!=3:
            raise ValueError("The 3rd dimension of RGB image is not 3. The image cannot be processed.")
        height=pil_image.shape[0]
        width=pil_image.shape[1]

        array_modified=np.copy(pil_image)
        array_modified=apply_formulas(array_modified)
        output_rgb_to_grayscale=array_modified.reshape(1,height,width)

        if np.issubdtype(pil_image.dtype,np.integer):
            output_rgb_to_grayscale=np.round(output_rgb_to_grayscale*255)
            output_rgb_to_grayscale=output_rgb_to_grayscale.astype(pil_image.dtype)
        else:
            output_rgb_to_grayscale = (output_rgb_to_grayscale * 255).astype(pil_image.dtype)

        return output_rgb_to_grayscale

    else:
        raise ValueError("The shape of the image does not correspond to neither RGB nor grayscale images.")
