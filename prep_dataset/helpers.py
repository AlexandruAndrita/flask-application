def less_equal_than(pixel_value):
    return pixel_value/12.92


def greater_than(pixel_value):
    tmp = (pixel_value+0.055)/1.055
    tmp = tmp**2.4
    return tmp


def get_y_linear(r_linear, g_linear, b_linear, red_coef, green_coef, blue_coef):
    y_linear = r_linear*red_coef+g_linear*green_coef+b_linear*blue_coef
    return y_linear


def apply_normalization(tensor_target, max_value, min_value):
    tensor_tmp = (tensor_target-min_value)/(max_value-min_value)
    return tensor_tmp


def denormalize(tensor_target, max_value, min_value):
    tensor_tmp = tensor_target*(max_value-min_value)+min_value
    return tensor_tmp
