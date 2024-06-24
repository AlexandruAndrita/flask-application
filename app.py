import os
from werkzeug.utils import secure_filename
import io
import base64
from io import BytesIO

from flask import Flask, render_template, request, redirect, flash, send_file, make_response
import pickle

from PIL import Image
import numpy as np
import torch
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.random_projection import GaussianRandomProjection
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline

from evaluation.model_file import CNNModel, CNNEncDecModel
from prep_dataset import to_grayscale, helpers
from evaluation.train_individual_model import find_model_output


app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# instantiating both models here since it makes no sense to instantiate them every time an image is processed
simple_cnn_model = CNNModel()
try:
    simple_cnn_model.load_state_dict(torch.load("CNNModel.pth"))
except FileNotFoundError:
    raise FileNotFoundError("CNNModel.pth is not found.")
except RuntimeError:
    raise RuntimeError("CNNModel.pth does not have the right configuration")
simple_cnn_model.eval()

cnn_enc_dec_model = CNNEncDecModel()
try:
    cnn_enc_dec_model.load_state_dict(torch.load("CNNDecEndModel.pth"))
except FileNotFoundError:
    raise FileNotFoundError("CNNDecEndModel.pth is not found.")
except RuntimeError:
    raise RuntimeError("CNNDecEndModel.pth does not have the right configuration")
cnn_enc_dec_model.eval()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['jpg', 'jpeg', "JPG", "JPEG"]


def allowed_pickle(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ['pkl']


def prepare_image_for_interface(image):
    final_image_pil = Image.fromarray(image.astype(np.uint8))
    buffered = BytesIO()
    final_image_pil.save(buffered, format="JPEG")
    processed_img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return processed_img_str


def save_image(image, filename):
    images_directory = os.path.join(os.getcwd(), 'savedImages')
    if not os.path.exists(images_directory):
        os.makedirs(images_directory)
    image_path = os.path.join(images_directory, filename)
    with open(image_path, 'wb') as f:
        f.write(image)
    return image_path


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part','error')
            return render_template('index.html')
        file = request.files['file']
        pickle_file = request.files['pickle_file']
        if file.filename == '':
            flash('No image selected','error')
            return render_template('index.html')
        if pickle_file.filename == '':
            flash('No boolean mask selected','error')
            return render_template('index.html')
        if file and allowed_file(file.filename) and allowed_pickle(pickle_file.filename):
            filename = secure_filename(file.filename)
            file_content = file.read()
            pickle_content = pickle_file.read()
            encoded_pickle = base64.b64encode(pickle_content).decode('utf-8')
            encoded_img = base64.b64encode(file_content).decode('utf-8')
            return render_template('index.html', filename=encoded_img, pickle_file=encoded_pickle)
        else:
            flash('Invalid file type.','error')
    return render_template('index.html')


def model_prediction(grayscale_img, known_array, max_value, min_value, option):
    grayscale_img_tensor = torch.tensor([])
    if option == "simple":
        grayscale_img_tensor = torch.tensor(grayscale_img.copy(), dtype=torch.float32).unsqueeze(0)
    elif option == "enc-dec":
        grayscale_img_tensor = torch.tensor(grayscale_img.copy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    grayscale_img_tensor_normalized = helpers.apply_normalization(grayscale_img_tensor,max_value,min_value)

    with torch.no_grad():
        if option == "simple":
            output = simple_cnn_model(grayscale_img_tensor_normalized)
        elif option == "enc-dec":
            output = cnn_enc_dec_model(grayscale_img_tensor_normalized)

    if option == "simple":
        grayscale_img_tensor_normalized = grayscale_img_tensor_normalized.squeeze(0)
        output = output.squeeze(0)
    elif option == "enc-dec":
        grayscale_img_tensor_normalized = grayscale_img_tensor_normalized.squeeze(0).squeeze(0)
        output = output.squeeze(0).squeeze(0)

    grayscale_img_tensor_normalized = helpers.denormalize(grayscale_img_tensor_normalized, max_value, min_value)
    output = helpers.denormalize(output, max_value, min_value)

    output = output.cpu().detach().numpy()
    grayscale_img_tensor_normalized = grayscale_img_tensor_normalized.cpu().detach().numpy()

    grayscale_img_tensor_normalized[known_array == 0] = output[known_array == 0]

    # (output with mask, whole model output)
    return grayscale_img_tensor_normalized, output

@app.route('/process', methods=['POST'])
def process_image():
    filename = request.form['filename']
    pickle_file = request.form['pickle_file']
    pickle_bytes = base64.b64decode(pickle_file)
    known_array = pickle.loads(pickle_bytes)
    image = base64.b64decode(filename)
    img = Image.open(io.BytesIO(image))
    np_arr = np.array(img)
    """
    converting user input (user's image) to grayscale image
    """
    grayscale_img = to_grayscale.to_grayscale(np_arr)
    grayscale_img = grayscale_img.reshape(grayscale_img.shape[1],grayscale_img.shape[2],1).mean(axis=2)
    known_array = known_array.squeeze(0)

    if known_array.shape[0]!=grayscale_img.shape[0] or known_array.shape[1]!=grayscale_img.shape[1]:
        flash("Shapes of Image and Mask do not match",'error')
        return redirect('/')

    processed_filenames = list()
    max_value = grayscale_img.max().item()
    min_value = grayscale_img.min().item()
    
    if grayscale_img.shape[0] in [128,170] and grayscale_img.shape[1] in [128,170] and grayscale_img.shape[0]!=grayscale_img.shape[1]:
        # Simple CNN Architecture
        output_with_mask, whole_output_model = model_prediction(
            grayscale_img=grayscale_img,
            known_array=known_array,
            max_value=max_value,
            min_value=min_value,
            option="simple"
        )

        processed_filenames.append(prepare_image_for_interface(output_with_mask)) # output using the boolean mask
        processed_filenames.append(prepare_image_for_interface(whole_output_model)) # whole output from model
        
    if grayscale_img.shape[0] == 128 and grayscale_img.shape[1] == 170:
        # Encoder-Decoder CNN Architecture
        output_with_mask, whole_output_model = model_prediction(
            grayscale_img=grayscale_img,
            known_array=known_array,
            max_value=max_value,
            min_value=min_value,
            option="enc-dec"
        )

        processed_filenames.append(prepare_image_for_interface(output_with_mask)) # output using the boolean mask
        processed_filenames.append(prepare_image_for_interface(whole_output_model)) # whole output from model

    final_knn20neighbors = find_model_output(
        regressor=KNeighborsRegressor(n_neighbors=20, metric='canberra'),
        known_array=known_array,
        image=grayscale_img
    )

    final_knn25neighbors = find_model_output(
        regressor=KNeighborsRegressor(n_neighbors=25, weights='distance'),
        known_array=known_array,
        image=grayscale_img
    )

    final_randomforest = find_model_output(
        regressor=RandomForestRegressor(n_estimators=100),
        known_array=known_array,
        image=grayscale_img
    )

    final_decisiontressdepth40leaf7 = find_model_output(
        regressor=DecisionTreeRegressor(max_depth=40,min_samples_leaf=7),
        known_array=known_array,
        image=grayscale_img
    )

    base = make_pipeline(GaussianRandomProjection(n_components=10),DecisionTreeRegressor(max_depth=10, max_features=5))
    final_adaboost01 = find_model_output(
        regressor=AdaBoostRegressor(base, n_estimators=50, learning_rate=0.01),
        known_array=known_array,
        image=grayscale_img
    )

    processed_filenames.append(prepare_image_for_interface(final_knn20neighbors))
    processed_filenames.append(prepare_image_for_interface(final_knn25neighbors))
    processed_filenames.append(prepare_image_for_interface(final_randomforest))
    processed_filenames.append(prepare_image_for_interface(final_decisiontressdepth40leaf7))
    processed_filenames.append(prepare_image_for_interface(final_adaboost01))

    return render_template('index.html', filename=filename, processed_filenames=processed_filenames)


@app.route('/save_image', methods=['POST'])
def save_image_route():
    # in bytes
    filename = request.form['filename']
    decoded_img = base64.b64decode(filename)
    saved_path = save_image(decoded_img, 'processed_image.jpg')
    flash(f'Image saved successfully. Discarding input','imageSaved')
    response = make_response(send_file(saved_path, as_attachment=True))
    response.set_cookie('fileDownload', 'true', max_age=60)
    return response


@app.route('/discard_images', methods=['POST'])
def discard_images():
    return redirect('/')


@app.route('/discard_input_image', methods=['POST'])
def discard_input_image():
    return redirect('/')


if __name__ == '__main__':
    app.run()

