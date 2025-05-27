import os
import numpy as np
import nibabel as nib
import pydicom
from flask import Flask, request, render_template, redirect, url_for
from keras.models import load_model
from werkzeug.utils import secure_filename
from matplotlib import pyplot as plt
from scipy.ndimage import zoom

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploaded/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the pre-trained model
MODEL_PATH = 'models/brats_3d.hdf5'
model = load_model(MODEL_PATH, compile=False)

# Supported file extensions
ALLOWED_EXTENSIONS = {'npy', 'nii', 'nii.gz', 'dcm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Handle file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    if not allowed_file(file.filename):
        return "Unsupported file type. Please upload a .npy, .nii, .nii.gz, or .dcm file.", 400

    # Save the uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Load and preprocess the file
    try:
        if filename.endswith('.npy'):
            test_img = np.load(filepath)
        elif filename.endswith(('.nii', '.nii.gz')):
            nifti_img = nib.load(filepath)
            test_img = nifti_img.get_fdata()

            # Resize to (128, 128, 128)
            target_shape = (128, 128, 128)
            scale_factors = (
                target_shape[0] / test_img.shape[0],
                target_shape[1] / test_img.shape[1],
                target_shape[2] / test_img.shape[2],
            )
            test_img = zoom(test_img, scale_factors, order=1)  # Use linear interpolation

            # Add channel dimension and duplicate to create 3 channels
            test_img = np.expand_dims(test_img, axis=-1)  # Add single channel
            test_img = np.repeat(test_img, 3, axis=-1)  # Duplicate to 3 channels
        elif filename.endswith('.dcm'):
            dicom_img = pydicom.dcmread(filepath)
            test_img = dicom_img.pixel_array

            # Ensure the image has the correct shape
            if len(test_img.shape) == 3:  # If 3D, add a channel dimension
                test_img = np.expand_dims(test_img, axis=-1)
            test_img = np.repeat(test_img, 3, axis=-1)  # Duplicate to 3 channels
        else:
            return "Unsupported file type.", 400

        # Add batch dimension
        test_img_input = np.expand_dims(test_img, axis=0)
    except Exception as e:
        return f"Error loading file: {e}", 400

    # Predict using the model
    test_prediction = model.predict(test_img_input)
    test_prediction_argmax = np.argmax(test_prediction, axis=4)[0, :, :, :]

    # Get slice index from form, default to middle if not provided or invalid
    try:
        slice_index = int(request.form.get('slice_index', -1))
    except ValueError:
        slice_index = -1

    # Validate and adjust slice index
    max_slice = test_prediction_argmax.shape[2] - 1
    if slice_index < 0 or slice_index > max_slice:
        n_slice = test_prediction_argmax.shape[2] // 2  # Middle slice
    else:
        n_slice = slice_index

    plt.figure(figsize=(16, 8))

    # Plot the input image (selected slice of channel 0)
    plt.subplot(1, 2, 1)
    plt.imshow(test_img[:, :, n_slice, 0], cmap='gray')  # Assuming channel 0 is used for visualization
    plt.title(f'Input Image (Channel 0), Slice {n_slice}')

    # Plot the prediction
    plt.subplot(1, 2, 2)
    plt.imshow(test_prediction_argmax[:, :, n_slice], cmap='gray')
    plt.title(f'Prediction, Slice {n_slice}')

    # Save and build relative path
    prediction_image_filename = 'prediction_' + filename + '.png'
    prediction_image_path = os.path.join(app.config['UPLOAD_FOLDER'], prediction_image_filename)
    plt.savefig(prediction_image_path)
    plt.close()

    # Build relative URL for Flask static route
    relative_path = os.path.join('uploaded', prediction_image_filename).replace('\\', '/')
    # Ensure the path is URL-safe
    return render_template('result.html', prediction_image=url_for('static', filename=relative_path))
    
# Run the app
if __name__ == '__main__':
    app.run(debug=True)