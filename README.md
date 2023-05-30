# COX3-ML-Model
This repository contains code for building and training an image classification model using the EfficientNetB3 architecture. The model is trained on a dataset of images and can predict the diagnosis of a given image.

## Requirements

- Python 3.7 or later
- TensorFlow 2.0 or later
- NumPy
- pandas
- OpenCV (cv2)
- Matplotlib

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/Himasnhu-AT/COX3-ML-Model.git
   cd COX3-ML-Model
   ```

2. Create a virtual environment (optional but recommended):

   ```
   python3 -m venv env
   source env/bin/activate
   ```

3. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

4. Download the dataset:
   
   - Download the dataset from [Dataset Download Link](https://www.kaggle.com/datasets/andyczhao/covidx-cxr2).
   - Extract the dataset into the project directory.

## Usage

1. Open the `main.py` file and modify the `data_path` variable to point to the directory where you extracted the dataset.

2. Run the script:

   ```
   python main.py
   ```

   This will train the model on the dataset and save the trained model in the specified directory.

3. To evaluate the model on the test dataset and generate predictions, you can use the following code:

   ```python
   import tensorflow as tf
   import cv2

   # Load the saved model
   model = tf.saved_model.load('path/to/saved/model')

   # Load and preprocess the test image
   image_path = 'path/to/test/image.jpg'
   image = cv2.imread(image_path)
   preprocessed_image = preprocess_image(image)  # Preprocess according to the model requirements

   # Make predictions
   predictions = model.predict(preprocessed_image)

   # Process the predictions and get the diagnosis
   diagnosis = process_predictions(predictions)

   print("Diagnosis:", diagnosis)
   ```

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [UnLicense](LICENSE).
