# MNIST Digit Classifier App (Naive Bayes)

## Overview
This project presents an interactive web application for classifying handwritten digits (0-9). The core of this system is a **Gaussian Naive Bayes classifier**, implemented entirely from scratch, demonstrating a fundamental understanding of probabilistic machine learning algorithms.

The application allows users to draw a digit on a canvas, which is then processed and fed into the trained Naive Bayes model for real-time prediction. This serves as a prototype for building interactive machine learning demonstrations.

## Features
* **Interactive Drawing Canvas:** Draw digits directly in the web application.
* **Real-time Prediction:** Get an instant prediction of the drawn digit using the custom-built Naive Bayes model.
* **Image Preprocessing:** Automatic resizing, grayscale conversion, and scaling of drawn input to match model's expected format.
* **User-Friendly Interface:** Built with Streamlit for a simple and intuitive user experience.

## Model
The digit classification is performed by a **Gaussian Naive Bayes classifier**.
* **Implementation:** Developed from scratch in Python, showcasing the underlying principles of Naive Bayes, including prior probability calculation and Gaussian likelihood estimation.
* **Training:** Trained on the standard MNIST handwritten digits dataset.

## Technologies Used
* **Python:** Core programming language.
* **Streamlit:** For creating the interactive web application.
* **streamlit-drawable-canvas:** A Streamlit component enabling the drawing canvas functionality.
* **NumPy:** For efficient numerical operations and array manipulation (e.g., image flattening, scaling).
* **Pillow (PIL):** For image processing tasks like converting to grayscale and resizing.
* **Scikit-learn:** Used for initial MNIST dataset loading during model training (not directly used in `app.py` for model prediction).

## Live Demo
You can try out the live prototype of the application here:
[https://mnistdigitclassifierapp-jwbwipfxrwqkt8ewnxssx4.streamlit.app/](https://mnistdigitclassifierapp-jwbwipfxrwqkt8ewnxssx4.streamlit.app/)

## How to Run Locally
To set up and run this application on your local machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Sreelakshmi-rv/MNIST_Digit_Classifier_App.git](https://github.com/Sreelakshmi-rv/MNIST_Digit_Classifier_App.git)
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd MNIST_Digit_Classifier_App
    ```
3.  **Ensure you have the trained model file:**
    Make sure `naive_bayes_mnist_model.pkl` is present in this directory. This file contains the pre-trained parameters of the Gaussian Naive Bayes model. (You would have generated this in your Colab notebook and downloaded it).
4.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```
5.  **Activate the virtual environment:**
    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
6.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
7.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    This will open the application in your web browser, typically at `http://localhost:8501`.

## Future Improvements & Next Steps
* **Model Accuracy:** For significantly higher accuracy in digit classification, consider exploring more advanced models like:
    * **Bernoulli Naive Bayes:** Which can often perform better than Gaussian Naive Bayes on binary pixel data.
    * Support Vector Machines (SVMs)
    * Simple Convolutional Neural Networks (CNNs) using frameworks like TensorFlow or PyTorch.
* **Data Augmentation:** Augmenting the training data (e.g., slight rotations, shifts of digits) can make the model more robust.
* **User Feedback Loop:** Allow users to correct predictions to collect more data.
* **Performance Optimization:** For very large datasets or real-time high-throughput, optimize model prediction speed.

## Contact
* **GitHub:** [Sreelakshmi-rv](https://github.com/Sreelakshmi-rv)
