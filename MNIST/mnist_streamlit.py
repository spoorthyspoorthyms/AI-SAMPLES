import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from PIL import Image
import numpy as np
import io

# Load and preprocess the MNIST dataset
def load_and_preprocess_data():
    # Load the dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Reshape the images to include a channel dimension (28, 28, 1)
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    # Normalize the images to have values between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    return train_images, train_labels, test_images, test_labels

# Build the CNN model
def build_cnn_model():
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Train the model
def train_model(model, train_images, train_labels, test_images, test_labels):
    model.fit(train_images, train_labels, epochs=5, batch_size=64)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    return model, test_acc

# Function to preprocess an uploaded image
def preprocess_image(image):
    # Convert to grayscale and resize to (28, 28)
    img = image.convert("L").resize((28, 28))
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = img_array.reshape((1, 28, 28, 1))  # Add batch dimension
    return img_array

# Main function to run the Streamlit app
def main():
    # Load and preprocess the MNIST dataset
    train_images, train_labels, test_images, test_labels = load_and_preprocess_data()

    # Build the CNN model
    model = build_cnn_model()

    # Train the model
    model, test_acc = train_model(model, train_images, train_labels, test_images, test_labels)

    # Streamlit app UI
    st.title("MNIST Digit Classifier with CNN")
    st.write("This app classifies handwritten digits using a Convolutional Neural Network (CNN).")
    st.write(f"Model test accuracy: {test_acc * 100:.2f}%")

    # Upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Preprocess the image and make a prediction
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Preprocess the image
        img_array = preprocess_image(image)

        # Predict the digit
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)

        st.write(f"Predicted Digit: {predicted_digit}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
