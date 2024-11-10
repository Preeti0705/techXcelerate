import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image file not found at path: {image_path}")
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    return np.expand_dims(img, axis=-1)  


def load_dummy_image(dummy_path):
    dummy_img = cv2.imread(dummy_path, cv2.IMREAD_GRAYSCALE)
    if dummy_img is None:
        raise FileNotFoundError(f"Dummy image file not found at path: {dummy_path}")
    dummy_img = cv2.resize(dummy_img, (256, 256))
    dummy_img = dummy_img / 255.0
    return np.expand_dims(dummy_img, axis=-1)  


def build_unet_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    

    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    

    b = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    b = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(b)
    

    u1 = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(b)
    u1 = layers.concatenate([u1, c3])
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u1)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
    
    u2 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c4)
    u2 = layers.concatenate([u2, c2])
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u2)
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c5)
    
    u3 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c5)
    u3 = layers.concatenate([u3, c1])
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u3)
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c6)
    

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c6)
    
    model = models.Model(inputs, outputs)
    return model


def load_images_from_directory(directory_path):
    image_paths = [os.path.join(directory_path, fname) for fname in os.listdir(directory_path) if fname.endswith('.jpg') or fname.endswith('.png')]
    return image_paths


image_directory = r"C:\Users\Ananya Pulluri\Desktop\mern stack" 
dummy_image_path = r"C:\Users\Ananya Pulluri\Desktop\dummy\dummy_mri.png" 
image_paths = load_images_from_directory(image_directory)


dummy_image = load_dummy_image(r"C:\Users\Ananya Pulluri\Desktop\hackathone\leg1.png")


model = build_unet_model((256, 256, 1))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


for image_path in image_paths:

    input_image = load_and_preprocess_image(image_path)
    input_image = np.expand_dims(input_image, axis=0)  
    
 
    prediction = model.predict(input_image)
    

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_prediction = cv2.filter2D(prediction[0, :, :, 0], -1, kernel)
    

    difference = np.abs(input_image[0, :, :, 0] - dummy_image[:, :, 0])
    threshold = 0.3  
    abnormal_regions = (difference > threshold).astype(np.uint8)
    

    abnormal_regions = cv2.morphologyEx(abnormal_regions, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(abnormal_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        prediction_image_colored = cv2.cvtColor((sharpened_prediction * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(prediction_image_colored, (x, y), (x + w, y + h), (255, 0, 0), 2)
    

    overlay = cv2.addWeighted(prediction_image_colored, 0.5, cv2.cvtColor((input_image[0, :, :, 0] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), 0.5, 0)
    

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(input_image[0, :, :, 0], cmap='gray')
    plt.title(f"Input Image: {os.path.basename(image_path)}")
    
    plt.subplot(1, 3, 2)
    plt.imshow(sharpened_prediction, cmap='gray')
    plt.title("Patient x-Ray")
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Problem Highlighted")
    
    plt.show()
