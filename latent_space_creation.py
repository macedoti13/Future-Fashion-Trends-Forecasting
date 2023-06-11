from tensorflow.keras.models import Model, load_model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import cv2 
import os 
import re


def get_images_and_filenames(path):
    """
    This function takes in a path to a directory, and returns two lists:
    1. A list of all images in the directory, read and processed.
    2. A list of filenames corresponding to these images.
    """
    image_list = []
    filename_list = []

    for filename in os.listdir(path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            img = img / 255.0
            image_list.append(img)
            filename_list.append(filename)
            
    filename_list = [int(re.findall(r'\d+', s)[0]) for s in filename_list]


    X_train = np.array(image_list, dtype=np.float32)
    
    return X_train, filename_list


def create_encoder(path):
    """
    This function takes in a path to the pre-trained autoencoder model, and returns an encoder model.
    """
    autoencoder = load_model(path)
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('dropout_4').output)
        
    return encoder


def get_latent_spaces(data, encoder):
    """
    This function takes in data and an encoder model, and returns the latent spaces of the data.
    """
    latent_spaces = []
    
    for img in data:
        img = np.expand_dims(img, axis=0) 
        latent_space = encoder.predict(img)
        flattened_latent_space = np.reshape(latent_space, (-1)) 
        latent_spaces.append(flattened_latent_space)

    return np.array(latent_spaces)


def scale_latent_spaces(latent_spaces):
    """
    This function takes in the latent spaces, scales them using StandardScaler, and returns the scaled latent spaces.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(latent_spaces)
    
    
def use_pca_latent_spaces(latent_spaces, n_components=180):
    """
    This function takes in the latent spaces, applies PCA to reduce their dimensionality, and returns the transformed latent spaces.
    """
    pca = PCA(n_components=n_components) 
    return pca.fit_transform(latent_spaces)


def save_to_dataframe(file_names, latent_spaces_pca):
    """
    This function takes in a list of filenames and the corresponding PCA-transformed latent spaces, and saves them to a dataframe, which is then saved to an HDF5 file.
    """
    df = pd.DataFrame({'path':file_names, 'latent_space': list(latent_spaces_pca)})
    df.to_hdf('data/latent_spaces.h5', key='df_items', mode='w')
    
    
def main():
    """
    This function orchestrates the whole process of loading images, extracting latent spaces, scaling and applying PCA, and saving the results.
    """
    IMAGES_PATH = "images/segmented_images/"
    MODEL_PATH = "models/autoencoder.h5"
    
    X_train, file_names = get_images_and_filenames(IMAGES_PATH)
    encoder = create_encoder(MODEL_PATH)
    latent_spaces = get_latent_spaces(X_train, encoder)
    latent_spaces_normalized = scale_latent_spaces(latent_spaces)
    latent_spaces_pca = use_pca_latent_spaces(latent_spaces_normalized)
    save_to_dataframe(file_names, latent_spaces_pca)

if __name__ == "__main__":
    main()
