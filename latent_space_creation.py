from tensorflow.keras.models import Model, load_model
import pandas as pd
import numpy as np
import cv2 
import os 


def get_images_and_filenames(path):
    """"""
    # Create lists to store images and filenames
    image_list = []
    filename_list = []

    # Iterate over all files in the directory
    for filename in os.listdir(path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Create the full image path
            img_path = os.path.join(path, filename)
            # Read the image
            img = cv2.imread(img_path)
            # Convert the image to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Resize the image
            img = cv2.resize(img, (256, 256))
            # Normalize the image
            img = img / 255.0
            # Append the image and the filename to the lists
            image_list.append(img)
            filename_list.append(filename)

    # Convert the list to numpy array
    X_train = np.array(image_list, dtype=np.float32)
    
    return X_train, filename_list

    
    
def create_encoder(path):
    """"""
    # autoencoder model
    autoencoder = load_model(path)

    # Create the encoder model
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('dropout_4').output)
        
    return encoder
        

def main():
    
    # paths 
    IMAGES_PATH = "images/segmented_images/"
    MODEL_PATH = "models/autoencoder.h5"
    
    X_train, file_names = get_images_and_filenames(IMAGES_PATH)
    encoder = create_encoder(MODEL_PATH)
    
    latent_spaces = []
    
    for img in X_train:
        img = np.expand_dims(img, axis=0) 
        latent_space = encoder.predict(img)
        flattened_latent_space = np.reshape(latent_space, (-1)) 
        latent_spaces.append(flattened_latent_space)

    latent_spaces = np.array(latent_spaces)

    df = pd.DataFrame({'path':file_names, 'latent_space': list(latent_spaces)})
    df.to_hdf('data/latent_spaces.h5', key='df_items', mode='w')

if __name__ == "__main__":
    main()