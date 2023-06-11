from tensorflow.keras.models import Model, load_model
import numpy as np
import cv2 
import os 


def get_images(path):
    """"""
    # Create a list to store images
    image_list = []

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
            # Append the image to the list
            image_list.append(img)

    # Convert the list to numpy array
    X_train = np.array(image_list, dtype=np.float32)
    
    return X_train
    
    
def create_encoder(path):
    """"""
    # autoencoder model
    autoencoder = load_model(path)
        
    # input layer
    autoencoder_input = autoencoder.input

    # layer that outputs the latent space
    latent_space_output = autoencoder.get_layer('max_pooling2d').output

    # Create the encoder model
    encoder = Model(autoencoder_input, latent_space_output)
        
    return encoder
        

def main():
    
    # paths 
    IMAGES_PATH = "images/segmented_images/"
    MODEL_PATH = "models/autoencoder.h5"
    
    X_train = get_images(IMAGES_PATH)
    encoder = create_encoder(MODEL_PATH)
    

if __name__ == "__main__":
    main()