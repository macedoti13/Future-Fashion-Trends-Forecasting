import pandas as pd
import requests
import os 
import time

def download_images(df, batch_size, delay):
    """
    Download images in a batch with a delay between batches.

    Args:
        df (pd.DataFrame): Dataframe with image URLs and corresponding IDs.
        batch_size (int): The number of images to be downloaded in a batch.
        delay (int): The delay time in seconds between batches.

    """
    # make sure the images directory exists
    if not os.path.exists("images"):
        os.makedirs("images")

    # Iterate over the DataFrame rows with batch control
    for idx, row in df.iterrows():
        image_url = row["image"]
        id = row["id"]
        file_path = f"images/{id}.jpg"

        # Check if the file already exists
        if os.path.exists(file_path):
            continue

        # Send a HTTP request to the URL of the image
        response = requests.get(image_url)

        # Check if the request is successful
        if response.status_code == 200:
            # Open a file in write mode
            # Save the image under the "images" directory and name it with the id
            with open(file_path, "wb") as f:
                # Write the contents of the response (the image) to the file
                f.write(response.content)

        # If we"ve reached the batch limit, sleep for a while
        if (idx + 1) % batch_size == 0:
            time.sleep(delay)

def main():
    
    df = pd.read_csv("posts.csv", index_col=0)
    download_images(df=df, batch_size=10, delay=10)
    
if __name__ == "__main__":
    main()

