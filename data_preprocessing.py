import pandas as pd
import numpy as np

def read_data(path: str) -> pd.DataFrame:
    """
    Reads a json file and returns a pandas dataframe.
    
    Args:
    path (str): Path to the json file.
    
    Returns:
    pd.DataFrame: A pandas dataframe containing the data.
    """
    df = pd.read_json(path)
    
    return df 


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes the dataframe and returns a new dataframe.
    
    Args:
    df (pd.DataFrame): Dataframe to be processed.
    
    Returns:
    pd.DataFrame: A processed pandas dataframe.
    """
    # Selecting relevant columns
    df = df[["id", "type", "commentsCount", "likesCount", "latestComments", "images"]]
    
    # Renaming columns
    new_columns = {"id": "id","commentsCount": "n_comments","likesCount": "n_likes","latestComments": "comments","images": "image"}
    df = df.rename(columns=new_columns)
    
    # Filtering out rows where type is 'Video', likes count is -1.0 and image is not null
    df = df[(df["type"] != "Video")] 
    df = df[df["n_likes"] != -1.0]
    df = df[df["image"].notna()]
    
    # Removing rows with no image
    df = df[df["image"].apply(len) > 0]
    
    # Selecting the first image if there are multiple images
    df["image"] = df["image"].apply(lambda x: x[0])
    
    # Extracting text from comments
    df["comments"] = df["comments"].apply(lambda x: [i["text"] for i in x if "text" in i])
    
    # Resetting the id
    df.reset_index(drop=True, inplace=True)
    df["id"] = df.index + 1
    
    # Converting empty lists in comments to np.nan and creating a separate dataframe for comments
    df["comments"] = df["comments"].apply(lambda x: x if isinstance(x, list) and x else np.nan)
    df_comments = df.explode("comments")[["id", "comments"]]
    
    # Removing comments column from the original dataframe
    df = df.drop("comments", axis=1)
    
    return df, df_comments


def save_data(df: pd.DataFrame, df_comments: pd.DataFrame) -> None:
    """
    Saves the dataframe and comments dataframe as csv files.
    
    Args:
    df (pd.DataFrame): Dataframe to be saved.
    df_comments (pd.DataFrame): Comments dataframe to be saved.
    """
    df.to_csv("data/posts.csv", index=False)
    df_comments.to_csv("data/posts_comments.csv", index=False)
    
    
def main():
    path = "data/posts.json"
    
    # Read, process and save the data
    df = read_data(path)
    df, df_comments = process_data(df)
    save_data(df, df_comments)
    
if __name__ == "__main__":
    main()
