import pandas as pd
import emot

def substituir_emojis_por_texto(texto):
    emot_obj = emot.emot()
    
    try:
        
        emoji_info = emot_obj.emoji(texto)
        idx = len(emoji_info["value"])
        
        for i in range(idx):
            texto = texto.replace(emoji_info["value"][i], emoji_info["mean"][i])
            
    except:
        
        print(texto)
        pass
    
    return texto

def main():
    # Criando um DataFrame de exemplo
    df = pd.read_csv("data/posts_comments.csv")

    # Aplicando a função aos comentários
    df['comments'] = df['comments'].apply(substituir_emojis_por_texto)
    
    df = df.drop("Unnamed: 0", axis=1)
    
    df.to_csv("./data/preprocessed_comments.csv", header=["id","comments","label"], index = True)

if __name__ == "__main__":
    main()