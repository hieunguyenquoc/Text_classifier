from fastapi import FastAPI
import pickle
from text_preprocess import text_preprocess
from remove_stopwords import remove_stopwords

# MODEL_PATH = "models"

nb_model = pickle.load(open("naive_bayes.pkl",'rb'))
app = FastAPI()

def preprocess(text):
    text = text_preprocess(text)
    text = remove_stopwords(text)
    return text

@app.post("/classify_text")
async def classify_text(text: str):
    text = preprocess(text)
    label = nb_model.predict([text])
    result = {
        'label': int(label[0])
        }
    return result