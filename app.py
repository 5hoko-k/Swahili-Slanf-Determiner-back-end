from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from aiohttp import ClientSession
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    'http://localhost:5173/'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    allow_credentials=True,
    expose_headers=["Content-Disposition"]
)

model = load_model('./model.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)


@app.get('/')
async def home():
    cors_headers = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    }
    return JSONResponse(content={"message": "Hello World"}, headers=cors_headers)


@app.get('/predict')
async def trial(request: Request):
    api_url = "https://v1.nocodeapi.com/mbaddy/fbsdk/SzvDUuvbwudDbsEf/firestore/allDocuments?collectionName=MLData"

    async with ClientSession() as session:
        async with session.get(api_url) as response:
            if response.status == 200:
                data = await response.json()

                texts = [item['_fieldsProto']['msg']['stringValue'] for item in data]
                sequences = tokenizer.texts_to_sequences(texts)
                sequences = pad_sequences(sequences, maxlen=250)
                predictions = model.predict(sequences)
                prediction_labels = ['Slang' if pred >= 0.5 else 'Not Slang' for pred in predictions]
                
                # Create a list of dictionaries with text and prediction label
                results = []
                for text, label in zip(texts, prediction_labels):
                    result = {"text": text, "prediction": label}
                    results.append(result)

                cors_headers = {
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type",
                }
                return JSONResponse(content=results, headers=cors_headers)

            else:
                print("Error: Failed to retrieve data from the API")
                return {"message": "Error"}

class TextInput(BaseModel):
    text: str

@app.post('/test_prediction')
async def test_prediction(text_input: TextInput):
    # Assuming 'tokenizer', 'model', and 'pad_sequences' are already defined

    # Preprocess the text
    sequence = tokenizer.texts_to_sequences([text_input.text])
    sequence = pad_sequences(sequence, maxlen=250)

    # Make the prediction
    predictions = model.predict(sequence)

    # Get the prediction label
    prediction_label = 'Slang' if predictions[0][0] >= 0.5 else 'Not Slang'

    return {"prediction": prediction_label}