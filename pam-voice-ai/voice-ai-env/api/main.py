from fastapi import FastAPI,UploadFile, File
import uvicorn
import soundfile as sf
import numpy as np

from .audio_processing import transcribe_audio, generate_response
from .app import  text_to_speech

app = FastAPI()

@app.post("/process_audio/")
async def process_audio(file: UploadFile=File(...)):
    audio_data,sameple_rate=sf.read(file.file)

    #convert audio to text
    transcription=transcribe_audio(audio_data)

    #generate a response using llm
    response_text=generate_response(transcription)

    #convert reponse text to speech
    text_to_speech(response_text)

    return {"response": response_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0", port=8000)