# from fastapi import FastAPI
# from flask import Flask

# from fastapi import FastAPI, File, UploadFile
# import whisper
# import soundfile as sf
# import io

# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# import pyttsx3


# app = FastAPI()

# @app.get("/")
# async def root():
#     return {"message": "Welcome to the PAM Voice AI API!"}

# if __name__ == "__main__":
#     import uvicorn 
#     uvicorn.run(app,host="0.0.0.0", port=8000)

# app = Flask(__name__)
# @app.route("/")
# def home():
#     return {"message": "Welcome to the PAM Voice AI API!"}

# if __name__=="__main__":
#     app.run(debug=True, host="0.0.0.0", port=8000)


# # async def upload_audio (file:UploadFile=File(...)):
# #     contents = await file.read()
# #     audio_data, samplerate-sf.read(io.BytesIO(contents))
# #     model=whisper.load


# #loading tokenizer and mode
# tokenizer=AutoTokenizer.from_pertrained("gpt2")
# model=AutoModelForCausalLM.from_pretrained("gpt2")

# #enabling CUDA 
# device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model=model.to(device)

# def preprocess_text(text):
#     text=text.strip()
#     return text

# def generate_response(input_text,max_length=50):
#     #preprocessing the text
#     input_text=preprocess_text(input_text)

#     #tokenize input text
#     input_ids=tokenizer.encode(input_text,return_tensors='pt').to(device)

#     #generating response using the model
#     output=model.generate(input_ids, max_length,pad_token_id=tokenizer.eos_token_id)

#     #decode output tokens to get generated text
#     response=tokenizer.decode(output[0],skip_special_tokens=True)

#     return response

# if __name__ == "__main__":
#     #transcribed text
#     transcribed_text="How can I help you with your car purchase today?"

#     response = generate_response(transcribed_text)
#     print("AI Response:", response)


# with torch.no_grad():
#     output = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)



# def process_transcription(transcription):
#     response=generate_response(transcription)
#     return response

# transcribed_text="Tell me more about the financing options"
# response=process_transcription(transcribed_text)

# #initialize TTS engine 
# tts_engine=pyttsx3.init()

# #setting properties for the tts engine
# tts_engine.setProperty('rate',150)
# tts_engine.setProperty('volume',1)

# #list of voices
# voices=tts_engine.getProperty('voice')
# for voice in voices:
#     print(f"Voice: {voice.name}")

# tts_engine.setProperty('voice',voices[0].id)

# def text_to_speech(text):
#     tts_engine.say(text)

#     tts_engine.runAndWait()

# def process_transcription_and_response(transcription):
#     response=generate_response(transcription)


#     text_to_speech(response)

# transcribed_text="tell me more about financiing options"
# process_transcription_and_response(transcribed_text)

# if __name__ == "__main__":
#     transcribed_text = "How can I help you with your car purchase today?"
#     process_transcription_and_response(transcribed_text)

from fastapi import FastAPI, UploadFile, File
import whisper
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pyttsx3

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the PAM Voice AI API!"}

# Initialize Whisper model
model = whisper.load_model("base")

# # Initialize TTS engine
# try:
#     # Initialize TTS engine
#     tts_engine = pyttsx3.init()
#     tts_engine.setProperty('rate', 150)  # Set the speed of the speech
#     tts_engine.setProperty('volume', 1)  # Set volume level (0.0 to 1.0)
#     voices = tts_engine.getProperty('voices')
#     tts_engine.setProperty('voice', voices[0].id)  # Use default voice

# except Exception as e:
#     print(f"Error initializing TTS engine: {e}")

# if voices:
#     tts_engine.setProperty('voice', voices[0].id)  # Use default voice
# else:
#     raise RuntimeError("No TTS voices found on the system.")


# Initialize TTS engine
try:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 150)  # Set the speed of the speech
    tts_engine.setProperty('volume', 1)  # Set volume level (0.0 to 1.0)
    
    voices = tts_engine.getProperty('voices')  # Retrieve available voices
    if voices and len(voices) > 0:
        tts_engine.setProperty('voice', voices[0].id)  # Use default voice
    else:
        raise RuntimeError("No TTS voices found on the system.")

except Exception as e:
    print(f"Error initializing TTS engine: {e}")
    tts_engine = None


def text_to_speech(text):
    """Convert text to speech using pyttsx3."""
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        print(f"Error with TTS engine: {e}")



# Load tokenizer and model for LLM
tokenizer = AutoTokenizer.from_pretrained("gpt2")
llm_model = AutoModelForCausalLM.from_pretrained("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llm_model = llm_model.to(device)

def preprocess_text(text):
    return text.strip()

def generate_response(input_text, max_length=50):
    input_text = preprocess_text(input_text)
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    with torch.no_grad():
        output = llm_model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def text_to_speech(text):
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 150)  # Set the speed of the speech
    tts_engine.setProperty('volume', 1)  # Set volume level (0.0 to 1.0)
    
    voices = tts_engine.getProperty('voices')
    tts_engine.setProperty('voice', voices[0].id)  # Use default voice
    
    tts_engine.say(text)
    tts_engine.runAndWait()

def process_transcription_and_response(transcription):
    response = generate_response(transcription)
    text_to_speech(response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    transcribed_text = "How can I help you with your car purchase today?"
    process_transcription_and_response(transcribed_text)
