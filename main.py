import os
import time
import torch
import torchaudio
import tempfile
import logging
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from generator import load_csm_1b, Segment

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (Allow requests from any domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ✅ Allow all origins (change for security)
    allow_credentials=True,
    allow_methods=["*"],  # ✅ Allow all HTTP methods
    allow_headers=["*"],  # ✅ Allow all headers
)

# Set logging
logging.basicConfig(level=logging.INFO)

# Determine device
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

logging.info(f"Using device: {device}")

MODEL_PATH = "ckpt.pt"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    logging.info("Downloading model file...")
    model_path = hf_hub_download(
        repo_id="sesame/csm-1b",
        filename="ckpt.pt",
        local_dir=".",  
        local_dir_use_symlinks=False,
    )
else:
    model_path = MODEL_PATH
    logging.info("Model file already exists. Skipping download.")

# Load CSM-1B generator


generator = load_csm_1b(model_path, device)
logging.info("Model loaded successfully.")
logging.info("Model loaded successfully.")

# Example speaker embeddings
speakers = [0, 1, 0, 0]
transcripts = [
    "Hey, how are you doing?",
    "Pretty good, pretty good.",

]
audio_paths = [
    "audio_files/utterance_0.wav",
    "audio_files/utterance_1.wav",

]

# Function to load reference audio
def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    return audio_tensor

# Prepare reference segments
segments = [
    Segment(text=transcript, speaker=speaker, audio=load_audio(audio_path))
    for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
]

@app.post("/generate_audio", summary="Generate speech from text")
async def generate_audio(text: str = Query(..., description="Text to convert into speech")):
    """
    Generate speech from input text using CSM-1B model.
    Returns an audio file (WAV format).
    """

    if not text:
        raise HTTPException(status_code=400, detail="Text input is required")

    start_time = time.time()

    # Generate speech
    logging.info(f"Generating audio for: '{text}'")
    audio = generator.generate(
        text=text,
        speaker=1,
        context=segments,
        max_audio_length_ms=10_000,
    )

    # Save to a temporary file
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    torchaudio.save(temp_audio.name, audio.unsqueeze(0).cpu(), generator.sample_rate)

    generation_time = time.time() - start_time
    logging.info(f"Audio generated in {generation_time:.2f} seconds")

    # Streaming response for audio file
    def iter_audio():
        with open(temp_audio.name, "rb") as f:
            yield from f
        os.remove(temp_audio.name)  # Cleanup after streaming

    return StreamingResponse(
        iter_audio(),
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'attachment; filename="generated_audio.wav"',
            "X-Processing-Time": f"{generation_time:.2f}s"
        },
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
