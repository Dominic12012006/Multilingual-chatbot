import sounddevice as sd
import numpy as np
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
def translate_audio(srclang,type):
    # Record a 10-second audio clip
    duration = 5  # seconds
    sample_rate = 16000  # model expects 16kHz; adjust if necessary
    channels = 1  # mono recording
    print("Recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='float32')
    sd.wait()  # wait until recording finishes
    print("Recording complete!")

    # 'recording' is now your NumPy array with the audio data.
    '''print("Playing back the recording...")
    sd.play(recording, samplerate=sample_rate)
    sd.wait()  # Wait until playback is complete
    print("Playback finished!")'''

    # Load model and processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=srclang, task=type)

    # Process the recorded audio
    input_features = processor(recording.squeeze(), sampling_rate=sample_rate, return_tensors="pt").input_features

    # Generate token ids and decode them
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    #print("Transcription:", transcription[0])
    return transcription[0]

