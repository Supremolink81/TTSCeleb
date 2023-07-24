import bark
import os
from bark.generation import ALLOWED_PROMPTS
from torchaudio.backend.sox_io_backend import save

ALLOWED_PROMPTS.add("barack_obama")

def generate_speech(model_name: str, text: str, saved_audio_path: str) -> None:

    """
    Converts a line of text to speech with a given model name and saves it as a waveform file.

    Args:

        str model: the name of the model to use for text-to-speech.

        str text: the text to convert to speech.

        str saved_audio_path: the path to save teh audio to.

    Returns:

        None
    """
    
    generated_audio = bark.generate_audio(text=text, history_prompt=model_name)

    save(saved_audio_path, generated_audio)

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["SUNO_USE_SMALL_MODELS"] = "True" 

    bark.preload_models(text_use_gpu=False)   

    generate_speech("barack_obama", "Hello Donald, are you going to pick the most boring class in the game again?", "obama.wav")