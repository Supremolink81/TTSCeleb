from tortoise_tts.tortoise.api import TextToSpeech
import torch
from torchaudio.backend.sox_io_backend import load
from scipy.io.wavfile import write as save_audio
from torchaudio.transforms import Resample
from streamlit import _DeltaGenerator

class VoiceManager:

    """
    A class to wrap the voices stored in the current session. It contains
    convenient methods for adding voices, removing voices, and generating text
    to speech from a voice.

    Fields:

        dict[str, tuple[torch.Tensor, torch.Tensor]] voices: the dictionary to map
        voice names to their latent vectors, which are necessary for voice cloning.

        TextToSpeech text_to_speech: the Tortoise class used for text-to-speech.
    """

    voices: dict[str, tuple[torch.Tensor, torch.Tensor]]

    text_to_speech: TextToSpeech

    def __init__(self):

        self.voices = {}

        self.text_to_speech = TextToSpeech()

    def add_voice(self, voice_name: str, voice_recording_paths: list[str]):

        """
        Adds a voice to the database, if it doesn't already exist.

        Args:

            str voice_name: the name to give the voice.

            list[str] voice_recording_paths: the paths to the voice recordings to use to clone the voice.
        """

        recording_tensors: list[torch.Tensor] = []

        for voice_recording_path in voice_recording_paths:

            audio: tuple[torch.Tensor, int] = load(voice_recording_path)

            audio_tensor: torch.Tensor = audio[0]

            audio_sample_rate: int = audio[1]

            # tortoise TTS model requires audio be sampled at 22.05 kHz
            resampler: Resample = Resample(orig_freq=audio_sample_rate, new_freq=22050)

            recording_tensor: torch.Tensor = resampler(audio_tensor)

            recording_tensors.append(recording_tensor)

        latent_vectors: tuple[torch.Tensor, torch.Tensor] = self.text_to_speech.get_conditioning_latents(recording_tensors)

        self.voices[voice_name] = latent_vectors

    def delete_voice(self, voice_name: str):

        """
        Deletes a voice from the database, if it exists.

        Args:

            str voice_name: the name to give the voice.
        """

        if voice_name in self.voices:

            del self.voices[voice_name]

    def text_to_speech(self, voice_name: str, text: str, audio_file_path: str):

        """
        Converts a given text into speech in the given voice and saves it as a waveform file.

        Args:

            str voice_name: the voice to use.

            str text: the text to convert to speech.

            str audio_file_path: the path to save the audio to. 
        """

        try:

            voice_latent_vectors: tuple[torch.Tensor, torch.Tensor] = self.voices[voice_name]

            audio_tensor: torch.Tensor = self.text_to_speech.tts(text, conditioning_latents=voice_latent_vectors)

            # sample rate of generated speech is 24 kHz in tortoise backend
            save_audio(audio_file_path, 24000, audio_tensor)

        except KeyError:

            raise KeyError("Voice not found: " + voice_name)
        
    def render_voices(self, column: _DeltaGenerator):

        """
        Function to render list of voices in a Streamlit application.

        Args:

            ```py
            _DeltaGenerator column
            ```

            The column to render the voices to.
        """