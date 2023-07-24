from tortoise_tts.tortoise.api import TextToSpeech
import torch
from torchaudio.backend.sox_io_backend import load
from torchaudio.transforms import Resample

class VoiceDatabase:

    """
    A class to wrap the voices stored in the current session. It contains
    convenient methods for adding voices, removing voices, and generating text
    to speech from a voice.
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