from tortoise_tts.tortoise.api import TextToSpeech
import torch
import streamlit as st
import numpy as np
from torchaudio.backend.sox_io_backend import load
from functools import total_ordering
from torchaudio.transforms import Resample
import sortedcontainers
from PIL import Image

question_mark_image: Image.Image = Image.open("questionmark.png")

@total_ordering
class Voice:

    """
    Data class for voices.

    Fields:

        `str name`: the name of the voice.

        `tuple[torch.Tensor, torch.Tensor] latent_vectors`: the latent vectors 
        of teh voice; necessary for Tortoise TTS.

        `Image.Image profile_picture`: the profile picture of the voice. Used
        for rendering.
    """

    name: str
    latent_vectors: tuple[torch.Tensor, torch.Tensor]
    profile_picture: Image.Image

    def __init__(self, name: str, latent_vectors: tuple[torch.Tensor, torch.Tensor], profile_picture: Image.Image):

        self.name = name

        self.latent_vectors = latent_vectors

        self.profile_picture = profile_picture

    def __hash__(self):

        return (self.name.__hash__() ^ self.latent_vectors.__hash__()) + self.profile_picture.__hash__()
    
    def check_if_other_is_voice(self, other):

        if not isinstance(other, Voice):

            raise ValueError("Tried to compare a Voice object to a non-Voice object.")
    
    def __eq__(self, other) -> bool:

        """
        Returns true if both Voice objects have the same name and false otherwise.
        """

        self.check_if_other_is_voice(other)
        
        return self.name == other.name
    
    def __lt__(self, other) -> bool:

        """
        Compares two Voice objects by lexicographical order of their names.
        """

        self.check_if_other_is_voice(other)

        return self.name < other.name

class VoiceManager:

    """
    A class to wrap the voices stored in the current session. It contains
    convenient methods for adding voices, removing voices, and generating text
    to speech from a voice.

    Fields:

        `dict[str, tuple[torch.Tensor, torch.Tensor]] voices`: the dictionary to map
        voice names to their latent vectors, which are necessary for voice cloning.

        `TextToSpeech text_to_speech`: the Tortoise class used for text-to-speech.

        `sortedcontainers.SortedSet voice_name_set`: an ordered set of voice names;
        used for easy ordering when rendering voices on Streamlit.
    """

    voices: dict[str, tuple[torch.Tensor, torch.Tensor]]

    text_to_speech: TextToSpeech

    voice_name_set: sortedcontainers.SortedSet[Voice]

    def __init__(self):

        self.voices = {}

        self.text_to_speech = TextToSpeech(half=True, use_deepspeed=True, device=torch.device("cuda:0"))

        self.voice_name_set = sortedcontainers.SortedSet()

    def add_voice(self, voice_name: str, voice_recording_paths: list[str], profile_picture: Image.Image = question_mark_image):

        """
        Adds a voice to the database, if it doesn't already exist.

        Args:

            `str voice_name`: the name to give the voice.

            `list[str] voice_recording_paths`: the paths to the voice recordings to use to clone the voice.

            `Image.Image profile_picture`: the profile picture to give the voice. Default: question mark image.
        """

        recording_tensors: list[torch.Tensor] = self._get_recording_tensors(voice_recording_paths)

        latent_vectors: tuple[torch.Tensor, torch.Tensor] = self.text_to_speech.get_conditioning_latents(recording_tensors)

        self.voices[voice_name] = latent_vectors

        voice_object: Voice = Voice(voice_name, latent_vectors, profile_picture)

        self.voice_name_set.add(voice_object)

    def delete_voice(self, voice_name: str):

        """
        Deletes a voice from the database, if it exists.

        Args:

            `str voice_name`: the name to give the voice.
        """

        if voice_name in self.voices:

            del self.voices[voice_name]

    def text_to_speech(self, voice_name: str, text: str) -> torch.Tensor:

        """
        Converts a given text into speech in the given voice.

        Args:

            `str voice_name`: the voice to use.

            `str text`: the text to convert to speech.

        Returns:

            a tensor representing the audio generated.

        Raises:

            `KeyError` if the voice name is not in the VoiceManager.
        """

        try:

            voice_latent_vectors: tuple[torch.Tensor, torch.Tensor] = self.voices[voice_name]

            audio_tensor: torch.Tensor = self.text_to_speech.tts(text, conditioning_latents=voice_latent_vectors)

            return audio_tensor

        except KeyError:

            raise KeyError("Voice not found: " + voice_name)
        
    def render_voices(self):

        """
        Function to render list of voices in a Streamlit application.

        Args:
            
            `None
        """

        for voice in self.voice_name_set:

            profile_picture_as_array: np.array = np.array(voice.profile_picture)

            st.write(voice.name)

            st.image(profile_picture_as_array)

    def _get_recording_tensors(self, voice_recording_paths: list[str]) -> list[torch.Tensor]:

        """
        Obtains the recording tensors for a list of voice recording paths.

        Args:

            `list[str] voice_recording_paths`: the paths to convert.

        Returns:

            a list of tensors corresponding to the audio data.
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

        return recording_tensors