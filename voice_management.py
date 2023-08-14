from tortoise.api import TextToSpeech
import torch
import math
from tempfile import NamedTemporaryFile
import streamlit as st
import librosa
from streamlit.runtime.uploaded_file_manager import UploadedFile
import numpy as np
from functools import total_ordering
import sortedcontainers
from PIL import Image

question_mark_image: Image.Image = Image.open("drive/MyDrive/questionmark.png")

@st.cache_resource
def load_tts():

  return TextToSpeech(half=True)

class AudioHelpers:

    @classmethod
    def get_audio_data(cls, voice_recording: UploadedFile) -> tuple[torch.Tensor, int]:

        with NamedTemporaryFile(suffix=voice_recording.name[voice_recording.name.find(".")+1:]) as temp_audio_file:

            temp_audio_file.write(voice_recording.getvalue())

            temp_audio_file.seek(0)

            audio_array, _ = librosa.load(temp_audio_file.name, sr=22050)

            # Tortoise requires audio be sampled at 22.05 kHz
            return torch.from_numpy(audio_array).unsqueeze(0)

    @classmethod
    def get_parts_to_split_audio_into(cls, audio_tensor: torch.Tensor) -> int:

        audio_duration_in_seconds: float = audio_tensor.shape[0] / 22050

        # Tortoise TTS recommends each clip be around 6-10 seconds
        return math.ceil(audio_duration_in_seconds / 6)

    @classmethod
    def split_audio(cls, audio_tensor: torch.Tensor) -> list[torch.Tensor]:

        recording_tensors: list[torch.Tensor] = []

        parts_to_split_audio_into: torch.Tensor = AudioHelpers.get_parts_to_split_audio_into(audio_tensor)

        audio_chunk_size: int = 22050 * 6

        for part_index in range(parts_to_split_audio_into):

            if part_index == parts_to_split_audio_into-1:

                audio_chunk: torch.Tensor = audio_tensor[part_index * audio_chunk_size : ]

            else:

                audio_chunk: torch.Tensor = audio_tensor[part_index * audio_chunk_size : (part_index + 1) * audio_chunk_size]

            recording_tensors.append(audio_chunk)

        return recording_tensors

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

        name_hash: int = self.name.__hash__()

        latent_vector_hash: int = self.latent_vectors.__hash__()

        if name_hash == latent_vector_hash:

            return name_hash

        return name_hash ^ latent_vector_hash

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

    voices: dict[str, Voice]

    text_to_speech_model: TextToSpeech

    voice_name_set: sortedcontainers.SortedSet[Voice]

    def __init__(self):

        self.voices = {}

        self.text_to_speech_model = load_tts()

        self.voice_name_set = sortedcontainers.SortedSet()

    def add_voice(self, voice_name: str, voice_recording: UploadedFile, profile_picture: UploadedFile = None):

        """
        Adds a voice to the database, if it doesn't already exist.

        Args:

            `str voice_name`: the name to give the voice.

            `UploadedFile voice_recording`: the voice recording to use to clone the voice.

            `Image.Image profile_picture`: the profile picture to give the voice. Default: question mark image.
        """

        recording_tensors: list[torch.Tensor] = self._get_recording_tensors(voice_recording)

        latent_vectors: tuple[torch.Tensor, torch.Tensor] = self.text_to_speech_model.get_conditioning_latents(recording_tensors)

        if profile_picture is None:

            profile_picture_as_pil: Image.Image = question_mark_image

        else:

            with NamedTemporaryFile(suffix=profile_picture.name[profile_picture.name.find(".")+1:]) as temp_picture_file:

                temp_picture_file.write(profile_picture.getvalue())

                temp_picture_file.seek(0)

                profile_picture_as_pil: Image.Image = Image.open(temp_picture_file.name)

        voice_object: Voice = Voice(voice_name, latent_vectors, profile_picture_as_pil)

        self.voices[voice_name] = voice_object

        self.voice_name_set.add(voice_object)

    def delete_voice(self, voice_name: str):

        """
        Deletes a voice from the database, if it exists.

        Args:

            `str voice_name`: the name to give the voice.
        """

        if voice_name in self.voices:

            self.voice_name_set.remove(self.voices[voice_name])

            del self.voices[voice_name]

    def text_to_speech(self, voice_name: str, text: str, preset: str) -> torch.Tensor:

        """
        Converts a given text into speech in the given voice.

        Args:

            `str voice_name`: the voice to use.

            `str text`: the text to convert to speech.

            `str preset`: the preset settings to pass to Tortoise.

        Returns:

            a tensor representing the audio generated.

        Raises:

            `KeyError` if the voice name is not in the VoiceManager.
        """

        try:

            voice_latent_vectors: tuple[torch.Tensor, torch.Tensor] = self.voices[voice_name].latent_vectors

            audio_tensor: torch.Tensor = self.text_to_speech_model.tts_with_preset(text, conditioning_latents=voice_latent_vectors, preset=preset)

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

    def _get_recording_tensors(self, voice_recording: UploadedFile) -> list[torch.Tensor]:

        audio_tensor: torch.Tensor = AudioHelpers.get_audio_data(voice_recording)

        return AudioHelpers.split_audio(audio_tensor)