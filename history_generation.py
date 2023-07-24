import torch
import numpy as np
from torchaudio.backend.sox_io_backend import load
from encodec.model import EncodecModel, EncodedFrame
from encodec.utils import convert_audio
from transformers import HubertModel, BertTokenizer

AUDIO_ENCODER: EncodecModel = EncodecModel.encodec_model_24khz()
SEMANTIC_VECTOR_MODEL: HubertModel = HubertModel.from_pretrained("facebook/hubert-base-ls960")

def generate_history_prompt(audio_file_path: str, history_prompt_path: str, use_gpu: bool = True) -> None:

    """
    Generates a history prompt for use with Bark and saves it to disk.

    Args:

        str audio_file_path: the path ot the audio file to use to make the prompt.

        str history_prompt_path: the path to where to save the prompt to disk.

        bool use_gpu: whether to use a GPU. Default: True

    Returns:

        None
    """

    audio_tensor, _ = load(audio_file_path)

    preprocessed_audio_tensor: torch.Tensor = get_preprocessed_audio(audio_file_path)

    coarse_audio_tensor: torch.Tensor = fine_to_coarse(preprocessed_audio_tensor)

    semantic_vectors: torch.Tensor = SEMANTIC_VECTOR_MODEL(audio_tensor)

    np.savez(history_prompt_path, 
        fine_prompt=preprocessed_audio_tensor, 
        coarse_prompt=coarse_audio_tensor
    )

def get_preprocessed_audio(audio_file_path: str) -> torch.Tensor:

    """
    Retrieves an audio file and preprocesses it to a format that can be
    read by the HuBERT encoder and saved as part of a npz file.

    Args:

        str audio_file_path: the audio to preprocess.

    Returns:

        a PyTorch tensor representing the preprocessed audio.
    """

    audio_tensor, sample_rate = load(audio_file_path)

    audio_tensor = convert_audio(audio_tensor, sample_rate, AUDIO_ENCODER.sample_rate, 1).unsqueeze(0)

    with torch.no_grad():

        encoded_audio_frames: list[EncodedFrame] = AUDIO_ENCODER.encode(audio_tensor)

    audio_codes: torch.Tensor = torch.cat([encoded_frame[0] for encoded_frame in encoded_audio_frames], dim=-1)

    return audio_codes.squeeze()

def fine_to_coarse(audio_tensor: torch.Tensor) -> torch.Tensor:

    """
    Retrieves a coarse audio sample from a fine audio sample.
    """

    return audio_tensor[:2, :]