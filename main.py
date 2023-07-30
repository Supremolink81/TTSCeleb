import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import torch
from backend import *
from voice_management import *

if __name__ == "__main__":

    st.set_page_config(layout="wide")

    st.title("TTSCeleb: voice clone your favority celebrity, politician or even yourself!")

    voice_column, add_voice_column, text_column, preset_column, audio_file_column = st.columns([1, 2, 2, 3, 1], gap="medium") 

    add_state_to_session({
        "test" : [],
        "audio_array" : torch.tensor([]),
        "voice_manager" : VoiceManager(),
    })

    with voice_column:

        st.subheader("Voice List: ")

        st.session_state["voice_manager"].render_voices()

        selected_voice: str = single_selector_page(
            title="Select Your Voice:",
            description="",
            selector_list=st.session_state["test"],
        )

        st.button("Delete Voice", disabled=bool(selected_voice))

    with add_voice_column:

        st.subheader("Add A Voice To The Voice Database")

        st.write("It's recommended to have at least 10 seconds of reference audio.")

        voice_name: str = st.text_input("Voice Name: ")

        audio_file: Union[None, UploadedFile] = st.file_uploader("Upload Reference Audio:", accept_multiple_files=False, type=["wav", "mp3"])

        image_file: Union[None, UploadedFile] = st.file_uploader("Upload Profile Picture (Optional): ", accept_multiple_files=False, type=["png", "jpg"])

        def test_callback():

            if image_file is not None:

                st.session_state["voice_manager"].add_voice(voice_name, audio_file, image_file)

            else:

                st.session_state["voice_manager"].add_voice(voice_name, audio_file)

            st.session_state["test"] = [voice_name] + st.session_state["test"]

            st.session_state["test"].sort()

        st.button("Add Voice", on_click=test_callback, disabled=audio_file is None)

    with text_column:

        st.subheader("Enter Text To Convert:")

        input_text: str = st.text_area("")

    with preset_column:

        selected_preset: str = single_selector_page(
            title="Enter a preset for TTS generation.",
            description="""
            The options are:

            'ultra_fast': Low quality speech at a fast inference rate.

            'fast': Decent quality speech at a decent inference rate. A good choice for mass inference.

            'standard': Very good quality, somewhat slow inference. This is generally about as good as you are going to get.

            'high_quality': Use if you want the absolute best (WARNING: EXTREMELY SLOW).
            """,
            selector_list=["ultra_fast", "fast", "standard", "high_quality"]
        )

    with audio_file_column:

        st.subheader("Generated Audio")

        st.audio(st.session_state["audio_array"].numpy(), sample_rate=22050)

        def generate_audio_callback():

            st.session_state["audio_array"] = st.session_state["voice_manager"].text_to_speech(selected_voice, input_text, selected_preset)

        st.button("**Generate Audio**", disabled = bool(selected_preset) or bool(selected_voice), on_click=generate_audio_callback)