import streamlit as st
import torch
from backend import *
from voice_management import *

if __name__ == "__main__":

    voice_column, text_column, audio_file_column = st.columns(3) 

    add_state_to_session({
        "test" : [],
        "audio_array" : torch.tensor(),
        "voice_manager" : VoiceManager(),
    })

    with voice_column:

        for thing in st.session_state["test"]:

            st.write(thing)

        st.session_state["voice_manager"].render_voices()

    with text_column:

        voice_name: str = st.text_input("Voice Name: ")

    with audio_file_column:

        st.audio(st.session_state["audio_array"])

        def test_callback():

            st.session_state["test"].append(voice_name)

        st.button("Add Voice", on_click=test_callback)