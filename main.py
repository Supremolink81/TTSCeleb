import streamlit as st
import torch
from backend import *
from voice_management import *

#voice_manager: VoiceManager = VoiceManager()

if __name__ == "__main__":

    voice_column, text_column, audio_file_column = st.columns(3) 

    add_state_to_session({
        "test" : [],
        "audio_array" : torch.tensor(),
    })
    
    voice_name: str = st.text_input("Voice Name: ")

    def test_callback():

        st.session_state["test"].append(voice_name)

    for thing in st.session_state["test"]:

        st.write(thing)

    st.button("Add Voice", on_click=test_callback)

    with audio_file_column:

        st.audio()

    st.write("End of list")