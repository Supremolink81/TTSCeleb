import streamlit as st
import torch
from backend import *
#from voice_management import *

if __name__ == "__main__":

    st.set_page_config(layout="wide")

    st.title("TTSCeleb: voice clone your favority celebrity, politician or even yourself!")

    voice_column, add_voice_column, text_column, preset_column, audio_file_column = st.columns([1, 2, 2, 3, 1], gap="medium") 

    add_state_to_session({
        "test" : [],
        "audio_array" : torch.tensor([]),
        #"voice_manager" : VoiceManager(),
    })

    with voice_column:

        st.subheader("Voice List: ")

        for thing in st.session_state["test"]:

            st.write(thing)

        selected_voice: str = single_selector_page(
            title="Select Your Voice:",
            description="",
            selector_list=st.session_state["test"],
        )

    with add_voice_column:

        st.subheader("Add Voice To Voice Database")

        st.write("It's recommended to have at least 10 seconds of reference audio.")

        voice_name: str = st.text_input("Voice Name: ")

        st.file_uploader("Upload Reference Audio:")

        def test_callback():

            st.session_state["test"] = [voice_name] + st.session_state["test"]

            st.session_state["test"].sort()

        st.button("Add", on_click=test_callback)

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

        st.button("**Generate Audio**", disabled = (selected_preset == "" or selected_voice == ""))