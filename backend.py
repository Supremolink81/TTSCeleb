import streamlit as st
from voice_management import VoiceManager
from typing import Union

#voice_manager: VoiceManager = VoiceManager()

def add_state_to_session(state: Union[str, dict], state_value=None) -> None:

    """
    Adds some data to the current session state if it is not already present.

    Args:

        ```py 
        Union[str, dict] state
        ```
        The state to add. If it is a string, it is interpreted as
        a key and is assigned state_value's value. If it is a dictionary, all key value pairs 
        are inserted as state into the session.

        ```py
        Any state_value
        ```
        
        the state value to assign to the key, if the state parameter is a string.

    Raises:

        ```py
        ValueError 
        ```   
        
        in the following cases:

        - `state_value` is not None when state is a dict

        - state_value is None when state is a string

        - a type for state other than a string or dictionary is given

        - if state is a dict and any of the keys in state are not strings
    """

    if type(state) == dict:

        if state_value is not None:

            raise ValueError("value must not be given when the given state is a dictionary")

        for state_element, state_element_value in state.items():

            if type(state_element) != str:

                raise ValueError("All keys in state dictionary must be strings")

            if state_element not in st.session_state:

                st.session_state[state_element] = state_element_value

    elif type(state) == str:

        if state_value is None:

            raise ValueError("value must be given when the given state is a string key")

        if state not in st.session_state:

            st.session_state[state] = state_value

    else:

        raise ValueError("Passed in state must be a string or dict")

if __name__ == "__main__":

    voice_column, text_column = st.columns(2) 

    add_state_to_session("test", [])

    add_state_to_session("audio_path", "")
    
    voice_name: str = st.text_input("Voice Name: ")

    def test_callback():

        st.session_state["test"].append(voice_name)

    for thing in st.session_state["test"]:

        st.write(thing)

    st.button("Add Voice", on_click=test_callback)

    st.write("End of list")