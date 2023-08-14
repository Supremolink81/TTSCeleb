import streamlit as st
from typing import Union

def add_state_to_session(state: Union[str, dict], state_value = None) -> None:

    """
    Adds some data to the current session state if it is not already present.

    Args:

        `Union[str, dict] state`: the state to add. If it is a string, it is interpreted as
        a key and is assigned state_value's value. If it is a dictionary, all key value pairs
        are inserted as state into the session.

        `Any state_value`: the state value to assign to the key, if the state parameter is a string.

    Raises:

        `ValueError` in the following cases:

        - `state_value` is not `None` when `state` is a dictionary

        - `state_value` is `None` when `state` is a string

        - a type for `state` other than a string or dictionary is given

        - if `state` is a dictionary and any of the keys in `state` are not strings
    """

    if isinstance(state, dict):

        if state_value is not None:

            raise ValueError("expected state_value to be None when state is a dictionary, got" + str(state_value))

        add_multi_state_to_session(state)

    elif isinstance(state, str):

        add_single_state_to_session(state, state_value)

    else:

        raise ValueError("Passed in state key must be a string or dict, is of type " + str(type(state)) + " with value " + str(state))

def add_single_state_to_session(state: str, state_value) -> None:

    """
    Adds a single state value to the session state.

    Args:

        `str state`: the state to add.

        `Any state_value`: the value to give the state.

    Raises:

        `ValueError` in the following cases:

        - `state` is not a string

        - `state_value` is None
    """

    if isinstance(state, str):

        if state_value is None:

            raise ValueError("state value must be given when the given state is a string key")

        if state not in st.session_state:

            st.session_state[state] = state_value

    else:

        raise ValueError("Passed in state key must be a string, is of type " + str(type(state)) + " with value " + str(state))

def add_multi_state_to_session(state: dict) -> None:

    """
    Adds multiple state values to the session state.

    Args:

        `dict state`: the state to add. Keys must all be strings.

    Raises:

        `ValueError` in the following cases:

        - any of `state`'s keys are not strings
    """

    for state_element, state_element_value in state.items():

        if type(state_element) != str:

            raise ValueError("All keys in state dictionary must be strings, found one of type " + str(type(state_element)) + " with value " + str(state_element))

        if state_element not in st.session_state:

            st.session_state[state_element] = state_element_value

def get_selected_value(selector: list) -> str:

    """
    Obtains the value selected in a selector object.

    Args:

        `list selector`: the selector to get the selected item from.

    Returns:

        the selected item.
    """

    return selector[0] if len(selector) == 1 else ""

def single_selector_page(title: str, description: str, selector_list: list[str]) -> str:

    """
    Given a set of options, creates a selector for a user interface
    and continuously returns the currently selected value.

    Args:

        `str title`: the title to give the selector page.

        `str description`: the description to give the selector.

        `list[str] selector_list`: the list of options for the selector.

    Returns:

        the currently selected value.
    """

    st.subheader(title)

    st.write(description)

    selector: list[str] = st.multiselect("", selector_list, max_selections=1)

    return get_selected_value(selector)