



def generate_speech(model_name: str, text: str) -> bytes:

    """
    Converts a line of text to speech with a given model name.

    Args:

        str model: the name of the model to use for text-to-speech.

        str text: the text to convert to speech.

    Returns:

        a byte string representing the audio.
    """

    trained_model = select_model(model_name)

def select_model(model_name: str):

    pass