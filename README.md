# TTSCeleb

This is a text-to-speech project where I train a model to clone the voices of various popular figures, including (but not limited to):

- Barack Obama

- Donald Trump

- Ben Shapiro

# Task Description

The task is to clone the voices of a popular person (such as Barack Obama) given an audio recording of the person. It is preferable that the voice not only be recognizable for a given person, but also have tonal depth and non verbal sounds such as laughing and screaming.

Through the HuggingFace platform, there are many pre trained models available which can be used as a baseline for transfer learning. 

# Dataset

We will be using audio recordings of various famous individuals. These recordings contain little to no background noise and others speaking, ensuring our ground truth for training is high quality.

The exact recordings used are found [at this Kaggle link](https://www.kaggle.com/datasets/verracodeguacas/voices).

# Model Selection

I use the Bark model, as it has the ability to create both verbal and non verbal sounds, as desired. It is a transformer based text to audio model created by Suno. For more detailed information regarding Bark, read the GitHub repo [here](https://github.com/suno-ai/bark).