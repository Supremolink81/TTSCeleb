# TTSCeleb

This is a text-to-speech project where I create a simple web app to allow people to clone the voices of various popular figures, including (but not limited to):

- Barack Obama

- Donald Trump

- Ben Shapiro

This README will serve as both a description of the project, and a high level tutorial of how to reproduce it. Though, for a more comprehensive view of the concepts presented here, I would encourage readers to consult the References section at the end of this README, where I have provided sources I used to make this project.

We will first describe the task and how the text-to-speech pipeline works, and then move on to the structure of the app.

# Detailed Task Description

The task is to clone the voices of a popular person (such as Barack Obama) given an audio recording of the person. It is preferable that the voice not only be recognizable for a given person, but also have tonal depth and non verbal sounds such as laughing and screaming.

# Dataset

We will be using audio recordings of various famous individuals [1]. These recordings contain little to no background noise and others speaking, ensuring our ground truth for training is high quality.

# Model Selection

Before telling you which model I selected and why, I will first detail other alternatives and why I ultimately chose not to use them.

I started at the Bark text to audio generative model in the hopes of learning how to clone with it [2]. Unfortunately, there were various complications. I was fortunate to find a user called GitMylo had created a ![way](https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer) to clone voices with Bark using a quantizer based on HuBERT [3], but it did not produce very high quality clones (whether this was due to Bark's quality or his model specifically, I am unsure). So I decided to study his pipeline and see if I could use pretrained models to improve on it.

Here is a sample recording produced by GitMylo's method which is meant to clone Joe Biden:

![Biden Recording](./biden_mylo_clone.mov)

While it isn't terrible, I wouldn't say it would be recognizable as Joe Biden.

Though, what Mylo did provide me was a starting point. I had to generate 3 things; a semantic prompt (whatever that meant), a fine representation and a coarse representation. To do these things, I needed an audio encoder and an acoustic tokenizer (or so I thought). For an encoder, I found Encodec by Meta AI, which had been seen to do a decent job at neural audio compression from their experiments [4]. For the tokenizer, I tried BEATs, a model developed by Microsoft Research Asia for audio embedding and tokenization.

However, this had its own problems; BEATs was too large to fit on even Colab's 15 GB VRAM GPUs, so it was out the door. Furthermore, I discovered later on that you shouldn't use a tokenizer for Bark's semantic prompts (despite the fact GitMylo's model was named CustomTokenizer), and that what you needed was a quantizer. In addition, there were specific restrictions in place for generating voice clones that were difficult to untangle, and Bark has no documentation on how to voice clone (which is intentional according to them).

With all of these difficulties, I decided to try my luck on other models. I eventually found a model called Tortoise, a TTS model developed by an OpenAI employee called neonbjb [6]. His repo had what I needed; voice cloning capabilities, flexible configuration, simple API, no restrictions. And with some of the samples I had seen on websites, it seemed like a good quality model. So it is ultimately the one I chose for this project.

# References

[1] https://www.kaggle.com/datasets/verracodeguacas/voices

[2] https://github.com/suno-ai/bark

[3] HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units: https://arxiv.org/abs/2106.07447

[4] High Fidelity Neural Audio Compression: https://arxiv.org/abs/2210.13438

[5] BEATS: Audio Pre-Training with Acoustic Tokenizers: https://arxiv.org/pdf/2212.09058.pdf

[6] Tortoise TTS repo: https://github.com/neonbjb/tortoise-tts