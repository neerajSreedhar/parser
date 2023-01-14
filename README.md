# Parser

The Translator is a multimodal deep learning model, which consists of an image encoder, a sentence
encoder, a fusion module and a sentence decoder that takes a complex natural language instruction and outputs simple executable atomic sentences for a UR-5 Robot. The image and the sentence encoder are from the CLIP, OpenAI's open source image and language encoders.

The Executor takes the simpler atomic natural language robotic instructions and executes the task in a simulated environment for a UR-5 Robot. The tasks involve identifying objects, picking and placing objects and pushing and rotating objects.

Authors: Neeraj Sreedhar, Yifan Zhou, Anirudh Bhupalarao, Nupoor Bibawe

Translator-Executor Acrhitecture

![alt text](https://github.com/neerajSreedhar/parser/blob/main/methodology.png)
