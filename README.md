# ComfyStereo
# Introduction
A very basic port of the stereoscopic script used in thygate/stable-diffusion-webui-depthmap-script

Most of the credit goes to them, and most of the rest to some LLMs.

* The Stereo Image Node is a basic port from the Automatic1111 stereo script
* The LazyStereo node is a naïve stereo image generator I created with the help from LLMs

Both nodes supports handling of batches.
Beware of bugs.

# Installation
To install the these nodes, clone this repository and add it to custom_nodes folder in your ComfyUI nodes directory:
```
git clone https://github.com/Dobidop/ComfyStereo.git
pip install -r requirements.txt
```

# Example workflow

![workflow](https://github.com/user-attachments/assets/87b2292b-21ff-4fff-afac-31cfc572a277)
