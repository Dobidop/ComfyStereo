# ComfyStereo
# Introduction
A very basic port of the stereoscopic script used in thygate/stable-diffusion-webui-depthmap-script

Most of the credit goes to them, and most of the rest to some LLMs.

The Stereo Image Node is a basic port from the Automatic1111 stereo script. In the current implementation it doesn't support image batches.
The LazyStereo node is another very basic stereo image generator I created with help from LLMs. This node supports image batches.



# Installation
To install the these nodes, clone this repository and add it to custom_nodes folder in your ComfyUI nodes directory:
```
git clone https://github.com/Dobidop/ComfyStereo.git
pip install -r requirements.txt
```
or, download the files manually from the repo and put them in the 'ComfyUI\custom_nodes' folder

# Example workflow

![workflow](https://github.com/user-attachments/assets/87b2292b-21ff-4fff-afac-31cfc572a277)
