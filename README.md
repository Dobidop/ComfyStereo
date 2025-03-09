# ComfyStereo

## Introduction

- **Stereo Image Node** – A port of the **Automatic1111 stereo script** with added features like **edge-aware and direction-aware blurring of the depth map**.
- **DeoVR View Node** – A node for launching and viewing images and videos in **DeoVR**.

---

## DeoVR View Node Setup

To launch DeoVR from this node, you must configure the path to **DeoVR.exe** in the configuration file:

```
ComfyUI\custom_nodes\comfystereo\config.json
```

### Default Path:
```
"C:\\Program Files (x86)\\Steam\\steamapps\\common\\DeoVR Video Player\\DeoVR.exe"
```

### Important Notes:
- You **must** already have an image or video open in DeoVR before this node can change what is displayed in the headset.
- If DeoVR is on its **main menu screen**, this will not work.
- The launcher skips the main menu screen automatically, but if you manually start DeoVR, you need to open an image/video before using this node.

---

## New Features & Depth Map Blurring

New functionality includes **edge-aware and direction-aware blurring** of the depth map, additional interpolation methods, and **stereo distortion improvements**.  

### **Depth map blurring:**
- **Reduces artifacts and harsh transitions** in the final stereo image, especially at **higher divergence settings**.
- **Trade-off**: It **increases computation time** (5-25%). If speed is a concern, you may want to disable it.
- `depth_map_threshhold` sets the depth map gradient sharpness application cutoff. Low values will apply the blur to more shallow gradients, blurring the depth map more broadly (which can negativbely affect the end result). Higher values isolates the steeper gradients.

### **How to Enable Adaptive Blurring**
- Set `'depth_map_blur' = True`.

### **Mask Output Considerations**
- The mask output is **imperfect** and is mainly useful for the **"No Fill"** and **"Imperfect Fill"** options. This outputs a black and white image of the areas which were not filled in.

---

## Stereo Image Generation Methods

### **Key Parameters**
#### **Separation (`separation`)**
- Defines an additional **horizontal shift percentage** applied to the left and right images.
- Modifies the distance between the stereo pair, affecting alignment.

#### **Divergence (`divergence`)**
- Controls the **strength of the 3D effect** (in percentage terms).
- A higher divergence increases depth perception, while a lower value creates a flatter effect.

#### **Stereo Balance (`stereo_balance`)**
- Determines how **divergence is distributed** between the two eyes.
- `0.0` = Even distribution  
- Positive/negative values shift the effect toward one eye.

#### **Stereo Offset Exponent (`stereo_offset_exponent`)**
- Adjusts **depth-to-offset mapping**, influencing how depth values are converted into horizontal shifts.

---

## Infill Methods

Some fill methods are **faster**, while others **preserve structure better**.  
- **Naive methods** are slightly quicker.  
- **Polylines Soft** is often the best general filler.

### **Comparison of Infill Techniques**

| Method                                      | Description |
|---------------------------------------------|-------------|
| **No Fill**                                  | Shifts pixels based on depth without filling gaps. |
| **No Fill - Reverse Projection**            | Works backward to assign pixel values but leaves gaps. |
| **Imperfect Fill - Hybrid Edge**            | Mixes "Polylines" and "Reverse Projection" for better structure. |
| **Fill - Naive**                             | Fills gaps using nearest pixels (causes stretching). |
| **Fill - Naive Interpolating**               | Uses interpolation to smooth gaps. |
| **Fill - Polylines Soft**                    | Uses polylines with **soft edges** to maintain structure. |
| **Fill - Polylines Sharp**                   | Similar to "Soft" but with **sharper transitions**. |
| **Fill - Post-Fill**                         | "No fill" + **edge-aware interpolation**. |
| **Fill - Reverse Projection + Post-Fill**    | "Reverse Projection" + **directional interpolation**. |
| **Fill - Hybrid Edge with Fill**             | Enhanced "Hybrid Edge" with adaptive smoothing. |

---

## Installation

### **Easy Method (Recommended)**
Use **ComfyUI-Manager** for quick installation.

### **Manual Installation**
Clone the repository and place it in ComfyUI's `custom_nodes` directory:
```sh
git clone https://github.com/Dobidop/ComfyStereo.git
pip install -r requirements.txt
```

---

## Example Workflow

### **Image Output**
![workflow(7)](https://github.com/user-attachments/assets/4fb1ff7d-7d3e-4c9f-b389-ec134d624f4e)


### **Video Output**
![workflow(8)](https://github.com/user-attachments/assets/0a61bc30-1821-40b8-b90d-12733f85bdea)


---

## Troubleshooting

- **Low CPU utilization?**  
  - Try **updating Python and ComfyUI-Manager**—this fixed the issue for one user.  

