
# ComfyStereo
# Introduction

* **Stereo Image Node** - a basic port from the Automatic1111 stereo script with some extra features added, such as edge and direction aware blurring of the depth map. 
* **DeoVR View** - A node for launching and viewing images and videos in DeoVR

-----------

For the DeoVR View node to be able to launch DeoVR you must configure tha DeoVR.exe path in the config file:

*ComfyUI\custom_nodes\comfystereo\config.json*

---
This by default is set to:

"C:\\Program Files (x86)\\Steam\\steamapps\\common\\DeoVR Video Player\\DeoVR.exe"

You **MUST** already be viewing an image/video before this node can change what is being viewed in the headset. If you are at the main menu screen of DeoVR it will not work.
The launcher does this by default, but if you have already started DeoVR you have to open up an image before this works.

------
https://github.com/user-attachments/assets/b76e6243-557b-454b-9baa-1aac2a7eb22a


Added some new functionality such as edge and direction aware blur of the depth map. I've also managed to add some additional interpolation and stereo distortion methods. By 'I', I mean I tortured ChatGPT for hours.

#### Things to note
To activate the **adaptive blurring**, set 'depth_blur_sigma' (the amount of blur applied) to a **value greater than 0**.

The mask output is imperfect and only provides useful output for the 'No fill' options.

If you see **low CPU utilization** when running the node without depth map blurring you might want to **update python and/or ComfyUI-Manager** as that seemed to have solved the issue for one user who had this issue.

### Stereo Image Generation Methods

### Summary Table

| Method                 | Description |
|------------------------|-------------|
| **'No fill'**            | Simple depth shift with no gap filling. |
| **'No fill - Reverse projection'**         | Works backward to assign pixel values but leaves gaps. |
| **'Imperfect fill - Hybrid Edge'**     | Mixes 'Polylines' and 'Reverse projection' for better results. |
| **'Fill - Naive'**           | Fills gaps with nearest pixel, causing stretching. |
| **'Fill - Naive interpolating'** | Uses interpolation to smooth out gaps. |
| **'Fill - Polylines Soft'**  | Uses polylines with soft edges to maintain structure. |
| **'Fill - Polylines Sharp'** | Like 'soft' but with sharper transitions. |
| **'Fill - Post-fill'**       | 'No fill' with edge-aware interpolation and blending. |
| **'Fill - Reverse projection with Post-fill'**    | 'Reverse projection' with directional interpolation and blurring. |
| **'Fill - Hybrid Edge with fill'** | Enhanced version of 'Hybrid Edge' with adaptive smoothing. |

1. **'No fill'**  
   - Basic method that applies the depth-based shift but does **not** fill in gaps left by the transformation.  
   - Results in visible holes where pixels are moved but no data is available to replace them.

2. **'No fill - Reverse projection'**  
   - Instead of shifting pixels **away from their original positions**, it works **backward**:  
     - For each output pixel, it looks at where it would have originated and assigns a value accordingly.  
   - More accurate in some cases but tends to leave gaps.

3. **'Imperfect fill - Hybrid Edge'**  
   - Combines **'Polylines'** and **'Reverse projection'** techniques:  
     - Uses 'Polylines' for **continuous** regions to preserve structure.  
     - Uses 'Reverse projection' for **discontinuous** areas to avoid stretching.  
   - Works well in most cases but may still struggle with extreme depth changes.

4. **'Fill - Naive'**  
   - Moves pixels according to depth but **fills in gaps** using the nearest available pixel.  
   - This leads to **stretched artifacts** but removes the most obvious missing data.

5. **'Fill - Naive interpolating'**  
   - Like 'naive' but **interpolates between neighboring pixels** to fill gaps smoothly.  
   - Reduces stretching artifacts but still relies on simple heuristics.

6. **'Fill - Polylines Soft'**  
   - Treats each row of pixels as a **polyline** and shifts it while attempting to maintain structure.  
   - Uses **soft edges**, meaning transition areas are blended more smoothly.  
   - Better at preserving fine details and gradients.

7. **'Fill - Polylines Sharp'**  
   - Similar to 'Polylines Soft' but with **sharper transitions** at edges.  
   - Avoids excessive blending but may introduce jagged artifacts in areas with sharp depth changes.

8. **'Fill - Post-fill'**  
   - Applies the **'No fill'** method first, then post-processes using:  
     - **Edge-aware interpolation**, attempting to fill gaps based on nearby structures.  
     - **Blending techniques** to smooth out harsh transitions.  
   - Improves upon 'No fill' but still has artifacts in extreme cases.

9. **'Fill - Reverse projection with Post-fill'**  
   - Applies the **'inverse'** method first, then post-processes using:  
     - **Directional interpolation** to fill missing areas more intelligently.  
     - **Blurring corrections** to reduce sharp edge artifacts.  
   - A more refined version of 'Reverse projection' that significantly improves final image quality.

10. **'Fill - Hybrid Edge with fill'**  
   - An advanced version of 'hybrid_edge' with:  
     - **Stronger edge-aware processing** to ensure smooth depth transitions.  
     - **Adaptive interpolation** based on local depth complexity.  
     - **Directional-aware smoothing** to further refine edges.  
   - Better in some cases.

---


# Installation

## Easy method

Use ComfyUI-Manager

## Manual install

To install the these nodes, clone this repository and add it to custom_nodes folder in your ComfyUI nodes directory:
```
git clone https://github.com/Dobidop/ComfyStereo.git
pip install -r requirements.txt
```

# Example workflow

Image
![workflow(5)](https://github.com/user-attachments/assets/22c56260-3029-4a61-ae90-d925924e8fcf)

Video
![workflow(6)](https://github.com/user-attachments/assets/a13d37da-a62f-43b6-9e92-0c0c5e8592fc)
