# Blender Dataset Generator for U-Net 🧠🎥

This project provides a **Blender-based dataset generation pipeline** for training **U-Net–style segmentation models**.  
The scripts automate the generation of **synthetic images** of a target object (prim) under **multiple camera angles, orientations, backgrounds, and occlusions**, enabling robust dataset creation for deep learning.

---

## 📁 Repository Structure

```bash
.
├── Blender_dataset/
│   ├── assets/                     # 3D models, textures, backgrounds
│   ├── blender_scripts/            # Blender Python scripts
│   ├── images/                     # Generated images
│   │
│   ├── data_generator.py           # Core dataset generation script
│   ├── data_generation_no_background.py
│   ├── data_generation_with_background.py
│
└── README.md
```

## 📌 Project Overview

This project provides a **script-based synthetic dataset generator** built on **Blender**, designed for computer vision and robotics research.

The dataset generator:

- Spawns a **target prim (object)** in a Blender scene  
- Captures images from **multiple camera viewpoints**  
- Randomizes:
  - Object orientation  
  - Camera pose  
  - Backgrounds  
  - Occlusions  
- Saves generated images to a **configurable output directory**

This synthetic dataset is well-suited for:

- **U-Net segmentation**
- **Vision-based robotics**
- **Object detection and perception research**

## 🧠 Key Features
- Multi-view camera capture
- Multiple background configurations
- Randomized object orientations
- Occlusion-aware image generation
- Configurable dataset size
- Script-based automation (no manual rendering)

## 🧰 Requirements
### Software
- Blender (3.x recommended)
- Python (bundled with Blender)
⚠️ These scripts must be run inside Blender’s Python environment, not standard Python.

## 🚀 How to Use
### 1️⃣ Open Blender
Launch Blender and open a new or existing scene.

### 2️⃣ Load the Script
Go to:
```bash
Scripting → Text Editor → Open
```
Open one of the following scripts:
- data_generator.py
- data_generation_with_background.py
- data_generation_no_background.py

### 3️⃣ Configure Script Parameters
Inside the script, modify:
- Number of images to generate
- Output directory (out_dir)
- Target object path (prim_path)
```bash
out_dir = "/absolute/path/to/output/images"
prim_path = "/absolute/path/to/target/object"
num_images = 1000
```
### 4️⃣ Run the Script
Click Run Script inside Blender.
The script will automatically:
- Position the object
- Randomize scene parameters
- Render and save images

## 🖼️ Output
Generated images are saved in the specified out_dir, typically organized by:
- Camera angle
- Background type
- Object configuration
These images can be directly used for U-Net training or further annotation pipelines.

## 📌 Notes
- Use absolute paths for reliability
- Rendering quality can be adjusted in render_settings
- Background images should be placed inside the assets folder
- Scripts can be extended to generate masks / labels if required

## 🛠️ Possible Extensions
- Automatic segmentation mask generation
- Depth image export
- Domain randomization (lighting, textures)
- Dataset split (train / val / test)

## 📜 License
This project is intended for academic and research use.
You are free to modify and extend the scripts for your own experiments.

## 👤 Author
Shakthi Bala
Computer Vision | Synthetic Data Generation | Blender | Deep Learning
---

### ✅ Why this README works well
- Matches **your actual repo structure**
- Explains Blender-specific execution clearly
- U-Net and dataset context is explicit
- Clean and recruiter-friendly

If you want next, I can:
- Add **example images section**
- Add **mask generation workflow**
- Convert this into a **paper-ready dataset README**
- Unify styling across all your repos

Just say 👍
