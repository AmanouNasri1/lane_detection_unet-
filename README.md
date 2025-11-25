# 🛣️ Real-Time Lane Detection System (U-Net++)

A deep learning project implementing Semantic Segmentation to detect traffic lanes in unstructured driving environments. Built with **PyTorch** and optimized for real-time inference using a **ResNet34 backbone**.

![Project Demo](assets/demo_screenshot.jpg)
*(Note: Add a screenshot of your result here!)*

## 💡 Overview
This project tackles the challenge of identifying road lane topologies under varying lighting conditions. Unlike traditional computer vision methods (Canny Edge/Hough Transform), this system uses a **data-driven Deep Learning approach** to robustly segment lanes even on curved roads or amidst visual noise.

**Key Features:**
* **Architecture:** Nested U-Net (U-Net++) for capturing fine-grained spatial details.
* **Transfer Learning:** Pre-trained **ResNet34** encoder (ImageNet) for rapid convergence.
* **Metric:** Optimized using **Dice Loss** to handle extreme class imbalance (thin lanes vs. large road background).
* **Performance:** Optimized inference pipeline achieving **~30 FPS** on standard hardware through frame-skipping logic and resolution scaling.

## 🛠️ Tech Stack
* **Language:** Python 3.9+
* **Frameworks:** PyTorch, Segmentation Models PyTorch (SMP)
* **Computer Vision:** OpenCV (cv2), NumPy
* **Dataset:** Custom subset of the [CULane Benchmark Dataset](https://xingangpan.github.io/projects/CULane.html)

## 📊 Performance
The model was trained for 10 epochs on a custom ETL pipeline of the CULane dataset.
* **Final Training Loss (Dice):** 0.1502
* **Estimated Dice Score:** ~0.85
* **Inference Speed:** Real-time (configurable via frame-skip parameters)

## 🚀 How to Run Locally

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/AmanouNasri1/lane_detection_unet-.git](https://github.com/AmanouNasri1/lane_detection_unet-.git)
    cd lane_detection_unet-
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Model Weights**
    Download the trained weights (`lane_model_epoch_10.pth`) from [https://drive.google.com/file/d/1xfN37KhR8ksigBfXNzx35MosMtzNd9U6/view?usp=sharing] and place them in the root folder.
    *(Note: Large model files are not stored in this repo).*

4.  **Run Inference**
    To test on a video file or webcam:
    ```bash
    python inference.py --source video.mp4
    ```

## 🧠 Engineering Challenges
* **Data Imbalance:** Implemented Dice Loss instead of Cross-Entropy to force the model to focus on the thin lane pixels rather than the dominant background.
* **Latency Optimization:** The raw ResNet34 model is computationally heavy. I implemented an asynchronous frame-processing logic (processing every Nth frame) to ensure fluid video playback while maintaining detection accuracy.

## 📜 License
MIT License