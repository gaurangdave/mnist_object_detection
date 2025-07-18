# MNIST Multi-Digit Object Detection with Transfer Learning

## 🎯 Project Goal

This project advances beyond simple classification to tackle a more complex computer vision task: **multi-digit object detection**. The primary goal is to build an end-to-end application that can identify and localize multiple handwritten digits drawn on a single canvas.

This project will serve as a deep dive to solidify my understanding of several key deep learning concepts:

1. **Custom Data Generation:** Programmatically creating a labeled dataset for a novel task.

2. **Object Detection Principles:** Applying YOLO-style grid-based detection logic.

3. **Transfer Learning:** Using a pre-trained CNN as a feature-extracting backbone.

4. **End-to-End Deployment:** Creating an interactive web application for real-time inference.

The final application will allow a user to draw multiple digits on a canvas, and a trained deep learning model will draw bounding boxes around each detected digit with its predicted class.

## ✅ Solution Details

### 🧮 Performance Measure

To evaluate and compare models, the following metrics will be used:

* **Intersection over Union (IoU):** To measure the accuracy of bounding box localization.

* **Mean Average Precision (mAP):** The standard metric for object detection model performance.

* **Class-wise F1 Score & Confusion Matrix:** To analyze the classification accuracy for individual digits.

* **Learning Curves:** To diagnose model behavior during training.

### 🚧 Data Transformation & Generation

A key challenge of this project is that a standard object detection dataset for MNIST does not exist. A custom, synthetic dataset will be programmatically generated with the following pipeline:

1. **Base Digit Augmentation:** The original MNIST digits will be diversified using on-the-fly `tf.keras.layers` for data augmentation, including `RandomTranslation`, `RandomZoom`, and `RandomRotation`.

2. **Composite Image Creation:** A script will generate `100x100` grayscale canvases, placing a variable number of augmented digits at random, non-overlapping locations.

3. **Automated Bounding Box Labeling:** For each digit placed on a canvas, its tight bounding box coordinates (`x, y, width, height`) will be programmatically calculated and saved as the label for that image.

### 📂 Dataset

The training data will be generated from the base MNIST dataset. The generation script will be configurable to produce datasets of varying complexity (e.g., varying the number of digits per image) to support an iterative training strategy.

### 📒 Notebooks

*(This section will be updated as notebooks are created)*

* `01_data_generation.ipynb`: Script to generate and save the synthetic object detection dataset. (TBD)

* `02_model_training.ipynb`: Building, training, and evaluating the object detection model. (TBD)

* `03_model_conversion_and_deployment.ipynb`: Exporting the final model and converting to TensorFlow.js. (TBD)

### 🧠 Model Insights

*(This section will be filled out after model training and evaluation)*

**Production Model:**

* **Architecture**: Transfer Learning using a pre-trained backbone (e.g., MobileNetV2) with a custom YOLO-style detection head. (TBD)

* **Performance**:

  * Mean Average Precision (mAP): **TBD**

  * Inference Speed: **TBD**

* **Training Details**:

  * The model will be trained using a "curriculum learning" approach, starting with simple images (1 digit) and progressively increasing the difficulty (multiple digits).

**Observations:**

* TBD

## 💻 Tech Stack

## 🎯 Project Game Plan

To manage the complexity of this project, an iterative, "curriculum learning" approach will be used:

* **Stage 1: The Configurable Data Script:** The first and most important step is to build the robust, configurable data generation script. This script will be the foundation for all subsequent stages.

* **Stage 2: "Hello World" - Single Digit Detection:**

  * Configure the script to generate images with only **one** randomly placed digit.

  * Train the full YOLO-style model on this simple task to validate that the entire pipeline (data generation, model architecture, loss function, and training loop) is working correctly.

* **Stage 3: Increasing Difficulty:**

  * Use the script to generate images with 3-5 digits.

  * Continue training (fine-tune) the model from Stage 2 on this harder dataset.

* **Stage 4: The Final Model:**

  * Train on the most complex dataset (e.g., up to 10 digits, with overlaps and scale variation) to produce the final, robust model for deployment.

## 🏫 Lessons Learnt

*(This section will be updated throughout the project)*

1. **Data Engineering for CV:** TBD

2. **Object Detection Architecture:** TBD

3. **Transfer Learning for Detection:** TBD

## 🚀 About Me

A jack of all trades in software engineering, with 15 years of crafting full-stack solutions, scalable architectures, and pixel-perfect designs. Now expanding my horizons into AI/ML, blending experience with curiosity to build the future of tech—one model at a time.

## 🔗 Links