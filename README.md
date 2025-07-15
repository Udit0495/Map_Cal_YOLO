# Map_Cal_YOLO

This project, **Map_Cal_YOLO**, is designed to calculate the Mean Average Precision (mAP) for object detection models using the YOLO (You Only Look Once) framework, specifically tailored for evaluating model performance on custom datasets. It provides scripts to compute mAP and generate confusion matrices for object detection tasks, with a focus on the YOLO training dataset format.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview
The Map_Cal_YOLO project is a utility for evaluating object detection models by calculating the Mean Average Precision (mAP) and generating confusion matrices. It is based on the YOLO training dataset format and leverages scripts inspired by the Darknet framework (specifically from [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)). The project is useful for researchers and developers working on object detection tasks who need to evaluate model performance on datasets like COCO, VOC, or custom datasets.

## Features
- **mAP Calculation**: Computes the Mean Average Precision (mAP) for object detection models, including metrics like mAP@0.5 and mAP@0.5:0.95.
- **Confusion Matrix Generation**: Produces confusion matrices to analyze model performance across different classes.
- **YOLO-Compatible**: Works with YOLO-style dataset formats (e.g., `.txt` label files).
- **Customizable Thresholds**: Allows tuning of confidence and IoU thresholds for evaluation.
- **Modular Scripts**: Includes modular Python scripts for inference and evaluation, making it easy to integrate into existing workflows.

## Prerequisites
To use this project, ensure you have the following installed:
- Python >= 3.8
- PyTorch >= 1.8
- CUDA (optional, for GPU acceleration)
- Dependencies listed in `requirements.txt` (e.g., `numpy`, `opencv-python`, `pycocotools`, etc.)
- A YOLO-compatible dataset with images and corresponding `.txt` label files in YOLO format.
- Pre-trained YOLO model weights (e.g., `.weights` or `.pt` files).

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Udit0495/Map_Cal_YOLO.git
   cd Map_Cal_YOLO
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install PyTorch** (if not already installed):
   Follow the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) to install PyTorch with CUDA support (if using a GPU).

5. **Compile Darknet (if using Darknet-based inference)**:
   If you are using the Darknet framework for inference:
   ```bash
   cd darknet
   make clean
   make
   ```

## Dataset Preparation
To evaluate your model, prepare your dataset in the YOLO format:
1. **Directory Structure**:
   ```
   dataset/
   ├── images/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   ├── labels/
   │   ├── image1.txt
   │   ├── image2.txt
   │   └── ...
   └── test.txt
   ```
   - `images/`: Contains `.jpg` images for testing.
   - `labels/`: Contains corresponding `.txt` files with ground-truth annotations in YOLO format (`<class_id> <x_center> <y_center> <width> <height>`).
   - `test.txt`: A text file listing the paths to the test images (one per line).

2. **YOLO Label Format**:
   Each `.txt` file should match the corresponding image filename (e.g., `image1.jpg` → `image1.txt`) and contain lines in the format:
   ```
   <class_id> <x_center> <y_center> <width> <height>
   ```
   Example:
   ```
   0 0.5 0.5 0.2 0.3
   ```

3. **Update Configuration**:
   Modify the settings in `modulized/save_label_as_yolo_format.py` to point to your dataset and model:
   - `FILE_CFG`: Path to the YOLO configuration file (e.g., `yolov4.cfg`).
   - `FILE_WEIGHTS`: Path to the pre-trained weights file (e.g., `yolov4.weights`).
   - `FILE_DATA`: Path to the dataset configuration file (e.g., `coco.data`).
   - `THRESH_YOLO`: Detection threshold (e.g., `0.25` for initial predictions).

## Usage
1. **Generate Predictions**:
   Run the inference script to generate predictions in YOLO format:
   ```bash
   python modulized/save_label_as_yolo_format.py
   ```
   This will save the predicted labels in the `labels_prediction/` directory under your test dataset folder.

2. **Evaluate mAP and Confusion Matrix**:
   Run the evaluation script to compute mAP and generate a confusion matrix:
   ```bash
   python modulized/compare_simple.py
   ```
   - Modify `modulized/compare_simple.py` to adjust settings like `THRESH_CONFIDENCE` for evaluation.
   - Comment/uncomment the relevant sections in `compare_simple.py` to compute `metric.get_mAP` or `metric.get_confusion`.

3. **Output**:
   - **mAP**: The script outputs mAP metrics (e.g., mAP@0.5, mAP@0.5:0.95) for the test dataset.
   - **Confusion Matrix**: A matrix showing the true positives, false positives, and false negatives for each class.

## File Structure
```
Map_Cal_YOLO/
├── darknet/                    # Darknet framework for inference (optional)
├── modulized/                  # Core scripts for evaluation
│   ├── save_label_as_yolo_format.py  # Script to generate predictions
│   ├── compare_simple.py       # Script to compute mAP and confusion matrix
│   └── ...
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a pull request.

Please ensure your code follows the project's coding standards and includes appropriate documentation.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Inspired by [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) for YOLO implementation and evaluation scripts.
- Thanks to the open-source community for providing tools and datasets like COCO and VOC.
- Built with Python, PyTorch, and Darknet.
