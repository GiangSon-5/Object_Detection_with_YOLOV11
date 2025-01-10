# Project Overview

# I am using Kaggle's GPU for my project, and here is the link to my project (If you can't access it, it's because I set it to private mode): 
[Kaggle Notebook: Q2: Object Detection](https://www.kaggle.com/code/nguyenquyetgiangson/q2-object-detection)

# DEMO:
![Demo1](https://github.com/GiangSon-5/Object_Detection_with_YOLOV11/blob/main/images/demo1.jpg)

![Demo2](https://github.com/GiangSon-5/Object_Detection_with_YOLOV11/blob/main/images/demo2.jpg)

![Demo5](https://github.com/GiangSon-5/Object_Detection_with_YOLOV11/blob/main/images/demo5.jpg)

# 1. Steps Taken in the Project

This project focuses on object detection using YOLOv11 to identify five types of vehicles: Ambulance, Bus, Car, Motorcycle, and Truck. The following steps were performed:

## Data Preparation
- The dataset was reorganized to match YOLO's required structure.
- Images and labels were separated into train, valid, and test directories, ensuring compatibility with the YOLO framework.

## Configuration Creation
- A configuration file (`vehicle_classes_config.yaml`) was created, specifying dataset paths, class names, and the number of classes.

## Model Training
- The pre-trained YOLOv11 model (`yolo11m.pt`) was fine-tuned on the prepared dataset.
- Training parameters included 30 epochs, a batch size of 16, and a learning rate of 0.001, with additional optimizations like AdamW optimizer and warmup epochs.

## Evaluation and Prediction
- The model's performance was assessed using mAP (Mean Average Precision) @ 50 for all classes.
- Predictions on test samples were visualized with bounding boxes overlayed on the images.

## Model Export
- The trained model was exported in ONNX format for further deployment or integration.

# 2. Tools and Libraries Used

## Programming Language
- Python

## Libraries
- `os`, `shutil`: File and directory management.
- `matplotlib`: Visualization of prediction results.
- `ultralytics`: YOLO model training, validation, and prediction.

## Environment
- Kaggle Notebook, which provides GPU support and integrated dataset handling.

# 3. Overall Performance Metrics
- **mAP@50**: 0.713
- **Precision**: 0.693
- **Recall**: 0.725

### Reasons for Performance
- **Imbalanced Dataset**: The five labels were not equally represented, leading to better performance on dominant classes like Car and Ambulance while underrepresented classes (Truck, Motorcycle) performed poorly.
- **High Precision**: Indicates fewer false positives, but the imbalance likely skewed predictions toward majority classes.
- **Moderate Recall**: Suggests missed detections, likely due to insufficient samples for some classes.

### Suggestions for Improvement
- Balance the dataset with augmentation or sampling techniques.
- Use class-weighted loss to mitigate class imbalance.

![Val](https://github.com/GiangSon-5/Object_Detection_with_YOLOV11/blob/main/images/val.jpg)

# 4. Model Evaluation and Improvements
### Strengths
- Reliable detection for well-represented classes.
- Reasonable precision across all classes.

### Limitations
- Poor performance for underrepresented classes.
- Struggles with edge cases and occlusions.

### Improvements
- Augment the dataset to increase diversity and balance.
- Optimize hyperparameters for better generalization.
- Test on unseen datasets to evaluate robustness.

# 5. Explanation of Project Files

## Dataset
Organized into a directory structure:
- `/images/train`, `/images/valid`, `/images/test`
- `/labels/train`, `/labels/valid`, `/labels/test`

## Configuration File (vehicle_classes_config.yaml)
- Specifies dataset paths, number of classes (`nc=5`), and class names.

## Python Script
- Handles data preparation, model training, validation, and predictions.
- Includes code for exporting the model in ONNX format.

## Visualization
- Results were saved and visualized using Matplotlib.

# 6. YOLOv11 Integration in the Project

The integration of YOLOv11 followed these steps:

## Model Initialization
- The pre-trained YOLOv11 weights were loaded using the Ultralytics library.

## Training
- The model was trained on the provided dataset with fine-tuning to adapt to the specific classes.

## Validation and Export
- Post-training, the model was validated and exported to ONNX format for deployment.

## Prediction
- Predictions were run on the test set, and results were visualized with bounding boxes.

This structured integration of YOLOv11 ensured efficient model training and deployment while maintaining flexibility for further improvements.














