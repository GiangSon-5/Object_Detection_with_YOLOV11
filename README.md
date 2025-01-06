# Steps Taken in the Project

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
