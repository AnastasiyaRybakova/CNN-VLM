# CNN-VLM is a hybrid deep learning framework designed for real-time emotion recognition and semantic interpretation. 

### The system combines:

- A CNN-based emotion recognition network for fast frame-level inference
- A Vision-Language Model (VLM) for higher-level contextual interpretation
- Real-time video stream processing
- Emotion smoothing and stability control

The architecture is designed for use in human-robot interaction and adaptive educational systems.

## Pretrained model & Training dataset

1. Please download pretrained face recognition model from here: https://github.com/TreB1eN/InsightFace_Pytorch
2. Download fer2013 dataset: https://www.kaggle.com/msambare/fer2013

## System Architecture

### The framework follows a two-stage inference pipeline:

#### Fast CNN Inference (Frame-level)
- Performs real-time facial emotion classification
- Lightweight and optimized for continuous execution

#### Periodic VLM Inference
- Processes selected frames
- Generates semantic contextual interpretation
- Enhances emotional awareness beyond raw classification

This hybrid architecture enables high-speed inference while preserving semantic depth.

### Key Features
- Real-time webcam or video stream processing
- CNN-based emotion classification
- Vision-Language semantic analysis
- Emotion smoothing to reduce prediction noise
- Modular and extensible architecture
- Designed for robotics integration

<img width="1641" height="866" alt="Fig 1 (1)" src="https://github.com/user-attachments/assets/b6ba9306-3b50-4344-8651-75253c58bf5c" />

### Emotion_recognition
#### Classify emotion for 5 cases: 1.angry 2.happy 3.sad 4.surprise 5.neutral

<img width="851" height="432" alt="6 (1)" src="https://github.com/user-attachments/assets/66b7b008-3bdf-4b25-a670-a1ff929784a1" />

The real evironment experiment of CNN+VLM pilot study:

<img width="1095" height="940" alt="Fig 2 (1)" src="https://github.com/user-attachments/assets/cb399904-1f5a-4297-931b-7db35bb93b8e" />

These images show runtime behavior of the integrated CNN + Vision-Language pipeline.

### How to run the source code: 
    
    python main.py 
    
For the system we have used RealSnse D435 camera and linux system.
