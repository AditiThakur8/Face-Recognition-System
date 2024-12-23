# Face Recognition System

## Overview
This project implements a real-time face recognition system using the **K-Nearest Neighbors (KNN)** algorithm. By leveraging OpenCV for face detection and a pre-trained dataset, the system identifies individuals in real-time via a webcam feed.

---

## Features
- **Real-time Face Detection**: Utilizes Haar Cascade for detecting faces.
- **Face Recognition**: Employs KNN for classifying faces based on preprocessed data.
- **Custom Dataset Support**: Easily expandable to include new faces by adding `.npy` files.
- **Scalable and Lightweight**: Optimized for quick execution.

---

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - `OpenCV` for image processing and face detection.
  - `NumPy` for numerical computations.
- **Algorithm**: K-Nearest Neighbors (custom implementation).

---

## Dataset Preparation
1. **Dataset Structure**:
   - Each `.npy` file represents face embeddings for a specific individual.
   - File naming convention: `person_name.npy` (e.g., `john_doe.npy`).
2. **Data Directory**:
   - Store `.npy` files in a `./data/` directory.

---

## Installation and Usage

### Prerequisites
Ensure the following are installed:
- Python 3.8+
- OpenCV
- NumPy

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Face-Recognition-KNN.git
   cd Face-Recognition-KNN
   ```
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Place `.npy` files in the `./data/` directory.

### Running the Project
1. Launch the face recognition system:
   ```bash
   python face_recognition.py
   ```
2. The webcam feed will open, detecting and recognizing faces in real-time.
3. Press `q` to exit.

---

## How It Works
1. **Face Detection**:
   - Haar Cascade identifies faces in the webcam feed.
2. **Feature Extraction**:
   - Detected faces are resized to 100x100 and flattened into feature vectors.
3. **Classification**:
   - KNN algorithm calculates distances between the test face and training data to classify the face.
4. **Display**:
   - Recognized names are displayed on the webcam feed.

---

## Results
- **Accuracy**: Achieves high recognition rates with well-prepared datasets.
- **Real-time Performance**: Processes webcam feed efficiently for smooth recognition.

---

## Challenges Faced
- **Lighting Conditions**: Variations in lighting can affect detection.
- **Dataset Quality**: Recognition accuracy heavily depends on the quality and size of the dataset.
- **Real-time Optimization**: Balancing accuracy and performance.

---

## Future Scope
- **Add Support for Deep Learning Models**: Integrate CNNs for enhanced accuracy.
- **Live Training**: Allow adding new faces directly through the webcam.
- **Web Dashboard**: Develop a web interface for easier interaction and dataset management.

---

## Contributing
Contributions are welcome! Feel free to fork the repository, create a feature branch, and submit a pull request.

---


## Acknowledgments
- OpenCV community for their incredible library.
- Coding Club has inspired this project.

---

## Contact
For queries or collaboration:
- **Name**: Aditi Thakur
- **Email**: aditithakur907@gmail.com
- **GitHub**: [AditiThakur8](https://github.com/AditiThakur8)
