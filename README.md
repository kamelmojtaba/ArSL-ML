# Arabic Sign Language Comparative Analysis

## Project Overview

This project aims to develop and implement an innovative Arabic Sign Language (ArSL) recognition system to bridge communication gaps for the deaf and hard-of-hearing community. Utilizing advanced deep learning models, we conducted a comparative analysis to determine the most effective model based on accuracy, efficiency, and practical usability.

## Technical Details

### Development Environment
- **Jupyter Notebook:** Used for developing and testing the project.
- **Python:** Primary programming language.

### Libraries and Frameworks
- **TensorFlow:** Used for building and training deep learning models.
- **Keras:** High-level API for TensorFlow to simplify model building.
- **OpenCV:** Used for image processing tasks.
- **YOLO (You Only Look Once):** Utilized for hand detection in preprocessing.
- **Scikit-learn:** Used for data partitioning and evaluation metrics.
- **NumPy:** For numerical operations.
- **Matplotlib:** For visualizing training and validation results.
- **Pandas:** For data manipulation and analysis.

### Models Implemented
- **Convolutional Neural Network (CNN):** Effective for recognizing complex visual patterns in images.
- **Long Short-Term Memory (LSTM):** Designed for processing sequences and capturing temporal dynamics.
- **Residual Network (ResNet):** Leveraged pre-trained ResNet architecture for enhanced feature extraction.

### Preprocessing
- **YOLO Model:** Used for detecting and isolating hand regions in images.
- **Image Resizing:** Standardized to 64x64 pixels for consistency.

### Data Partitioning
- **Training Dataset:** Used for model training.
- **Validation Dataset:** Used for hyperparameter tuning and preventing overfitting.
- **Testing Dataset:** Used for final evaluation of model performance.

### Evaluation Metrics
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

### Evaluation Matrix
| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| ResNet| 89.65%   | 89.73%    | 89.40% | 89.33%   |
| CNN   | 93.21%   | 93.35%    | 93.33% | 93.19%   |
| LSTM  | 34.59%   | 34.15%    | 34.51% | 33.11%   |

### Key Results
- **CNN:** Achieved the highest accuracy at 93.21%, demonstrating strong feature recognition capabilities.
- **ResNet:** Also performed well with 89.65% accuracy, leveraging deep architecture for effective feature extraction.
- **LSTM:** Achieved lower accuracy at 34.59%, indicating challenges in handling high-dimensional gesture data.

## Full Report
For a detailed explanation of the project, including methodology, results, and future work, please refer to the full project report available [here](https://drive.google.com/file/d/1Q0vsDHZrXOzQiogw7wuuIPbj5_reH1Iu/view?usp=sharing).

## How to Run the Project
1. Clone the repository: 
    ```sh
    git clone https://github.com/kamelmojtaba/ArSL-ML.git
    ```
2. Navigate to the project directory:
    ```sh
    cd ArSL-ML
    ```
3. Install the required libraries:
    ```sh
    pip install -r requirements.txt
    ```
4. Open the Jupyter Notebook:
    ```sh
    jupyter code
    ```
5. Run the cells in the notebook to execute the project.


## Acknowledgements
- Special thanks to Multimedia University for supporting this research.
- References and academic papers that inspired this project.
