# Chest X-Ray Pneumonia and Covid-19 Detection

This repository contains projects focused on detecting pneumonia and Covid-19 using chest X-ray images. The projects utilize deep learning techniques to classify images and detect anomalies effectively.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The main goal of these projects is to develop machine learning models that can accurately detect pneumonia and Covid-19 from chest X-ray images. The models are trained on datasets of pediatric chest X-ray images and are evaluated for their diagnostic accuracy.

## Dataset

The datasets used in these projects contain X-ray images categorized into pneumonia, normal, and Covid-19 cases, organized in `train`, `test`, and `val` folders. The datasets are sourced from reputable medical centers and are publicly available for research purposes.

- **Pneumonia Dataset:**
  - **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
  - **Additional Data:** [Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj/2)
- **Covid-19 Dataset:**
  - **Source:** [Cell Full Text](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5)

## Installation

To run the notebooks and reproduce the results, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/chest-xray-detection.git
   cd chest-xray-detection
   ```

2. Create a virtual environment and install the required packages:
   ```bash
   python3 -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```

3. Download the datasets and place them in the appropriate directory:
   ```bash
   mkdir data
   # Move the downloaded datasets into the data directory
   ```

## Usage

To run the analysis, open the Jupyter notebooks and execute the cells in order:

1. `train.ipynb`: This notebook contains the training process for the pneumonia and Covid-19 detection models.
2. `test.ipynb`: This notebook contains the testing and evaluation process for the trained models.

## Methodology

### Pneumonia Detection

1. **Data Preprocessing:** Quality control screening, image resizing, and normalization.
2. **Model Training:** Using convolutional neural networks (CNN) to train the model on the X-ray images.
3. **Evaluation:** Assessing the model's performance using accuracy, precision, recall, and F1-score metrics.

### Covid-19 Detection

1. **Data Splitting:** Dividing the dataset into training, validation, and test sets.
2. **Data Augmentation:** Applying techniques such as rotations, horizontal flipping, and zooming to improve generalization.
3. **Model Architecture:** Implementing binary and multiclass classification models using CNNs and transfer learning with VGG16.
4. **Anomaly Detection:** Utilizing autoencoders for unsupervised learning to detect anomalies in chest X-ray images.
5. **Embedding and KNN:** Extracting embeddings from the multi-class model and using k-Nearest Neighbors for classification.

## Results

### Pneumonia Detection

The trained model achieves a high accuracy in detecting pneumonia from chest X-ray images. Detailed results and visualizations can be found in the `test.ipynb` notebook.

### Covid-19 Detection

The models for Covid-19 detection showed significant performance, with binary classification achieving over 89% accuracy and multi-class classification reaching 82% accuracy. Anomaly detection and KNN classification further enhanced the robustness of the system.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Let me know if you need any additional sections or modifications to this README.
