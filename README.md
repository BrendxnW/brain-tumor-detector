<h1 align="center">Brain Tumor Detector</h1>
<p align="center">
  Deep learning–based classification of brain MRI scans into <b>glioma</b>, <b>meningioma</b>, <b>pituitary tumor</b>, and <b>no tumor</b>.
</p>

<p align="center">
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.9.1-orange?logo=pytorch&logoColor=white" />
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green" />
  </a>
</p>

## Table of Contents
- [Security](#security)
- [Background](#background)
- [Demo Video](#demo-video)
- [Project Structure](#project-structure)
- [Install](#install)
- [Usage](#usage)
- [API](#api)
- [Model Details](#model-details)
- [Testing](#testing)
- [Deployment](#deployment)
- [License](#license)


## Security
This project is intended for educational and research purposes only. The model was trained on publicly avalibale datasets and is not valided for clinical use.
This application includes several safeguards to ensure safe useage:
- The model runs in an isolated environment to prevent arbitary code execution.
- Docker containers are used to isolate dependencies and runtime environment.
- No user data or uploaded images are permanently stored on the server. 

## Background
Brain tumors are typically diagnosed through MRI scans, which requires expert interpretation by radiologists. However, manual analysis can be time-consuming and subject to human error.
This project explores the use of deep learning to automatically classify MRI images into thre tumor categories: glioma, meningioma, and pituitary tumors.
The goal of this system is to demonstrate how machine learning can assist medical professionals by providing rapid preliminary classification.

## Demo Video
[Filler]
[Link Filler]

## Project Structure
```text
brain-tumor-detector/
|____app/
|____src/
    |____models/
        |____brain_tumor_bot.py
    |____training/
        |____train.py
    |____utils/
        |____data_loader.py
|____.gitignore
|____README.md
|____requirements.txt

```

## Install
```bash
git clone https://github.com/BrendxnW/brain-tumor-detector.git  
cd brain-tumor-detector

# Windows
python -m venv venv
source venv/Source/activate 

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```
## Usage
### Local:
```bash
python -m src/models/brain_tumor_bot.py --image "[File Path to Image]"
```

### Example:
```bash
python -m src/models/brain_tumor_bot.py --image "data/dataset_1/Testing/pituitary/Te-pi_1.jpg"

# Output
Image: data/dataset_1/Testing/pituitary/Te-pi_1.jpg
Type of Tumor: Pituitary
```

## API
### POST /predict
Uploads an MRI image and returns the predicted tumor class

Requests:
- image: MRI scan file

Reponse:
```json
{
  "prediction": "glioma",
  "confidence": 0.92
}
```

## Model Details
- Architecture: CNN (ResNet18)
- Framework: PyTorch
- Datasets:
    - Brain Tumor MRI Dataset
    - Brain Tumor MRI Dataset for Deep Learning
- Classes:
    - Glioma
    - Meningioma
    - Pituitary
    - No Tumor
 
## Testing

Run tests with:
```bash

```

## Deployment

Backend:  
Frontend:


## License
[MIT © Richard McRichface.](./LICENSE)
