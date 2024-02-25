# Introduction
Welcome to our Plant Classification Web Application aka AushadHub! This app uses a pre-trained deep learning model to classify images of plants.


## AushadHub Overview 🌿 :
AushadHub is a project designed to address the challenge of identifying medicinal herbs. It utilizes a Machine Learning Model based on ResNet, achieving a validation accuracy of 98% and a testing accuracy of 96%. The dataset used for training and testing comprises over 1500 images spanning 30 different medicinal herb species.

## Description:
### FrontEnd:
The front end of AushadHub is developed using Flask, HTML, CSS, and a touch of vanilla JavaScript.

### Backend:
The backend consists of an end-to-end Machine Learning solution dedicated to solving the problem of medicinal herb identification. It employs ResNet trained on a custom dataset, achieving a validation accuracy of 98%. The dataset used for training and testing consists of over 1500 images representing 30 different medicinal herb species.


## 📁 Project Structure

- **Hackmol5/**
  - **app.py**: Flask application script
  - **model/**
    - **resnet50-transfer.pth**: Pre-trained ResNet50 model checkpoint
  - **static/**
    - *upload/*: Folder for user-uploaded images
  - **templates/**
    - *index.html*: Homepage template
    - *result.html*: Result page template
  - **info1.csv**: CSV file containing class labels and descriptions
- **requirements.txt**: List of Python dependencies


## 🚀 Quick Start

1. Clone the repository:
```bash
git clone [https://github.com/abhinav0git/HackMol5.git](https://github.com/abhinav0git/HackMol5)
cd HackMol5
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```
3. Run the app:
```bash
python app.py
```
## 🤖 Model Details
The app uses a pre-trained ResNet50 model for plant classification.
Model checkpoint: Hackmol5/model/resnet50-transfer.pth

### Tech Stack Used:
- Python 3
- Flask
- Pillow
- HTML
- CSS
- JavaScript
- GitHub
