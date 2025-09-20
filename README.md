# Banana Decay Assessment Via Knowledge Distillation

## Abstract

Bananas are among the most consumed fruits worldwide but are highly perishable, with a shelf life of only a few days—contributing significantly to food waste. Ripeness assessment typically relies on human judgment, which is inconsistent and prone to errors. This project presents a lightweight deep learning framework for automated banana ripeness classification using knowledge distillation.

A large ResNet152 teacher model (200 MB) was used to transfer both hard-label accuracy and soft-label probability distributions to a significantly smaller ResNet10 student model. The student model attained an accuracy of 97.12%, closely approximating the teacher model’s 97.75%, while being substantially smaller. After converting to TensorFlow Lite, the model size was reduced to 7.5 MB, making it ideal for deployment on low-resource devices.

The model is integrated into a web application designed for small retailers and vendors with limited computational capacity. Users can upload banana images, receive immediate ripeness predictions, and store both results and images in the cloud for future retraining. This work demonstrates the potential of knowledge distillation for resource-aware agricultural AI solutions.

---

## Results

You can showcase your results here using images, graphs, or tables.  
For example:
```
![Confusion Matrix](images/confusion_matrix.png)
![Sample Prediction](images/sample_prediction.png)
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

Clone the repository:
```bash
git clone https://github.com/AYOCODEE/Banana-Decay-Assessment-Via-Knowledge-Distillation.git
cd Banana-Decay-Assessment-Via-Knowledge-Distillation
```

Install dependencies:
```bash
pip install tensorflow keras scikit-learn matplotlib seaborn jupyter
```
Or, if you use a requirements file:
```bash
pip install -r requirements.txt
```

---

## Running the Notebook

Open the notebook:
```bash
jupyter notebook files/Banana_Resnet152.ipynb
```
Follow the instructions in the notebook. Make sure your image dataset is available at the path specified, or update the notebook paths as needed.

---

## References

- Main code and experiments: [`files/Banana_Resnet152.ipynb`](files/Banana_Resnet152.ipynb)

---

## Contributing

Feel free to open issues or pull requests to improve the project.

---
