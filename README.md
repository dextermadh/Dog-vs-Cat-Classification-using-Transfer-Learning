# 🐱🐶 Dog vs Cat Image Classification

This project classifies images of dogs and cats using transfer learning with MobileNetV2 in TensorFlow/Keras. It includes preprocessing, data augmentation, model training, fine-tuning, and evaluation with classification reports and confusion matrices.

---

## 📁 Dataset

The dataset used is the **Dogs vs. Cats** dataset from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data).

you can download the dataset to your working directory using this 
```python
kaggle competitions download -c dogs-vs-cats
```

**Final Folder Structure:**
```
data/
└── train/
    ├── cat/
    └── dog/
```

---

## 🛠️ Features

- ✅ Corrupted and blurry image detection
- ✅ Dataset reorganization into `cat` and `dog` folders
- ✅ Data augmentation (flip, rotation, zoom, contrast)
- ✅ Transfer learning with `MobileNetV2`
- ✅ Fine-tuning of top layers
- ✅ Early stopping and learning rate reduction callbacks
- ✅ Evaluation with classification report and confusion matrix
- ✅ Single image prediction support

---

## 🧪 Requirements

- Python 3.8+
- TensorFlow
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- PIL (Pillow)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Model Training

```python
cnn.fit(train_generator, validation_data=val_generator, epochs=5)

# Fine-tuning
cnn.fit(train_generator, validation_data=val_generator, epochs=10, callbacks=callbacks)
```

---

## 📊 Evaluation

Generate a classification report and confusion matrix:

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
sn.heatmap(cm, annot=True, fmt='d', xticklabels=['cat', 'dog'], yticklabels=['cat', 'dog'])
plt.show()

# Classification report
print(classification_report(y_true, y_pred, target_names=['cat', 'dog']))
```

---

## 🖼️ Predicting a Single Image

```python
from tensorflow.keras.preprocessing import image
import numpy as np

img_path = 'path_to_image.jpg'
img = image.load_img(img_path, target_size=(180, 180))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

prediction = cnn.predict(img_array)
class_names = ['cat', 'dog']
predicted_class = class_names[np.argmax(prediction)]
print(f"Predicted: {predicted_class}")
```

---

## 📈 Sample Results

- Accuracy: ~90% after fine-tuning
- Confusion Matrix and classification report available in the notebook

---

## 📚 References

- [Kaggle Dogs vs Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- TensorFlow Documentation

---

## 🧠 Future Improvements

- Unfreeze more layers with layer-wise learning rates
- Test on external datasets
- Deploy using Streamlit or Flask for web inference

---

## 📝 License

MIT License. Feel free to use, modify, and distribute.


