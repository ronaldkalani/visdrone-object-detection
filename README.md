
#  VisDrone Object Detection (YOLOv5 + Streamlit)

This Streamlit app performs real-time **object detection** on the [VisDrone2019](http://aiskyeye.com/) dataset using a **YOLOv5 model**.

It allows users to:
- Browse and select drone imagery
- Run YOLOv5 detection locally or on Streamlit Cloud
- View results as images and structured detection tables

---

## Live App

👉 [Click here to launch the Streamlit app](https://streamlit.io/cloud) (Once deployed)

---

## 📂 Project Structure

```
visdrone-object-detection/
├── app.py                 # Streamlit application
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## Try Locally

```bash
# Clone the repo
git clone https://github.com/<your-username>/visdrone-object-detection.git
cd visdrone-object-detection

# (Optional) Create a virtual environment
python -m venv env
env\Scripts\activate        # Windows
source env/bin/activate       # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 🛠 Features

- **YOLOv5s model** for object detection
- Image browsing from local dataset
- Detection table view (coordinates, confidence, label)
- Export-ready for further tracking or labeling

---

## Requirements

- Python 3.10 (recommended)
- Streamlit ≥ 1.34.0
- PyTorch ≥ 2.0.1
- NumPy ≤ 1.26.4 (⚠️ for compatibility)

---

## Dataset (VisDrone2019)

This project uses aerial images from the [VisDrone2019-DET-train](http://aiskyeye.com/) dataset.  
Please download and organize the dataset in the following structure if running locally:

```
VisDrone2019-DET-train/
└── images/
    ├── 0000001_00000_d_0000001.jpg
    ├── ...
```

---

## 📄 License

This project is licensed under the MIT License.

---

## Credits

Built by [Ronald Kalani](https://github.com/ronaldkalani)  
Powered by: [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5), Streamlit, PyTorch
