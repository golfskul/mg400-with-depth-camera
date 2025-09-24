# MG400 with depth camera

# 1. clone project from Github
```bash
git clone https://github.com/golfskul/mg400-with-depth-camera.git
```
# 2. check Python version (Python 3.10.4)
```bash
python --version
```
if not; Download Python
```bash
https://www.python.org/downloads/release/python-3104/
```
# 3. Install pyorbbecsdk library
```bash
https://github.com/orbbec/OrbbecSDK_v2
```
# 4. build Virtual Environment (cmd)
```bash
cd your path
python -m venv .venv
```
# 5. use Virtual Environment (cmd)
```bash
.venv\Scripts\activate
```
# 6. Install lib
```bash
pip install -r requirements.txt
```
# 7. Run Program
```bash
.venv\Scripts\activate
python src/test_cal.py
```

# Data Preparation

1. Data Collection
```bash
python camera_capture.py
```
2. Labelling (Roboflow)

3. Train Data
```bash
python new_yolov8.ipynb
```
```

