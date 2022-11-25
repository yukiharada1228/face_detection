# face_detection
 OpenVINOによる顔検出

# DEMO
<img src="img/demo1.png" width="320">

# Feature
 近くに写った顔から順に，指定した数まで検出します．
 
# Requirements
* Python 3.8.10
* openvino 2021.4.2
* openvino-dev[pytorch] 2021.4.2

# Usage
```bash
git clone https://github.com/yukiharada1228/face_detection.git
cd face_detection
pip install -r requirements.txt
python face_detect.py ([検出数] [閾値])
ex) 
python face_detect.py
python face_detect.py 1
python face_detect.py 1 0.8
```

# Author 
* Yuki Harada
* yukiharada1228@gmail.com
