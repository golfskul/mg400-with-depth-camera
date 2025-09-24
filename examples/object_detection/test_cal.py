import os
import time
import socket
import numpy as np
import cv2
from PySide6.QtCore import Qt, QThread, Signal, QMutex
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton,
                               QVBoxLayout, QHBoxLayout, QMessageBox, QPlainTextEdit, QLineEdit, QFileDialog)

from pixel import pixel_to_world
from pyorbbecsdk import Pipeline, Config, OBSensorType
try:
    from pyorbbecsdk import AlignFilter, OBStreamType
    HAS_ALIGN = True
except Exception:
    HAS_ALIGN = False

from ultralytics import YOLO

# ================= Dobot Socket Client =================
class DobotSocketClient:
    def __init__(self, ip="192.168.1.6", port=6601):
        self.ip = ip
        self.port = port
        self.sock = None
        self.dobot_client = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(2)
        self.sock.connect((self.ip, self.port))

    def send_cmd(self, cmd, wait=0.1):
        if self.sock is None:
            raise RuntimeError("Not connected")
        self.sock.sendall((cmd + "\n").encode())
        time.sleep(wait)
        try:
            resp = self.sock.recv(1024)
            return resp.decode().strip()
        except Exception:
            return ""

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None

# ================= Utils =================
def to_qimage(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3*w, QImage.Format.Format_RGB888)

def decode_color_frame(frame):
    if frame is None:
        return None
    w, h = frame.get_width(), frame.get_height()
    buf = frame.get_data()
    arr = np.frombuffer(buf, dtype=np.uint8)
    if arr.size == w*h*3:
        img = arr.reshape(h, w, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif arr.size == w*h*2:
        yuv = arr.reshape(h, w, 2)
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_YUYV)
    else:
        dec = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return dec

def median_depth_mm(depth_mm, u, v, k=3):
    h, w = depth_mm.shape
    u1, v1 = max(0, u - k), max(0, v - k)
    u2, v2 = min(w - 1, u + k), min(h - 1, v + k)
    patch = depth_mm[v1:v2+1, u1:u2+1]
    vals = patch[patch > 0]
    return float(np.median(vals)) if vals.size else 0.0

def nms(boxes, scores, iou_th=0.45):
    idxs = scores.argsort()[::-1]; keep=[]
    while len(idxs)>0:
        i = idxs[0]; keep.append(i)
        if len(idxs)==1: break
        rest = idxs[1:]
        ious=[]
        for j in rest:
            xx1=max(boxes[i][0],boxes[j][0]); yy1=max(boxes[i][1],boxes[j][1])
            xx2=min(boxes[i][2],boxes[j][2]); yy2=min(boxes[i][3],boxes[j][3])
            w=max(0.0,xx2-xx1); h=max(0.0,yy2-yy1)
            inter=w*h
            a1=(boxes[i][2]-boxes[i][0])*(boxes[i][3]-boxes[i][1])
            a2=(boxes[j][2]-boxes[j][0])*(boxes[j][3]-boxes[j][1])
            u=a1+a2-inter; ious.append(inter/u if u>0 else 0.0)
        idxs=rest[np.array(ious)<=iou_th]
    return keep

def find_rotation_angle(bgr_img, bbox=None):
    """
    Find the main rotation angle (degrees) of the object using HoughLinesP.
    Returns angle in degrees. Positive means rotate counterclockwise.
    """
    img = bgr_img.copy()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        img = img[y1:y2, x1:x2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=10)
    if lines is None or len(lines) == 0:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Normalize angle to [-90, 90]
        if angle < -90:
            angle += 180
        elif angle > 90:
            angle -= 180
        angles.append(angle)

    rot_angle = float(np.median(angles))
    # Normalize angle to [0, 180]
    rot_angle = rot_angle % 180
    return rot_angle
# ================= Worker =================
class Worker(QThread):
    frame = Signal(QImage)
    xyz   = Signal(float, float, float, float, str)
    log   = Signal(str)
    err   = Signal(str) 
    def __init__(self, model="models/best.pt", labels=None, dobot=None):
        super().__init__()
        self.stop_flag = False
        self.mutex = QMutex()
        self.model = model
        self.labels = labels
        self.dobot = dobot 
        self.last_cmd_time = 0 
       

    def filter_pass_high_conf(self, dets, names, conf_th=85):
        """Return only detections with label 'pass' and confidence above threshold."""
        return [d for d in dets if names[d["cls"]] == "pass" and d["conf"] >= conf_th]

    def run(self):
        try:
            pipe = Pipeline(); cfg = Config()
            d_list = pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            if len(d_list) == 0: raise RuntimeError("No depth profiles")
            try: d_prof = d_list.get_default_video_stream_profile()
            except Exception: d_prof = d_list[0].as_video_stream_profile()
            cfg.enable_stream(d_prof)

            c_list = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            c_prof = None
            if len(c_list) > 0:
                try: c_prof = c_list.get_default_video_stream_profile()
                except Exception: c_prof = c_list[0].as_video_stream_profile()
                cfg.enable_stream(c_prof)

            intr = (c_prof.get_intrinsic() if c_prof else d_prof.get_intrinsic())

            pipe.start(cfg)
            self.log.emit("Camera started")

            align = None
            if c_prof and HAS_ALIGN:
                try:
                    align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
                    self.log.emit("AlignFilter enabled")
                except Exception as e:
                    self.log.emit(f"AlignFilter unavailable: {e}")
                    align = None

            if not os.path.exists(self.model): raise FileNotFoundError(self.model)
            yolo_model = YOLO(self.model)
            names = yolo_model.names

            while not self.stop_flag:
                fs = pipe.wait_for_frames(1000)
                if fs is None: continue
                if align is not None:
                    fs = align.process(fs)
                    if fs is None: continue
                    try: fs = fs.as_frame_set()
                    except Exception: pass

                d = fs.get_depth_frame()
                if d is None: continue

                dh, dw = d.get_height(), d.get_width()
                depth_raw = np.frombuffer(d.get_data(), dtype=np.uint16).reshape(dh, dw)
                scale = float(d.get_depth_scale())
                depth_mm = (depth_raw.astype(np.float32) * scale).astype(np.uint16)

                c = fs.get_color_frame() if c_prof else None
                color_bgr = decode_color_frame(c) if c is not None else None
                if color_bgr is None:
                    color_bgr = np.zeros((dh, dw, 3), dtype=np.uint8)

                # YOLOv8 inference
                results = yolo_model(color_bgr)
                dets = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    dets.append({
                        "xyxy": [x1, y1, x2, y2],
                        "conf": conf,
                        "cls": cls_id
                    })

                overlay = color_bgr.copy()
                for det in dets:
                    x1, y1, x2, y2 = det["xyxy"]; conf = det["conf"]; cls_id = det["cls"]
                    label = f"{names[cls_id] if cls_id < len(names) else cls_id}:{conf:.2f}"
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(overlay, label, (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                # Only use 'pass' and high confidence for X,Y,Z
                pass_dets = self.filter_pass_high_conf(dets, names, conf_th=0.5)
                if pass_dets:
                    best = max(pass_dets, key=lambda d: d["conf"])
                    bx1, by1, bx2, by2 = best["xyxy"]
                    target_uv = ((bx1 + bx2)//2, (by1 + by2)//2)
                    target_label = names[best["cls"]] if best["cls"] < len(names) else str(best["cls"])
                    cv2.circle(overlay, target_uv, 5, (0,0,255), -1)
                    angle = find_rotation_angle(color_bgr, bbox=(bx1, by1, bx2, by2))
                    self.log.emit(f"Rotation angle for pass object (HoughLinesP): {angle:.2f} deg")
                    print(f"Rotation angle for pass object (HoughLinesP): {angle:.2f} deg")

                else:
                    target_uv = None
                    target_label = ""

                if target_uv is not None:
                    u, v = target_uv
                    Z = median_depth_mm(depth_mm, u, v, 3)
                    if Z > 0:
                        # X = (u - intr.cx) * Z / intr.fx
                        # Y = (v - intr.cy) * Z / intr.fy
                        
                        # pixel from depth
                        X_cm, Y_cm = pixel_to_world(u, v) 
                        self.xyz.emit(float(X_cm), float(Y_cm), float(Z), float(angle), target_label)
                        print(f"\nPixel ({u}, {v}) → World ({X_cm:.2f} cm, {Y_cm:.2f} cm)")

                        # extra = int(X_cm // 20)
                        # move_x = -87.66 + 10 * X_cm + extra #P9
                        # move_y = 237.81 - (10 * Y_cm)
                        
                        # cal move + offset
                        u_move = ((X_cm)-29.5) + 2
                        v_move = ((Y_cm) - 16) -0.5
                        # x_move = -83.67 + (10)*(u_move) 
                        # y_move = 230.57 - (10)*(v_move)
                        
                        #cal move step robot
                        x_move = 10*(u_move)
                        y_move = -10*(v_move)
                        rotate = 180- angle
                        now = time.time()

                        # delay send command to robot
                        if self.dobot and (now - self.last_cmd_time >= 10):
                            try:
                               
                                cmd = f"go,{x_move:.2f},{y_move:.2f},{-30},{rotate}" # fix z,angle
                                print(cmd)
                                resp = self.dobot.send_cmd(cmd)
                                self.log.emit(f" {cmd} | Resp: {resp}")
                                self.last_cmd_time = now
                            except Exception as e:
                                self.log.emit(f"Dobot send error: {e}")
                        else:
                            self.log.emit("Skip sending command (waiting cooldown 1 min)")
                    else:
                        self.xyz.emit(0.0, 0.0, 0.0, 0.0, target_label)

                self.frame.emit(to_qimage(overlay))

        except Exception as e:
            self.err.emit(str(e))

    def stop(self):
        self.stop_flag = True
        self.wait(1000)

# ================= Main UI =================
class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Prediction")
        self.resize(1100, 850)
        self.worker = None

        self.dobot_client = None  # <-- Add this line

        self.robot_ip_input = QLineEdit("192.168.1.6")
        self.btn_connect_robot = QPushButton("Connect Robot")
        self.btn_connect_robot.setEnabled(True)

        self.label_img = QLabel()
        self.label_img.setAlignment(Qt.AlignCenter)
        self.label_img.setMinimumSize(960, 720)
        self.label_xyz = QLabel("X: 0.0, Y: 0.0, Z: 0.0, R: 0.0, Label: ")
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.model_input = QLineEdit("models/best.pt")
        self.btn_browse = QPushButton("Browse Model")

        hlayout = QHBoxLayout()
        hlayout.addWidget(self.btn_start)
        hlayout.addWidget(self.btn_stop)
        hlayout.addWidget(QLabel("Model:"))
        hlayout.addWidget(self.model_input)
        hlayout.addWidget(self.btn_browse)

        hlayout.addWidget(QLabel("Robot IP:"))
        hlayout.addWidget(self.robot_ip_input)
        hlayout.addWidget(self.btn_connect_robot)

        vlayout = QVBoxLayout()
        vlayout.addWidget(self.label_img)
        vlayout.addWidget(self.label_xyz)
        vlayout.addLayout(hlayout)
        vlayout.addWidget(self.log_box)

        widget = QWidget()
        widget.setLayout(vlayout)
        self.setCentralWidget(widget)

        self.btn_start.clicked.connect(self.start_worker)
        self.btn_stop.clicked.connect(self.stop_worker)
        self.btn_browse.clicked.connect(self.browse_model)
        self.btn_connect_robot.clicked.connect(self.connect_robot)

    def start_worker(self):
        if self.worker is not None:
            return
        model_path = self.model_input.text()
        self.worker = Worker(model=model_path, dobot=self.dobot_client)
        # self.worker = Worker(model=model_path)
        self.worker.frame.connect(self.update_frame)
        self.worker.xyz.connect(self.update_xyz)
        self.worker.log.connect(self.log_box.appendPlainText)
        self.worker.err.connect(self.show_error)
        self.worker.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.log_box.appendPlainText("Worker started.")
        

    def stop_worker(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.log_box.appendPlainText("Worker stopped.")

    def connect_robot(self):
        ip = self.robot_ip_input.text()
        try:
            self.dobot_client = DobotSocketClient(ip=ip)
            self.dobot_client.connect()
            self.log_box.appendPlainText(f"Connected to robot at {ip}")
            self.btn_connect_robot.setEnabled(False)
        except Exception as e:
            self.log_box.appendPlainText(f"Failed to connect to robot: {e}")

    def update_frame(self, qimg):
        self.label_img.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.label_img.width(), self.label_img.height(), Qt.KeepAspectRatio))

    def update_xyz(self, x, y, z, angle, label):
        self.label_xyz.setText(f"X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}, R: {angle:.1f},Label: {label}")

    def show_error(self, msg):
        QMessageBox.critical(self, "Error", msg)
        self.log_box.appendPlainText(f"Error: {msg}")

    def browse_model(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select YOLO model", "", "PyTorch Model (*.pt)")
        if fname:
            self.model_input.setText(fname)
            self.log_box.appendPlainText(f"Model set: {fname}")

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    win = Main()
    win.show()
    sys.exit(app.exec())