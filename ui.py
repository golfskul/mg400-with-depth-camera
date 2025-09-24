import sys, threading, time, traceback
import numpy as np
import cv2

from PySide6.QtCore import Qt, QTimer, Signal, QObject, QPoint
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QGroupBox, QFormLayout, QSpinBox, QPlainTextEdit, QMessageBox
)

# ========= Orbbec SDK =========
try:
    from pyorbbecsdk import Pipeline, Config, FrameSet
    from pyorbbecsdk import StreamType, Format, AlignMode, SensorType
except Exception as e:
    Pipeline = None
    SDK_IMPORT_ERROR = e
else:
    SDK_IMPORT_ERROR = None


def qimg_from_bgr(bgr):
    h, w, ch = bgr.shape
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888)


class OrbbecCamera(QObject):
    frame_ready = Signal(np.ndarray, np.ndarray)  # (color_bgr, depth_mm)
    intrinsics_ready = Signal(dict)
    status = Signal(str)
    error = Signal(str)

    def __init__(self, color_size=(640, 480), depth_size=(640, 480), fps=30, parent=None):
        super().__init__(parent)
        self.color_size = color_size
        self.depth_size = depth_size
        self.fps = fps
        self._pipeline = None
        self._running = False
        self._thread = None
        self._intrinsics = None  # {"fx":..,"fy":..,"cx":..,"cy":..}
        self._align_depth_to_color = True

    def open(self):
        if SDK_IMPORT_ERROR:
            raise RuntimeError(f"Import pyorbbecsdk failed: {SDK_IMPORT_ERROR}")
        if self._pipeline:
            return
        self._pipeline = Pipeline()
        cfg = Config()
        # เปิด stream
        cfg.enable_stream(StreamType.COLOR, self.color_size[0], self.color_size[1], Format.RGB888, self.fps)
        cfg.enable_stream(StreamType.DEPTH, self.depth_size[0], self.depth_size[1], Format.Y16, self.fps)
        if self._align_depth_to_color:
            cfg.set_align_mode(AlignMode.ALIGN_D2C)  # align depth-to-color

        self._pipeline.start(cfg)
        self.status.emit("Camera opened.")

        # ดึง intrinsics จากโปรไฟล์ (ของ color เมื่อ align D2C, จะใช้คู่อินทรินซิกส์ของ color)
        try:
            color_profiles = self._pipeline.get_stream_profile_list(StreamType.COLOR)
            color_profile = color_profiles.get_default_video_stream_profile()
            intr = color_profile.get_intrinsic()
            self._intrinsics = {"fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy}
            self.intrinsics_ready.emit(self._intrinsics)
        except Exception as e:
            # fallback: ใช้ค่ากลางภาพเป็น cx,cy; fx,fy เดาแบบพอใช้ (จะไม่แม่น)
            w, h = self.color_size
            self._intrinsics = {"fx": 600.0, "fy": 600.0, "cx": w/2.0, "cy": h/2.0}
            self.error.emit(f"Get intrinsics failed, using fallback: {e}")

    def close(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        try:
            if self._pipeline:
                self._pipeline.stop()
        except Exception:
            pass
        self._pipeline = None
        self.status.emit("Camera closed.")

    def start_stream(self):
        if not self._pipeline:
            raise RuntimeError("Camera not opened.")
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.status.emit("Streaming...")

    def _loop(self):
        while self._running:
            try:
                frames: FrameSet = self._pipeline.wait_for_frames(1000)
                color = frames.color_frame()
                depth = frames.depth_frame()

                # color to BGR numpy
                color_np = None
                if color:
                    # RGB888 → BGR
                    c = np.frombuffer(color.get_data(), dtype=np.uint8).reshape(
                        (color.get_height(), color.get_width(), 3)
                    )
                    color_np = cv2.cvtColor(c, cv2.COLOR_RGB2BGR)
                # depth mm numpy (Y16)
                depth_np = None
                if depth:
                    d = np.frombuffer(depth.get_data(), dtype=np.uint16).reshape(
                        (depth.get_height(), depth.get_width())
                    )
                    # สมมุติว่าค่า depth เป็น "มิลลิเมตร" (ส่วนมากของ Orbbec เป็น mm)
                    depth_np = d.astype(np.uint16)

                if color_np is not None and depth_np is not None:
                    self.frame_ready.emit(color_np, depth_np)
            except Exception as e:
                self.error.emit(f"Stream error: {e}")
                time.sleep(0.05)
    
    

    def get_xyz_from_pixel(self, u, v):
        """
        รับ (u,v) พิกเซลในภาพ color (เมื่อ align D2C)
        คืนค่า X,Y,Z ในหน่วย mm ตาม intrinsics + depth
        """
        if self._intrinsics is None:
            raise RuntimeError("Intrinsics not ready")
        # ดึงเฟรมล่าสุดผ่านสัญญาณ หรือให้ UI เก็บสำเนา depth ล่าสุดไว้แล้วเรียกที่นี่
        # เพื่อความง่าย เราจะให้ UI ส่ง depth_np ล่าสุดเข้ามาไม่ได้ — จึงย้ายตรรกะนี้ไปไว้ใน UI
        raise NotImplementedError("Use compute_xyz_from(u,v,depth_mm, intrinsics) helper below")


def compute_xyz_from(u, v, depth_mm, intr):
    """
    คำนวณจุด 3D จากพิกเซล (u,v) และค่า depth_mm (มม.)
    intr = {"fx","fy","cx","cy"}
    คืนค่า X,Y,Z หน่วย mm ในเฟรมกล้อง (camera frame)
    """
    z = float(depth_mm)
    if z <= 0:
        return None
    x = (float(u) - intr["cx"]) * z / intr["fx"]
    y = (float(v) - intr["cy"]) * z / intr["fy"]
    return (x, y, z)

# ========= UI =========
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Orbbec 3D → XYZ (PySide6)")
        self.resize(980, 640)

        self.cam = OrbbecCamera()
        self._intr = None
        self._last_color = None
        self._last_depth = None
        self._last_click = None  # QPoint

        # Signals
        self.cam.frame_ready.connect(self.on_frame)
        self.cam.intrinsics_ready.connect(self.on_intrinsics)
        self.cam.status.connect(self.on_status)
        self.cam.error.connect(self.on_error)

        # Widgets
        self.preview = QLabel("(preview)")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet("background:#11111110; border:1px dashed #cbd5e1; border-radius:8px;")
        self.preview.setMinimumSize(640, 360)
        self.preview.mousePressEvent = self.on_preview_click

        self.btn_open = QPushButton("📷 Open Camera")
        self.btn_stream = QPushButton("▶ Start Stream")
        self.btn_center_xyz = QPushButton("🎯 Get XYZ (center)")
        self.btn_close = QPushButton("⏹ Close")

        self.lbl_fx = QLabel("fx: -")
        self.lbl_fy = QLabel("fy: -")
        self.lbl_cx = QLabel("cx: -")
        self.lbl_cy = QLabel("cy: -")

        # XYZ big display
        self.lbl_x = QLabel("X = -- mm"); self._big(self.lbl_x)
        self.lbl_y = QLabel("Y = -- mm"); self._big(self.lbl_y)
        self.lbl_z = QLabel("Z = -- mm"); self._big(self.lbl_z)

        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMinimumHeight(140)

        # Layout
        left = QVBoxLayout()
        g1 = QGroupBox("Camera")
        f1 = QHBoxLayout()
        f1.addWidget(self.btn_open); f1.addWidget(self.btn_stream); f1.addWidget(self.btn_center_xyz); f1.addWidget(self.btn_close)
        g1.setLayout(f1)

        g2 = QGroupBox("Intrinsics")
        f2 = QHBoxLayout()
        f2.addWidget(self.lbl_fx); f2.addWidget(self.lbl_fy); f2.addWidget(self.lbl_cx); f2.addWidget(self.lbl_cy)
        g2.setLayout(f2)

        xyz_box = QGroupBox("XYZ (mm)")
        fx = QVBoxLayout()
        fx.addWidget(self.lbl_x); fx.addWidget(self.lbl_y); fx.addWidget(self.lbl_z)
        xyz_box.setLayout(fx)

        left.addWidget(g1)
        left.addWidget(self.preview, 1)
        left.addWidget(g2)
        left.addWidget(xyz_box)
        left.addWidget(QLabel("Log"))
        left.addWidget(self.log)

        root = QWidget(); root.setLayout(left)
        self.setCentralWidget(root)

        # Wire
        self.btn_open.clicked.connect(self.on_open)
        self.btn_stream.clicked.connect(self.on_stream)
        self.btn_center_xyz.clicked.connect(self.on_get_center)
        self.btn_close.clicked.connect(self.on_close)

        # SDK import check
        if SDK_IMPORT_ERROR:
            self.warn("pyorbbecsdk นำเข้าไม่ได้", f"{SDK_IMPORT_ERROR}\n\nติดตั้ง pyorbbecsdk และ Orbbec SDK ให้เรียบร้อยก่อน")

    # Helpers

    def _big(self, lab: QLabel):
        lab.setStyleSheet("font-size:24px; font-weight:800;")

    def log_line(self, s):
        self.log.appendPlainText(s)

    # Slots
    def on_open(self):
        try:
            self.cam.open()
        except Exception as e:
            self.warn("Open failed", str(e))
            self.log_line(f"Open failed: {e}")

    def on_stream(self):
        try:
            self.cam.start_stream()
        except Exception as e:
            self.warn("Start stream failed", str(e))
            self.log_line(f"Start stream failed: {e}")

    def on_close(self):
        try:
            self.cam.close()
        except Exception as e:
            self.log_line(f"Close error: {e}")

    def on_status(self, s):
        self.statusBar().showMessage(s, 2000)
        self.log_line(s)

    def on_error(self, s):
        self.warn("Error", s)
        self.log_line(f"ERROR: {s}")

    def on_intrinsics(self, intr):
        self._intr = intr
        self.lbl_fx.setText(f"fx: {intr['fx']:.2f}")
        self.lbl_fy.setText(f"fy: {intr['fy']:.2f}")
        self.lbl_cx.setText(f"cx: {intr['cx']:.2f}")
        self.lbl_cy.setText(f"cy: {intr['cy']:.2f}")
        self.log_line("Intrinsics loaded.")

    def on_frame(self, color_bgr, depth_mm):
        self._last_color = color_bgr
        self._last_depth = depth_mm

        # วาด crosshair ถ้ามีจุดที่คลิกล่าสุด
        disp = color_bgr.copy()
        if self._last_click is not None:
            u, v = self._last_click.x(), self._last_click.y()
            cv2.drawMarker(disp, (u, v), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)

        qimg = qimg_from_bgr(disp)
        self.preview.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.preview.width(), self.preview.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

        # ถ้ามีจุดคลิก → คำนวณ XYZ แบบ real-time
        if self._intr and self._last_click is not None:
            u, v = self._map_click_to_image(self._last_click, disp.shape[1], disp.shape[0])
            z = int(depth_mm[v, u]) if (0 <= v < depth_mm.shape[0] and 0 <= u < depth_mm.shape[1]) else 0
            xyz = compute_xyz_from(u, v, z, self._intr)
            if xyz:
                x, y, zmm = xyz
                self.lbl_x.setText(f"X = {x:.2f} mm")
                self.lbl_y.setText(f"Y = {y:.2f} mm")
                self.lbl_z.setText(f"Z = {zmm:.2f} mm")

    def on_get_center(self):
        if self._last_depth is None or self._intr is None:
            self.warn("ยังไม่มีภาพ/อินทรินซิกส์", "กด Open Camera + Start Stream ก่อน")
            return
        h, w = self._last_depth.shape
        u, v = w // 2, h // 2
        z = int(self._last_depth[v, u])
        xyz = compute_xyz_from(u, v, z, self._intr)
        if xyz:
            x, y, zmm = xyz
            self.lbl_x.setText(f"X = {x:.2f} mm")
            self.lbl_y.setText(f"Y = {y:.2f} mm")
            self.lbl_z.setText(f"Z = {zmm:.2f} mm")
            self.log_line(f"Center XYZ: ({x:.2f}, {y:.2f}, {zmm:.2f}) mm")
        else:
            self.warn("Depth ไม่พร้อม", "พิกเซลกลางไม่มีค่า depth หรือเป็น 0")

    def on_preview_click(self, ev):
        # เก็บตำแหน่งคลิกในพิกเซลของ QLabel ก่อน แล้วค่อยแมปไปพิกเซลจริงของภาพ
        self._last_click = ev.position().toPoint()
        self.log_line(f"Click at preview: {self._last_click.x()}, {self._last_click.y()}")

        # ถ้ามีเฟรมและอินทรินซิกส์ → คำนวณทันที
        if self._last_depth is not None and self._intr is not None and self._last_color is not None:
            u, v = self._map_click_to_image(self._last_click, self._last_color.shape[1], self._last_color.shape[0])
            if 0 <= v < self._last_depth.shape[0] and 0 <= u < self._last_depth.shape[1]:
                z = int(self._last_depth[v, u])
                xyz = compute_xyz_from(u, v, z, self._intr)
                if xyz:
                    x, y, zmm = xyz
                    self.lbl_x.setText(f"X = {x:.2f} mm")
                    self.lbl_y.setText(f"Y = {y:.2f} mm")
                    self.lbl_z.setText(f"Z = {zmm:.2f} mm")
                    self.log_line(f"Click XYZ: ({x:.2f}, {y:.2f}, {zmm:.2f}) mm")
                else:
                    self.warn("Depth=0", "ตำแหน่งนี้ไม่มีค่า depth (0)")
            else:
                self.warn("นอกภาพ", "คลิกรนอกขอบภาพจริง")

    def _map_click_to_image(self, pt: QPoint, img_w, img_h):
        """
        แปลงพิกเซลบน QLabel (ที่ resize/letterbox แล้ว) กลับไปเป็นพิกเซลบนภาพจริง
        """
        label_w = self.preview.width()
        label_h = self.preview.height()
        # scale โดยรักษาอัตราส่วน
        scale = min(label_w / img_w, label_h / img_h)
        disp_w = int(img_w * scale)
        disp_h = int(img_h * scale)
        off_x = (label_w - disp_w) // 2
        off_y = (label_h - disp_h) // 2
        u = int((pt.x() - off_x) / scale)
        v = int((pt.y() - off_y) / scale)
        # clamp
        u = max(0, min(img_w - 1, u))
        v = max(0, min(img_h - 1, v))
        return u, v

    def warn(self, title, msg):
        QMessageBox.warning(self, title, msg)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())