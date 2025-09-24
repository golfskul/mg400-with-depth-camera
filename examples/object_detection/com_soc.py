
import socket
import time

IP = "192.168.1.6"
port = 6601

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((IP, port))

def send_position(x, y, z):
    """ส่งพิกัด x,y,z ไปยัง DoBot"""
    message = f"go,{x},{y},{z}\n"   # ต้องมี go นำหน้า และ \n ปิดท้าย
    sock.sendall(message.encode())
    print(f"✅ Sent: {message.strip()}")

# ตัวอย่างการส่งพิกัด
x, y, z = 120, 253, 133
send_position(x, y, z)

sock.close()

