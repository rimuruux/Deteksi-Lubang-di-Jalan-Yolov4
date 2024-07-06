from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import requests
import cv2
import numpy as np
import threading
import time

# inisialisasi aplikasi flask
app = Flask(__name__)
# inisialisasi Flask-SocketIO dengan mode threading
socketio = SocketIO(app, async_mode='threading')

ESP32_STREAM_URL = "http://192.168.1.4:81/stream"  # Gantilah dengan alamat IP ESP32-CAM Anda

# memuat nama kelas dari file obj.names
classesfile = 'D:\DETEKSI LUBANG DIJALAN/obj.names'
classNames = []
with open(classesfile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
    
# memuat model yang didapat dari training melalui google colab
modelConfig = 'D:\DETEKSI LUBANG DIJALAN/yolov4_tiny.cfg'
modelWeights = 'D:\DETEKSI LUBANG DIJALAN/yolov4_tiny.weights'
net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)

# mengatur backend dan target untuk DNN
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Fungsi untuk mendeteksi objek dalam gambar
def findObject(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    #iterasi melalui semua output layer
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > 0.5:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
                
    # menggunakan non-maximum suppression untuk menghilangkan duplikasi bounding box
    indices = cv2.dnn.NMSBoxes(bbox, confs, 0.5, 0.3)
    for i in indices:
        i = i[0] if isinstance(i, (tuple, list)) else i  # handling for tuple or single value
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        
        # menggambar bouding box pada gambar
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        # menambahkan label pada bounding box
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

#fungsi untuk stream video dari ESP32-CAM
def stream_video():
    while True:
        try:
            # Mendapatkan stream dari ESP32-CAM
            stream = requests.get(ESP32_STREAM_URL, stream=True, timeout=(5, 10))
            if stream.status_code == 200:
                byte_data = bytes()
                for chunk in stream.iter_content(chunk_size=1024):
                    byte_data += chunk
                    a = byte_data.find(b'\xff\xd8')
                    b = byte_data.find(b'\xff\xd9')
                    if a != -1 and b != -1:
                        jpg = byte_data[a:b+2]
                        byte_data = byte_data[b+2:]
                        img_data = np.frombuffer(jpg, dtype=np.uint8)
                        if img_data.size > 0:
                            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                            if img is not None:
                                # Membuka blob dari gambar untuk diberikan ke jaringan YOLO
                                blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), [0, 0, 0], 1, crop=False)
                                net.setInput(blob)
                                layernames = net.getLayerNames()
                                outputNames = [layernames[i - 1] for i in net.getUnconnectedOutLayers()]
                                outputs = net.forward(outputNames)
                                # Mendeteksi objek dalam gambar
                                findObject(outputs, img)
                                ret, buffer = cv2.imencode('.jpg', img)
                                frame = buffer.tobytes()
                                # Mengirimkan frame melalui WebSocket
                                socketio.emit('frame', frame)
                            else:
                                print("Failed to decode image")
                        else:
                            print("Empty image data received")
            else:
                print(f"Error: Failed to fetch stream from ESP32-CAM with status code {stream.status_code}")
        except requests.exceptions.ConnectTimeout as e:
            print(f"ConnectTimeout: {e}")
            time.sleep(1)
        except requests.exceptions.ChunkedEncodingError as e:
            print(f"ChunkedEncodingError: {e}")
            time.sleep(1)
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(1)
# untuk menampilkan halaman utama
@app.route('/dashboard')
def dashboard():
    return render_template('index.html')
# untuk memulai streaming video
@socketio.on('stream')
def start_streaming():
    thread = threading.Thread(target=stream_video)
    thread.start()
# menjalankan aplikasi flask
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
