from flask import Flask, render_template, Response
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = load_model('yoga_model.keras')
classes = ['downdog', 'goddess', 'plank', 'tree', 'warrior2']

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            resized_frame = cv2.resize(frame, (300, 300))
            img_array = image.img_to_array(resized_frame)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            
            prediction = model.predict(img_array)
            label = classes[np.argmax(prediction)]
            print(label, np.max(prediction))
            
            if np.max(prediction)<0.7:
                print('Incorrect Pose')
                cv2.putText(frame, f'Pose: {label} Incorrect', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                print('Correct Pose')
                cv2.putText(frame, f'Pose: {label} Correct', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    print('video_feed')
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False)
