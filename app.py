import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('model.keras')
classes = ['downdog', 'goddess', 'plank', 'tree', 'warrior2']

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # k=cv2.waitKey(1)
            # cv2.imshow('frame', frame)
            # if k%256 == 27:
            #     # ESC pressed
            #     print("Escape hit, closing...")
            #     break
            # if k%256 == 32:
            #     # SPACE pressed
                
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
                    # cv2.putText(frame, f'Pose: {label} Incorrect', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    print('Correct Pose')
                    # cv2.putText(frame, f'Pose: {label} Correct', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
if __name__ == '__main__':
    gen_frames()
