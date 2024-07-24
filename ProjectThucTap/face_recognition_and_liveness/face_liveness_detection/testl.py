from imutils.video import VideoStream
import tensorflow as tf
import numpy as np
import argparse
import imutils
import time
import cv2
import os

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True,
                    help='Path to trained model')
parser.add_argument('-d', '--detector', type=str, required=True,
                    help='Path to OpenCV\'s deep learning face detector')
parser.add_argument('-c', '--confidence', type=float, default=0.5,
                    help='minimum probability to filter out weak detections')
args = vars(parser.parse_args())

print('[INFO] loading face detector...')
proto_path = os.path.sep.join([args['detector'], 'deploy.prototxt'])
model_path = os.path.sep.join([args['detector'], 'res10_300x300_ssd_iter_140000.caffemodel'])
detector_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

liveness_model = tf.keras.models.load_model(args['model'])

print('[INFO] starting video stream...')
vs = VideoStream(src=0).start()
time.sleep(2)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    detector_net.setInput(blob)
    detections = detector_net.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > args['confidence']:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)
            face = frame[startY:endY, startX:endX]
            try:
                face = cv2.resize(face, (32, 32))
            except:
                continue
            face = face.astype('float') / 255.0 
            face = tf.keras.preprocessing.image.img_to_array(face)
            face = np.expand_dims(face, axis=0)
            preds = liveness_model.predict(face)[0]
            label = 'real' if np.argmax(preds) == 1 else 'fake'
            label = f'{label}: {preds[np.argmax(preds)]:.4f}'
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
# cleanup
cv2.destroyAllWindows()
vs.stop()
