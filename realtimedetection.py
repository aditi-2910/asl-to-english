import tensorflow as tf
from tensorflow.keras.models import model_from_json
import cv2
import numpy as np

# Load the model from JSON
try:
    with open("signlanguagedetectionmodel48x48.json", "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights("signlanguagedetectionmodel48x48.h5")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Function to preprocess the image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Labels for prediction
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'blank']

# Start capturing video
cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
    crop_frame = frame[40:300, 0:300]
    crop_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
    crop_frame = cv2.resize(crop_frame, (48, 48))
    crop_frame = extract_features(crop_frame)
    
    # Make prediction
    try:
        pred = model.predict(crop_frame)
        prediction_label = labels[pred.argmax()]
        accuracy = np.max(pred) * 100
        cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
        if prediction_label == 'blank':
            cv2.putText(frame, " ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            accu = "{:.2f}%".format(accuracy)
            cv2.putText(frame, f'{prediction_label}  {accu}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    except Exception as e:
        print(f"Prediction error: {e}")
    
    cv2.imshow("output", frame)
    if cv2.waitKey(27) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
