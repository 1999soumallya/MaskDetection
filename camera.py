import cv2
import face_recognition
from tensorflow.keras.models import load_model
from tensorflow import expand_dims

# Create a video frame
cap = cv2.VideoCapture(0)
cap.set(10, 100)
model = load_model(r'models/face_detection.h5')

while True:
    success, img = cap.read()
    try:
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25, interpolation=cv2.INTER_NEAREST)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        facesCurrentFrame = face_recognition.face_locations(imgS)
        encodesCurrentFrame = face_recognition.face_encodings(imgS, facesCurrentFrame)
    except:
        break
    # expanding dimension for prediction
    img_arr = expand_dims(imgS, 0)
    # make prediction
    predict_val = model.predict(img_arr)
    print(predict_val)
    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        if predict_val[0][0] < 0.5:
            color = (83, 255, 84)
            text = 'Mask'
            store = 0.5 + predict_val[0][0]
        else:
            color = (255, 84, 83)
            text = 'No Mask'
            store = predict_val[0][0]
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        text = "{}: {:.2f}%".format(text, (store * 100))
        cv2.putText(img, text, org=(x1 + 6, y2 - 6), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=color,
                    thickness=2)
    resize_image = cv2.resize(img, (1000, 700))
    cv2.imshow('Webcam', resize_image)
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
