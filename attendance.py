import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

label_map = np.load(
    "labels.npy",
    allow_pickle=True
).item()

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

attendance_file = "attendance.csv"

if not os.path.exists(attendance_file):
    pd.DataFrame(
        columns=["Name", "Date", "Time"]
    ).to_csv(attendance_file, index=False)

marked = set()

cam = cv2.VideoCapture(0)

while True:

    ret, frame = cam.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        label, confidence = recognizer.predict(
            gray[y:y+h, x:x+w]
        )

        if confidence < 70:

            name = label_map[label]

            cv2.putText(
                frame,
                f"{name}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            if name not in marked:

                now = datetime.now()

                df = pd.read_csv(
                    attendance_file
                )

                df.loc[len(df)] = [
                    name,
                    now.strftime("%Y-%m-%d"),
                    now.strftime("%H:%M:%S")
                ]

                df.to_csv(
                    attendance_file,
                    index=False
                )

                marked.add(name)

                print(
                    f"Attendance marked for {name}"
                )

        else:

            cv2.putText(
                frame,
                "Unknown",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (255, 0, 0),
            2
        )

    cv2.imshow(
        "Biometric Attendance System",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
