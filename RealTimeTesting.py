import cv2
from ultralytics import YOLO
import datetime


model = YOLO("TheAbsoluteBest.pt")


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(" Error: Could not open webcam.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"yolov8_output_{timestamp}.avi"


fourcc = cv2.VideoWriter_fourcc(*"XVID")  
out = cv2.VideoWriter(output_filename, fourcc, 20.0, (frame_width, frame_height))

print(f"🎥 Recording to {output_filename}")


while True:
    ret, frame = cap.read()
    if not ret:
        break

   
    results = model.predict(source=frame, conf=0.5, verbose=False)

   
    annotated_frame = results[0].plot()

    
    cv2.imshow("YOLOv8 Detection", annotated_frame)

   
    out.write(annotated_frame)

    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
out.release()
cv2.destroyAllWindows()


