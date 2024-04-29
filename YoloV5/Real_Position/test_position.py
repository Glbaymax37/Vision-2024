import torch
import cv2

model = torch.hub.load('ultralytics/yolov5', 'yolov5l6', pretrained=True)

# Capture video from webcam
cap = cv2.VideoCapture(0)  # 0 represents the first webcam connected, you can change it if you have multiple webcams

while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        break  # Break the loop if frame is not read successfully

    detections = model(frame[..., ::-1])  # Perform object detection on the frame
    results = detections.pandas().xyxy[0].to_dict(orient="records")
    for result in results:
        con = result['confidence']
        cs = result['class']
        x1 = int(result['xmin'])
        y1 = int(result['ymin'])
        x2 = int(result['xmax'])
        y2 = int(result['ymax'])
        # Define COLORS (for example, using green color)
        COLORS = (0, 255, 0)  # Green color
        # Draw rectangle on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS, 2)
        # Display object name and coordinates
        text = f'{cs}: ({x1}, {y1}) - ({x2}, {y2})'
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS, 2)

    # Display the resulting frame
    cv2.imshow('Webcam', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
