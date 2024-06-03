import torch
import cv2

model = torch.hub.load('ultralytics/yolov5', 'best', pretrained=True)

# Define class names (replace with your actual class names)
class_names = ['person', 'car', 'dog', 'cat', 'bird']  # Example class names

# Corrected absolute path to the image file
frame = cv2.imread('/home/baymax/yolobot/src/yolobot_recognition/scripts/zidane.jpg')
detections = model(frame[..., ::-1])
results = detections.pandas().xyxy[0].to_dict(orient="records")
# Inside the loop where you process detections
for result in results:
    con = result['confidence']
    class_index = int(result['class'])
    if class_index >= len(class_names):
        print(f"Invalid class index: {class_index}")
    else:
        class_name = class_names[class_index]  # Get class name from class index
        x1 = int(result['xmin'])
        y1 = int(result['ymin'])
        x2 = int(result['xmax'])
        y2 = int(result['ymax'])
        # Define COLORS (for example, using green color)
        COLORS = (0, 255, 0)  # Green color
        # Draw rectangle on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS, 2)
        # Display object name and coordinates
        text = f'{class_name}: ({x1}, {y1}) - ({x2}, {y2})'
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS, 2)

# Display the resulting image
cv2.imshow('Result', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
