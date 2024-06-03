import torch
import cv2
from models.experimental import attempt_load
from utils.general import non_max_suppression

# Load your custom YOLOv5 model weights
weights_path = '/home/baymax/yolobot/src/yolobot_recognition/scripts/best.pt'  # Path to your custom weights file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(weights=weights_path, device=device)

# Set the model to evaluation mode
model.eval()

# Capture video from webcam
cap = cv2.VideoCapture(0)  # 0 represents the first webcam connected, you can change it if you have multiple webcams

class Annotator:
    def __init__(self, line_width=3, example=""):
        self.line_width = line_width
        self.example = example

    def box_label(self, xyxy, label=None, color=(255, 255, 255)):
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(self.example, (x1, y1), (x2, y2), color, self.line_width)
        if label:
            tf = max(self.line_width - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=self.line_width / 3, thickness=tf)[0]
            cv2.rectangle(self.example, (x1, y1), (x1 + t_size[0], y1 - t_size[1] - 3), color, -1)  # filled
            cv2.putText(self.example, label, (x1, y1 - 2), 0, self.line_width / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

# Initialize Annotator
line_thickness = 3
annotator = Annotator(line_width=line_thickness, example="")

while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        break  # Break the loop if frame is not read successfully

    # Convert the frame to a PyTorch tensor
    frame_tensor = torch.from_numpy(frame).to(device)

    # Reshape and normalize the frame
    frame_tensor = frame_tensor.permute(2, 0, 1).float().div(255.0).unsqueeze(0)

    # Perform object detection on the frame
    results = model(frame_tensor)

    # Apply non-maximum suppression to remove redundant detections
    results = non_max_suppression(results, 0.4, 0.5)
   
    # Process the inference results
    for detection in results:
        for *xyxy, conf, cls in detection:
            x1, y1, x2, y2 = map(int, xyxy)
            c = int(cls)  # integer class
            
            # Calculate center coordinates of the bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Define COLORS (for example, using green color)
            COLORS = (0, 255, 0)  # Green color
            COLORS2 = (0, 0, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS2, 2)
            
            # Display object name and coordinates at the center of the bounding box
            class_name = model.names[c] if c < len(model.names) else 'unknown'
            text = f'Class: {class_name}, Coordinates: ({center_x}, {center_y})'
            cv2.putText(frame, text, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS, 2)



    # Display the resulting frame
    cv2.imshow('Webcam', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
