import cv2
import torch
from PIL import Image
from torchvision.transforms import functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/OMEN/Desktop/stage codeclause/best.pt')  # local repo

model.eval()


cap = cv2.VideoCapture(0)  # Use 0 for the default webcam, or specify the index of a different camera

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Perform inference on the frame using your fine-tuned YOLOv5 model
    results = model(frame)

    # Extract the bounding box coordinates and labels of the detected faces
    boxes = results.xyxy[0][:, :4].cpu().numpy().astype(int)
    labels = results.xyxy[0][:, -1].cpu().numpy().astype(int)

    # Iterate over the detected faces
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()