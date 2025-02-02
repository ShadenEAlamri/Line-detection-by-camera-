# Line-detection-by-camera-

## Requirements:

1. **Python 3.x**: Make sure you have Python installed on your device.
2. **OpenCV**: A library for handling images and video.
3. **NumPy**: A library for handling arrays.

## Step 1: Install Libraries

Before you start, ensure that you have installed the required libraries. Open the terminal or command prompt and run the following command:

```bash
pip install opencv-python numpy
```

## Step 2: Create the Code

Create a new Python file (e.g., `line_detection.py`) and use the following code:

```python
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Unable to open the camera!")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Unable to read a frame from the camera!")
        break

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian filter to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Apply Hough Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    # Draw the detected lines on the original image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw the line in green

    # Display the modified frame with lines
    cv2.imshow('Webcam View', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Step 3: Code Explanation

1. **Opening the Camera**:
   - At the beginning of the code, we open the camera using `cv2.VideoCapture(0)`. This method captures video from the default camera on your device.
   - If the camera fails to open, a message "Unable to open the camera!" will be printed.

2. **Converting to Grayscale**:
   - We use `cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)` to convert the colored image to a grayscale image. This simplifies edge detection, as color does not affect this process.

3. **Applying GaussianBlur to Reduce Noise**:
   - We use `cv2.GaussianBlur()` to apply a Gaussian filter to the grayscale image to reduce noise.
   - This step is crucial for minimizing factors that could affect edge detection accuracy.

4. **Edge Detection Using Canny**:
   - We use `cv2.Canny()` to detect edges in the image. This algorithm is one of the most commonly used for edge detection.

5. **Detecting Lines Using HoughLinesP**:
   - We use `cv2.HoughLinesP()` to convert Canny edges into actual lines within the image.
   - This algorithm detects lines using the Hough Transform. By adjusting values like `threshold`, `minLineLength`, and `maxLineGap`, you can tailor the code to your needs.

6. **Drawing Lines on the Original Image**:
   - We use `cv2.line()` to draw the detected lines on the original image.

7. **Displaying the Modified Video**:
   - The video with the detected lines is displayed via `cv2.imshow()`. The video updates continuously until you press the 'q' key to exit.

## Step 4: Running the Code

- After creating the Python file, open the terminal or command prompt in the folder containing the code.
- Run the program using the following command:

```bash
python line_detection.py
```

- A video window will appear displaying the camera feed with detected lines.

## Application Control:

- To close the camera and exit the application, press the 'q' key.

## Step 5: Notes:

1. **OpenCV Library**: Ensure you have installed the OpenCV library correctly.
2. **Values Used in Canny and HoughLinesP**:
   - `Canny(blurred, 50, 150)`: These values set the minimum and maximum thresholds for edge detection.
   - `HoughLinesP()`: Parameters like `threshold`, `minLineLength`, and `maxLineGap` can be adjusted to improve detection based on your requirements.
3. **Requirements**: Ensure that the camera is connected and functioning properly.

## Conclusion

You have now created a program to detect lines using Python and OpenCV. This code can be improved or customized based on your needs. If you wish to add additional features such as color detection or user interaction, you can modify the code easily based on these ideas.
