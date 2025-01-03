import cv2

# Frame dimensions
frameWidth = 1000  # Frame Width
frameHeight = 480  # Frame Height

plateCascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
minArea = 500

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)  # Set width
cap.set(4, frameHeight)  # Set height
cap.set(10, 150)  # Set brightness
count = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from camera. Exiting...")
        break

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect number plates
    numberPlates = plateCascade.detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in numberPlates:
        area = w * h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            imgRoi = img[y:y + h, x:x + w]
            cv2.imshow("Number Plate", imgRoi)

    cv2.imshow("Result", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # Save the scan
        cv2.imwrite(f".\\IMAGES\\{str(count)}.jpg", imgRoi)
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Scan Saved", (15, 265), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
        cv2.imshow("Result", img)
        cv2.waitKey(500)
        count += 1
    elif key == ord('q'):  # Quit the program
        print("Turning off the camera...")
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
