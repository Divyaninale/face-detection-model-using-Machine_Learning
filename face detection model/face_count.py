import cv2

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the image
image = cv2.imread('img3.jpeg')

# Convert the image to grayscale (face detection works on grayscale images)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw bounding boxes around the detected faces and count them
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Count the number of faces
num_faces = len(faces)
print("Number of faces detected:", num_faces)

# Display the image with bounding boxes
cv2.imshow('Faces', image)
cv2.waitKey(3000)
cv2.destroyAllWindows()
