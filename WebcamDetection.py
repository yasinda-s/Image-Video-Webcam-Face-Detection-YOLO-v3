import cv2
import numpy as np

#yolov3.weights contain the trained model weights for the 80 classes we can now used to predict

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg') #weights and configs
classes = []

with open('coco.names', 'r') as f: #open the coco names file and then read the labels
    classes = f.read().splitlines() #Python splitlines() method splits the string based on the lines

cap = cv2.VideoCapture(0) #capture first frame of video

while True:
    _, img = cap.read()
    height, width, _ = img.shape

    #change image into the preferences of yolo
    #dimensions - 416x416
    #pixels divided by 255
    #BGR -> RGB
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0,0), swapRB=True, crop=False)

    net.setInput(blob) #feed blob file into the model (deep neural network)
    output_layers_names = net.getUnconnectedOutLayersNames() #to get the names of the output classes of the model
    layerOutputs = net.forward(output_layers_names) #this gives the confidence, index of class label and the coordinates of objects

    confidences = []
    class_ids = []
    boxes = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255, 255, 255), 2)


    cv2.imshow("Image", img)
    key = cv2.waitKey(1) #shows and image repeatedly until any key is pressed to close or go to next frame
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()