from flask import Flask, render_template, request
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
import requests
import time  # Import the time module

app = Flask(__name__)

def detect_objects(image):
    """
    Function receives an image,
    passes it through YOLOv8 neural network
    and returns an array of detected objects
    and their bounding boxes
    :param image: Image object
    :return: Array of bounding boxes in format 
    [[x1,y1,x2,y2,object_type,probability],..]
    """
    model = YOLO("best.pt")
    results = model.predict(image)
    
    # Get class names from the model
    names = model.model.names
    
    # Process detections
    output = []
    for result in results:
        boxes = result.boxes.xywh.cpu()
        clss = result.boxes.cls.cpu().tolist()
        confs = result.boxes.conf.float().cpu().tolist()

        for box, cls, conf in zip(boxes, clss, confs):
            class_name = names[int(cls)]  # Access class name using 'names'
            confidence_score = conf
            bounding_box = [int(box[0]), int(box[1]), int(box[0] + box[2]), int(box[1] + box[3])]
            output.append([bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3], class_name, confidence_score])
            
            # Print detection information to the console
            print(f"Class Name: {class_name}, Confidence Score: {confidence_score}, Bounding Box: {bounding_box}")
        
    return output

@app.route('/')
def index():
    # Set the default image URL
    default_image_url = "https://clarksvillenow.sagacom.com/files/2021/12/college-riverside-traffic-1200.jpg"
    return render_template('index.html', default_image_url=default_image_url)

@app.route('/detect', methods=['POST'])
def detect():
    image_url = request.form['image_url']
    response = requests.get(image_url)
    if response.status_code == 200:
        img_data = response.content
        img = Image.open(BytesIO(img_data))
        objects = detect_objects(img)
        return render_template('result.html', image_url=image_url, objects=objects)
    else:
        return "Failed to fetch image from URL"

if __name__ == "__main__":
    app.run(debug=True)
