import numpy as np
import cv2
import os
from keras.models import load_model
from flask import Flask, render_template, Response
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from gtts import gTTS #to convert text to speech
from playsound import playsound
global graph
global writer
from skimage.transform import resize
graph =tf.compat.v1.get_default_graph()

writer = None
model = load_model('conversation deaf and dumb.h5')
vals = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
app = Flask (__name__)   
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(0) #triggers the local camera
pred=""
@app.route('/')  
def index(): 
    return render_template('index.html')
#preprocessing the frame captured from camera
def detect(frame):
    img = resize(frame,(64,64,1)) 
    img = np.expand_dims(img,axis=0)
    if(np.max(img)>1):
        img=img/255.0
    #with graph.as_default():
    prediction = np.argmax(model.predict(img),axis=0)
    print(prediction)
    pred=vals[prediction[0]]
    print (pred)
    myobj=gTTS(text=pred,lang='en',slow=False)
    myobj.save("pred.mp3")
    playsound("pred.mp3")
    return pred

def gen_frames():  
    while True:
        success, frame = vs.read() 
        cv2.imwrite("index.png",frame)
        
        #if not success:
        #   break
        #else:
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        img_grey = cv2.imread("index.png",cv2.IMREAD_GRAYSCALE)[0]
        thresh=128
        img_binary=cv2.threshold(img_grey,thresh,255,cv2.THRESH_BINARY)[1]
        img = resize(img_binary,(64,64,1))
        detect(img)    
       
@app.route('/video_feed')
def video_feed(): 
    return Response(gen_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)




