from tkinter import *
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array
from tkinter import messagebox
import os
from twilio.rest import Client

def testDevice(source):
    cap = cv2.VideoCapture(source)
    if cap is None or not cap.isOpened():
        return('Warning: unable to open video source: ', source)
    else: 
        return('Video source is working properly')
        



client = Client(account_sid, auth_token)

#Loading the saved model
model = tf.keras.models.load_model('firemodel.h5')

i = 0

root = Tk()
# root = Toplevel()

root.title('Fire Detection System')
root.geometry("400x500")
root.resizable(width=False, height=False)
root.iconbitmap('fire.ico')

def alert():
    message = client.messages \
                .create(
                     body="URGENT! Fire has been DETECTED! Please respond IMMEDIATELY",
                     from_='+18304102998',
                     to='+17745038589'
                 )
    print(message.sid)
    

def system():
    global i
    video = cv2.VideoCapture(0)
    while True:
            _, frame = video.read()
            #Convert the captured frame into RGB
            im = Image.fromarray(frame, 'RGB')
            
            #Resizing into 224x224 because we trained the model with this image size.
            im = im.resize((224,224))
            
            # img_array = image.img_to_array(im)

            img_array = img_to_array(im)
            img_array = np.expand_dims(img_array, axis=0) / 255
            
            probabilities = model.predict(img_array)[0]
            #Calling the predict method on model to predict 'fire' on the image
            prediction = np.argmax(probabilities)
            #if prediction is 0, which means there is fire in the frame.
            if prediction == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                print(probabilities[prediction])
                if i == 0:                    
                    if probabilities[prediction]>=0.8:
                        alert()
                        messagebox.showwarning("Warning", "Fire has been detected! SMS Sent!")
                        print ('Alert has been sent!')
                        i = 1                            
            
            cv2.imshow("Live Footage", frame)
            key=cv2.waitKey(1)
            if key == ord('q'):
                    break
    video.release()
    cv2.destroyAllWindows()


myLabel = Label (root, text="Fire Detection System", font=("Georgia", 20, "bold")).pack(side=TOP, pady=6)
bg= PhotoImage(file='fire-safety.gif')


canvas1= Canvas(root,width=400,height=360)
canvas1.pack(fill="both", expand=True)

canvas1.create_image(0,0,image=bg,anchor="nw")

myCamera = Label (root, text=testDevice(0), font=("Georgia", 10, "underline")).pack(side=TOP)

photoimage= PhotoImage(file=r"quit.gif")
photoimage2= PhotoImage(file=r"fire.gif")

myButton = Button (root, text="RUN SYSTEM!", font="Dosis",  image=photoimage2,
                   compound=LEFT, command=system)
myButton.pack(side=LEFT, padx=8, pady=5)

myButton2 = Button (root, text="QUIT SYSTEM!", font="Dosis", image=photoimage,
                    compound=LEFT, command=root.destroy)
myButton2.pack(side=RIGHT, padx=8, pady=5)

root.mainloop()
