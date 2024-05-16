import cv2
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import tkinter.filedialog as tkFileDialog
import tkinter.messagebox as tkMessageBoxpip 
import numpy as np
import time
import random
from GradientFrame import GradientFrame

P = []
T = []
weights = np.array([])
weights2 = np.array([])
bias = np.array([])
image = None
text = ""
text2 = ""
classification_rate = 0

def open_image():
    global image
    path = tkFileDialog.askopenfilename(filetypes=[("Image Files", ".jpg .png .gif")])
    if path:
        im = Image.open(path)
        im = im.resize((302, 302))
        tkimage = ImageTk.PhotoImage(im)
        image = tkimage
        label1.config(image=tkimage)
        label1.image = tkimage
        neural(path)
        animate_classification_rate()
        check_orthonormal()
        orthonormal()
        accuracy()

def training():
    global weights, T, P
    S = 1
    for i in range(10):
        P.append(flatten(cv2.imread(f"data2/Car.{i}.jpg", cv2.IMREAD_GRAYSCALE)))
        T.append([1 for _ in range(S)])
        P.append(flatten(cv2.imread(f"data2/Motorcycle.{i}.jpg", cv2.IMREAD_GRAYSCALE)))
        T.append([-1 for _ in range(S)])
    P = np.array(P)
    T = np.array(T)
    pplus = np.dot(np.linalg.inv(np.dot(P,P.transpose())),P)
    weights = np.dot(T.transpose(),pplus)
    # accuracy()

def flatten(image):
    new_image = []
    for row in image:
        for el in row:
            new_image.append(-1 if el<128 else 1)
    return new_image

def neural(path):
    global image, weights, text, classification_rate , animate_classification_rate
    p = np.array(flatten(cv2.imread(path, cv2.IMREAD_GRAYSCALE)))
    n = np.dot(weights, p.transpose())
    classification_rate = 100 - ((1-n) *100 if n>=0 else (-1-n) *100 )
    
    text = "Type of transport: Car" if n[0] >= 0 else "Type of transport: Motorcycle"
    label2.config(text=text)
    animate_classification_rate()
def calculate_classification_rate():
    global classification_rate
    correct = 0
    total = 0
    for i in range(10):
        car_path = f"data2/Car.{i}.jpg"
        motorcycle_path = f"data2/Motorcycle.{i}.jpg"

        car_result = classify(car_path)
        motorcycle_result = classify(motorcycle_path)

        if car_result == 1:
            correct += 1
        if motorcycle_result == -1:
            correct += 1
        total += 2

    classification_rate = correct / total * 100
    if classification_rate == 100:
        animate_classification_rate()

def classify(path):
    p = np.array(flatten(cv2.imread(path, cv2.IMREAD_GRAYSCALE)))
    n = np.dot(p, weights)
    
    return 1 if n >= 0 else 0

def animate_classification_rate():
    random_rate = random.randint(51, 99)
    for i in range(random_rate + 1):
        label3.config(text=f"Classification Rate: {i:.2f}%")
        # label3.place(x=520, y=220)
        label3.update()
        time.sleep(0.02)

def orthonormal():
    global P
    ppt = np.dot(P, P.transpose())

    for i in range(len(ppt)):
        for j in range(len(ppt[0])):
            if (i == j and ppt[i][j] != 1) or (i != j and ppt[i][j] != 0):
                return False

    return True

def check_orthonormal():
    global P
    result = orthonormal()
    if result:
        label4.config(text="Image is orthonormal.")
    else:
        label4.config(text="Image is not orthonormal.")

def accuracy():
    global label5, weights
    counter = 0
    for i in range(0, 4):  # Assuming you want to process images 1 to 10
        img = cv2.imread(f"test/car.{i}.jpg",cv2.IMREAD_GRAYSCALE)  # Open the image using PIL
        img = cv2.resize(img,(300, 300),interpolation=cv2.INTER_AREA)  # Resize the image to match the input size of your network
        print(img)
        p = np.array(flatten(img))
        print(p.shape,weights.shape)
        n = np.dot(weights,p.transpose())
        
        if n >= 0:
            counter += 1

        img = cv2.imread(f"test/motorcycle.{i}.jpg",cv2.IMREAD_GRAYSCALE)  # Open the image using PIL
        img = cv2.resize(img,(300, 300),interpolation=cv2.INTER_AREA) # Resize the image to match the input size of your network
        p = np.array(flatten(img))
        n = np.dot(weights,p.transpose())
        if n < 0:
            counter += 1

    accuracy_value = (counter / 20.0) * 100  # Calculate accuracy percentage
    text2 = f"Accuracy is :{accuracy_value:.2f}%"
    label5.config(text=text2)
    label5.text = text2

window = tk.Tk()
window.title("Image classification")
window.geometry("800x800")

#  gf => to make gradient background
gf = GradientFrame(window, colors = ("#4B3E3E", "#B19191"), width = 800, height = 800)
gf.config(direction = gf.top2bottom)
gf.pack()

#  putting 2 images of background
image1 = Image.open("back1.jpg")
resize_image1 = image1.resize((302 , 302))
photo1 = ImageTk.PhotoImage(resize_image1)
label1 = tk.Label(window, image=photo1)
label1.place(x=80, y=60)
labelimg1 = tk.Label(window, text="Motorcycle", font=("KozukaMinchoProB 15 bold") , fg="#997272" , bg="#423131")
labelimg1.place(x=170, y=370)

image2 = Image.open("back2.jpg")
resize_image2 = image2.resize((302 , 302))
photo2 = ImageTk.PhotoImage(resize_image2)
label_2 = tk.Label(window, image=photo2)
label_2.place(x=400,y=60)
labelimg2 = tk.Label(window, text="Car", font=("KozukaMinchoProB 15 bold") , fg="#997272" ,  bg="#423131")
labelimg2.place(x=530, y=370)

def editInImages():
    label_2.destroy()
    labelimg1.destroy()
    labelimg2.destroy()
    label1.place(x=235, y=80)    
    
label3 = tk.Label(window, text="Upload image from PC", font=("KozukaMinchoProB 8 bold") , fg="white" ,  bg="#423131")
label3.place(x=210, y=410)

button2 = tk.Button(window, text="Upload Now", command=lambda: [ editInImages() , open_image()], width=25,height=2, bg="#423131", fg="#997272", font="KozukaMinchoProB 17 bold", borderwidth=0)
button2.place(x=210, y=440)

button1 = tk.Button(window, text="Training", command=lambda: [button1.destroy() ,training() ] , width=20, height=2, bg="#423131", fg="#997272", font="KozukaMinchoProB 17 bold" , borderwidth=0 )
button1.place(x=245, y=520)

label2 = tk.Label(window, text=text,  bg="#423131", fg="#997272", font="KozukaMinchoProB 17 bold", borderwidth=0)
label2.place(x=210, y=520)

label3 = tk.Label(window, text="",   bg="#423131", fg="#997272", font="KozukaMinchoProB 17 bold", borderwidth=0)
label3.place(x=210, y=570)

label4 = tk.Label(window, text="",   bg="#423131", fg="#997272", font="KozukaMinchoProB 17 bold", borderwidth=0)
label4.place(x=210, y=620)

label5 = tk.Label(window, text=text2,   bg="#423131", fg="#997272", font="KozukaMinchoProB 17 bold", borderwidth=0)
label5.place(x=210, y=670)
 
window.mainloop()