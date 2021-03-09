from tkinter import *
from PIL import ImageTk , Image
import pickle
import numpy as np
T = Tk()
def close():
    T.destroy()

def predict():
    model = pickle.load(open('breast-cancer.pickle','rb'))
    val = [a.get() ,b.get() ,c.get() ,d.get() ,e.get() ,f.get() ,g.get() ,h.get() ,i.get() ,j.get() ,k.get() ,l.get() ,m.get() ,n.get() ,o.get() ,p.get() ,q.get() ,r.get() ,s.get() ,t.get() ,u.get() ,v.get() ,w.get() ,x.get() ,y.get() ,z.get() ,aa.get() ,ab.get() ,ac.get() ,ad.get() ]
    val = np.array([val])
    result = model.predict(val)
    if (result[0] == 0):
        l2 = Label(text="Patient has cancer (Malignant Tumor)" )
        l2.place(x=300 , y = 510)
    else:
        l2 = Label(text="Patient has no cancer (Malignant banign)" )
        l2.place(x=300 , y = 510)
        
        
    
    

def clear(event,e1):
    e1.delete(0,END)
    
T.title("Breast-Cancer_predictor")
bg = ImageTk.PhotoImage(file = "hospital.png")
T.geometry("800x600+50+50")
T.resizable(width=False , height = False)
a = DoubleVar()
b = DoubleVar()
c = DoubleVar()
d = DoubleVar()
e = DoubleVar()
f = DoubleVar()
g = DoubleVar()
h = DoubleVar()
i = DoubleVar()
j = DoubleVar()
k = DoubleVar()
l = DoubleVar()
m = DoubleVar()
n = DoubleVar()
o = DoubleVar()
p = DoubleVar()
q = DoubleVar()
r = DoubleVar()
s = DoubleVar()
t = DoubleVar()
u = DoubleVar()
v = DoubleVar()
w = DoubleVar()
x = DoubleVar()
y = DoubleVar()
z = DoubleVar()
aa= DoubleVar()
ab =DoubleVar()
ac = DoubleVar()
ad = DoubleVar()
label=Label(image = bg)
label.place(x=0,y=0,relwidth=1 , relheight=1)

l1 = Label(text = "Jupitor Breast Cancer Detector" , font=("Georgia",35 ), fg="red")
l1.place(x=100,y=40)

e1 = Entry(bd=3,bg="linen",textvariable = a)
e1.insert(0, "mean radius ")
e1.bind("<Button-1>" , lambda event:clear(event , e1))
e1.place(x=30,y=200)

e2 = Entry(bg="linen",bd=3,textvariable =b)
e2.insert(0, "mean texture ")
e2.bind("<Button-1>" , lambda event:clear(event , e2))
e2.place(x=180,y=200)


e3 = Entry(bd=3,bg="linen",textvariable =c)
e3.insert(0, "mean perimeter ")
e3.bind("<Button-1>" , lambda event:clear(event , e3))
e3.place(x=330,y=200)



e4 = Entry(bd=3,bg="linen",textvariable =d)
e4.insert(0, "mean area ")
e4.bind("<Button-1>" , lambda event:clear(event , e4))
e4.place(x=480,y=200)



e5 = Entry(bd=3,bg="linen",textvariable =e)
e5.insert(0, "mean smoothness ")
e5.bind("<Button-1>" , lambda event:clear(event , e5))
e5.place(x=630,y=200)


e6 = Entry(bd=3,bg="linen",textvariable =f)
e6.insert(0, "mean compactness ")
e6.bind("<Button-1>" , lambda event:clear(event , e6))
e6.place(x=30,y=230)



e7 = Entry(bd=3,bg="linen",textvariable =g)
e7.insert(0, "mean concavity ")
e7.bind("<Button-1>" , lambda event:clear(event , e7))
e7.place(x=180,y=230)



e8 = Entry(bd=3,bg="linen",textvariable =h)
e8.insert(0, "mean concave points ")
e8.bind("<Button-1>" , lambda event:clear(event , e8))
e8.place(x=330,y=230)



e9 = Entry(bd=3,bg="linen",textvariable =i)
e9.insert(0, "mean symmetry ")
e9.bind("<Button-1>" , lambda event:clear(event , e9))
e9.place(x=480,y=230)



e10 = Entry(bd=3,bg="linen",textvariable =j)
e10.insert(0, "mean fractal dimension ")
e10.bind("<Button-1>" , lambda event:clear(event , e10))
e10.place(x=630,y=230)


e11 = Entry(bd=3,bg="linen",textvariable =k)
e11.insert(0, "radius error ")
e11.bind("<Button-1>" , lambda event:clear(event , e11))
e11.place(x=30,y=260)

e12 = Entry(bg="linen",bd=3,textvariable =l)
e12.insert(0, "texture error ")
e12.bind("<Button-1>" , lambda event:clear(event , e12))
e12.place(x=180,y=260)


e13 = Entry(bd=3,bg="linen",textvariable =m)
e13.insert(0, "perimeter error ")
e13.bind("<Button-1>" , lambda event:clear(event , e13))
e13.place(x=330,y=260)



e14 = Entry(bd=3,bg="linen",textvariable =n)
e14.insert(0, "area error ")
e14.bind("<Button-1>" , lambda event:clear(event , e14))
e14.place(x=480,y=260)



e15 = Entry(bd=3,bg="linen",textvariable =o)
e15.insert(0, "smoothness error ")
e15.bind("<Button-1>" , lambda event:clear(event , e15))
e15.place(x=630,y=260)


e16 = Entry(bd=3,bg="linen",textvariable =p)
e16.insert(0, "compactness error ")
e16.bind("<Button-1>" , lambda event:clear(event , e16))
e16.place(x=30,y=290)



e17 = Entry(bd=3,bg="linen",textvariable =q)
e17.insert(0, "concavity error ")
e17.bind("<Button-1>" , lambda event:clear(event , e17))
e17.place(x=180,y=290)



e18 = Entry(bd=3,bg="linen",textvariable =r)
e18.insert(0, "concave points error ")
e18.bind("<Button-1>" , lambda event:clear(event , e18))
e18.place(x=330,y=290)



e19 = Entry(bd=3,bg="linen",textvariable =s)
e19.insert(0, "symmetry error ")
e19.bind("<Button-1>" , lambda event:clear(event , e19))
e19.place(x=480,y=290)



e20 = Entry(bd=3,bg="linen",textvariable =t)
e20.insert(0, "fractal dimension error ")
e20.bind("<Button-1>" , lambda event:clear(event , e20))
e20.place(x=630,y=290)


e21 = Entry(bd=3,bg="linen",textvariable =u)
e21.insert(0, "wrost redius ")
e21.bind("<Button-1>" , lambda event:clear(event , e21))
e21.place(x=30,y=320)

e22 = Entry(bg="linen",bd=3,textvariable =v)
e22.insert(0, "wrost texture ")
e22.bind("<Button-1>" , lambda event:clear(event , e22))
e22.place(x=180,y=320)


e23 = Entry(bd=3,bg="linen",textvariable =w)
e23.insert(0, "wrost perimeter ")
e23.bind("<Button-1>" , lambda event:clear(event , e23))
e23.place(x=330,y=320)



e24 = Entry(bd=3,bg="linen",textvariable =x)
e24.insert(0, "wrost area ")
e24.bind("<Button-1>" , lambda event:clear(event , e24))
e24.place(x=480,y=320)



e25 = Entry(bd=3,bg="linen",textvariable =y)
e25.insert(0, "wrost smoothness ")
e25.bind("<Button-1>" , lambda event:clear(event , e25))
e25.place(x=630,y=320)


e26 = Entry(bd=3,bg="linen",textvariable =z)
e26.insert(0, "wrost compactness ")
e26.bind("<Button-1>" , lambda event:clear(event , e26))
e26.place(x=30,y=350)



e27 = Entry(bd=3,bg="linen",textvariable =aa)
e27.insert(0, "worst concavity ")
e27.bind("<Button-1>" , lambda event:clear(event , e27))
e27.place(x=180,y=350)



e28 = Entry(bd=3,bg="linen",textvariable =ab)
e28.insert(0, "wrost concave points ")
e28.bind("<Button-1>" , lambda event:clear(event , e28))
e28.place(x=330,y=350)



e29 = Entry(bd=3,bg="linen",textvariable =ac)
e29.insert(0, "wrost symmetry ")
e29.bind("<Button-1>" , lambda event:clear(event , e29))
e29.place(x=480,y=350)



e30 = Entry(bd=3,bg="linen",textvariable =ad)
e30.insert(0, "wrost fractal dimension ")
e30.bind("<Button-1>" , lambda event:clear(event , e30))
e30.place(x=630,y=350)


b1 = Button(text = "  Predict  " ,bd= 4 , bg="black" , fg="white",command = predict)
b1.place(x=390, y = 450)
b2 = Button(text = "  Exit  " ,bd= 4 , bg="black" , fg="white",command = close)
b2.place(x=330, y = 450)





T.mainloop()

