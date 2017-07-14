#!usr/bin/env python
#coding:utf-8
#The perceptron algorithm is a model that used to classify data.
#There are two forms: the original form and the dual form

#original form (The following code is the original form)
# Imports
import numpy as np
from numpy import array
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from random import choice

# matplotlib.rcParams['toolbar'] = 'None'
fig = plt.figure(figsize=(6,6), facecolor='white')
ax = fig.add_subplot(1,1,1)

#data set
X = array([[1,2,-1],[2,3,-1],[3,5,-1],[4,7,-1],[5,8,-1],[3,6,-1],[2,1,1],[3,1,1],[6,3,1],[7,2,1],[5,3,1],[6,3,1]])
plt.xlim(0,X[:,0].max()+1)
plt.ylim(0,X[:,1].max()+1)
plt.xticks(range(0,X[:,0].max()+1))
plt.yticks(range(0,X[:,1].max()+1))

for p in X:
	if(p[2]>0):
		plt.scatter([p[0],],[p[1],], 15, color ='red', marker = 'o')
	else:
		plt.scatter([p[0],],[p[1],], 15, color ='blue', marker = 'x')
p = X[random.randint(0,len(X)-1)]
w1 = round(10*random.random(),2)
w2 = round(10*random.random(),2)
w = [w1,w2]
b = -p[0]*w1-p[1]*w2

#draw an initial line
if(w1!=0 and w2!=0):
	x1 = range(0,X[:,0].max()+2)
	x2 = []
	for i in x1:
		x2.append((-b-w1*i)/w2)
elif(w1!=0 and w2==0):
	x2 = range(0,X[:,1].max()+2)
	x1 = []
	for i in x2:
		x1.append(-b/w1)
elif(w1==0 and w2!=0):
	x1 = range(0,X[:,0].max()+2)
	x2 = []
	for i in x1:
		x2.append(-b/w2)
else:
	x2 = range(0,X[:,1].max()+2)
	x1 = []
	for i in x2:
		x1.append(p[0])
lines = ax.plot(x1,x2,color="black", linewidth=2.5, linestyle="-")	

#check the model's performance
def isPerfect():
	global w1,w2,b,X
	n = 0.3
	flag = True
	ErrorPoint = []
	for index,i in enumerate(X):
		if((w1*i[0]+w2*i[1]+b)*i[2])<=5:
			ErrorPoint.append(index)
	if(len(ErrorPoint)==0):
		return flag
	else:
		flag = False
		p = X[choice(ErrorPoint)]
		w1 = w1 + n*p[2]*p[0]
		w2 = w2 + n*p[2]*p[1]
		b = b + n*p[2]
	return flag

#update the figure
def update(frame):
	global w1,w2,b,X

	flag = isPerfect()
	if(flag==True):
		print "Perfect!"
		return
	else:
		print w1,w2,b	
		if(w1!=0 and w2!=0):
			x1 = range(0,X[:,0].max()+2)
			x2 = []
			for i in x1:
				x2.append((-b-w1*i)/w2)
		elif(w1!=0 and w2==0):
			x2 = range(0,X[:,1].max()+2)
			x1 = []
			for i in x2:
				x1.append(-b/w1)
		elif(w1==0 and w2!=0):
			x1 = range(0,X[:,0].max()+2)
			x2 = []
			for i in x1:
				x2.append(-b/w2)
		else:
			x2 = range(0,X[:,1].max()+2)
			x1 = []
			for i in x2:
				x1.append(p[0])
		lines = ax.plot(x1,x2,color="black", linewidth=2.5, linestyle="-")
		ax.lines.pop(0)
		return plt,

animation = FuncAnimation(fig, update,blit=False, interval=15)

# animation.save('rain.mp4', writer='ffmpeg')
plt.show()



#dual form (the following code is dual form)
#!usr/bin/env python
#coding:utf-8
# Imports
import numpy as np
from numpy import array
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from random import choice

matplotlib.rcParams['toolbar'] = 'None'
fig = plt.figure(figsize=(6,6), facecolor='white')
ax = fig.add_subplot(1,1,1)

X = array([[1,2,-1],[2,3,-1],[3,5,-1],[4,7,-1],[5,8,-1],[3,6,-1],[2,1,1],[3,1,1],[6,3,1],[7,2,1],[5,3,1],[6,3,1]])
plt.xlim(0,X[:,0].max()+1)
plt.ylim(0,X[:,1].max()+1)
plt.xticks(range(0,X[:,0].max()+1))
plt.yticks(range(0,X[:,1].max()+1))

for p in X:
	if(p[2]>0):
		plt.scatter([p[0],],[p[1],], 15, color ='red', marker = 'o')
	else:
		plt.scatter([p[0],],[p[1],], 15, color ='blue', marker = 'x')
grams = []
for i in X:
	gram = []
	for j in X:
		gram.append(i[0]*j[0]+i[1]*j[1])
	grams.append(gram)
a = []
for i in xrange(0,len(X)):
	a.append(0)

def isaCorrect(a,i,index):
	global X
	ayxx_sum = 0
	for j in xrange(0,len(a)):
		ayxx_sum = ayxx_sum + a[j]*X[j][2]*grams[j][index]
	r = (ayxx_sum + b)*X[index][2]
	return r

def getWandB(a):
	w = [0,0]
	for i in xrange(0,len(X)):
		w[0] = w[0] + a[i]*X[i][2]*X[i][0]
		w[1] = w[1] + a[i]*X[i][2]*X[i][1]
	b = 0
	for i in xrange(0,len(X)):
		b = b + a[i]*X[i][2]
	return w,b
w,b = getWandB(a)

# #initial line

if(w[0]!=0 and w[1]!=0):
	x1 = range(0,X[:,0].max()+2)
	x2 = []
	for i in x1:
		x2.append((-b-w[0]*i)/w[1])
elif(w[0]!=0 and w[1]==0):
	x2 = range(0,X[:,1].max()+2)
	x1 = []
	for i in x2:
		x1.append(-b/w[0])
elif(w[0]==0 and w[1]!=0):
	x1 = range(0,X[:,0].max()+2)
	x2 = []
	for i in x1:
		x2.append(-b/w[1])
else:
	x2 = range(0,X[:,1].max()+2)
	x1 = []
	for i in x2:
		x1.append(0)

lines = ax.plot(x1,x2,color="red", linewidth=2.5, linestyle="-")	



def isPerfect():
	global a,b,X
	n = 0.3
	flag = True
	ErrorPoint = []
	for index,i in enumerate(X):
		if(isaCorrect(a,i,index)<=0):
			ErrorPoint.append(index)
	if(len(ErrorPoint)==0):
		return flag
	else:
		flag = False
		temp_index = choice(ErrorPoint)
		p = X[temp_index]
		a[temp_index] = a[temp_index] + n
		b = b + n*p[2]
		print a
		print b
	return flag

def update(frame):
	global a,b,X
	flag = isPerfect()
	if(flag==True):
		print "Perfect!"
		return
	else:
		w,b = getWandB(a)
		# print w[0],w[1],b	
		if(w[0]!=0 and w[1]!=0):
			x1 = range(0,X[:,0].max()+2)
			x2 = []
			for i in x1:
				x2.append((-b-w[0]*i)/w[1])
		elif(w[0]!=0 and w[1]==0):
			x2 = range(0,X[:,1].max()+2)
			x1 = []
			for i in x2:
				x1.append(-b/w[0])
		elif(w[0]==0 and w[1]!=0):
			x1 = range(0,X[:,0].max()+2)
			x2 = []
			for i in x1:
				x2.append(-b/w[1])
		else:
			x2 = range(0,X[:,1].max()+2)
			x1 = []
			for i in x2:
				x1.append(0)
		lines = ax.plot(x1,x2,color="red", linewidth=2.5, linestyle="-")
		ax.lines.pop(0)
		return plt,

animation = FuncAnimation(fig, update,blit=False, interval=10)

# animation.save('rain.mp4', writer='ffmpeg')
plt.show()
