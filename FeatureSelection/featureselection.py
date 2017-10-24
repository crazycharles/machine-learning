#coding:utf-8
#Write some feature selection algorithms,contents are listed below:
#Filter:Correlation coefficient,Chi-square test,Information gain,Mutual information
#Wrapper:Global search,Heuristic search,Random search(GA&SA)
#Embedded:Regularization(L1:Lasso,L2:Ridge),Decision tree(entropy,information gain),Deep learning
#Author:Chaojie An
#His github site:www.github.com/crazycharles 
#Time:22,Oct,2017

#Load wine data
#13 coloumn names
#Alcohol  
#Malic acid   
#Ash  
#Alcalinity of ash  
#Magnesium  
#Total phenols 
#Flavanoids  
#Nonflavanoid phenols  
#Proanthocyanins  
#Color intensity   
#Hue 
#OD280/OD315 of diluted wines  
#Proline 

# class 1 59
# class 2 71
# class 3 48
import pandas as pd
import math
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_wine
from sklearn.cross_validation import train_test_split
import random
wine = pd.read_csv('wine.csv')
# print wine.head()
y = wine.ix[:,0]
x = wine.ix[:,1:]

y_mean = y.mean()
# x = wine.ix[:,1:]

#Filter
#Correlation coefficient
def CorCoe(x,y):
	Length = len(x)
	sumxy = 0
	sumx2 = 0
	sumy2 = 0
	x_mean = x.mean()
	y_mean = y.mean()
	for i in range(0,Length):
		sumxy = sumxy + (x[i]-x_mean)*(y[i]-y_mean)
		sumx2 = sumx2 + pow((x[i]-x_mean),2)
		sumy2 = sumy2 + pow((y[i]-y_mean),2)
	r = sumxy/math.sqrt(sumx2*sumy2)
	return r
corcoe = []
for i in range(0,13):
	xs = wine.ix[:,i+1]
	corcoe.append(CorCoe(xs,y))
new_corcoe = []
for i in corcoe:
	if i>0:
		new_corcoe.append(i)
	else:
		new_corcoe.append(-i)
s = new_corcoe
def maxtomin(s,ss=-1,s_max=0):
	max_i = 0
	max_list = []
	max_index = []
	for i in range(0,len(s)):	
		for index,e in enumerate(s):
			if e>s_max:
				s_max = e
				max_i = index
		max_list.append(s_max)
		max_index.append(max_i)
		s[max_i]=ss
		s_max = 0
	return max_index,max_list
print "------------------correlation coefficent result---------------"
max_index,max_list = maxtomin(s)
for i in range(0,len(s)):
	print max_index[i],max_list[i]
#Chi-square test

def chi_square_fenbu(x,y):
	x_s = [0 if a<x.mean() else 1 for a in x]
	y1_p = 0
	y1_n = 0
	y2_p = 0
	y2_n = 0
	y3_p = 0
	y3_n = 0
	for i in range(0,178):
		if(y[i]==1):
			if(x_s[i]==1):
				y1_p = y1_p + 1
			else:
				y1_n = y1_n + 1
		if(y[i]==2):
			if(x_s[i]==1):
				y2_p = y2_p + 1
			else:
				y2_n = y2_n + 1
		if(y[i]==3):
			if(x_s[i]==1):
				y3_p = y3_p + 1
			else:
				y3_n = y3_n + 1
	chi = []
	chi.append(y1_p)
	chi.append(y1_n)
	chi.append(y2_p)
	chi.append(y2_n)
	chi.append(y3_p)
	chi.append(y3_n)
	return chi

def chi_square_test(chi):
	chi_result = 0
	above_mean = 0
	chi_sum = sum(chi)
	for i in range(0,6,2):
		above_mean = above_mean + chi[i]
	p = float(above_mean)/chi_sum
	new_chi = []
	for i in range(0,6,2):		
		new_chi.append(p*(chi[i]+chi[i+1]))
		new_chi.append((1-p)*(chi[i]+chi[i+1]))
	for i in range(0,6):
		chi_result = chi_result + pow((new_chi[i]-chi[i]),2)/new_chi[i]
	return chi_result
chi_test = []
for i in range(0,13):
	x = wine.ix[:,i+1]
	chi = chi_square_fenbu(x,y)
	chi_test.append(chi_square_test(chi))
print "--------------------Chi-square test result-----------------"
max_index,max_list = maxtomin(chi_test)
for i in range(0,len(chi_test)):
	print max_index[i],max_list[i]
	
#Information gain
#Mutual information
def mi(x,y):
	#y=1,y=2,y=3
	mi_value = 0
	for j in range(0,59):
		factor1 = x[0:59].count(x[j])/59.0
		factor2 = math.log(factor1)-math.log(x.count(x[j])*59.0/178/178)
		mi_value = mi_value + factor1*factor2
	for j in range(59,130):
		factor1 = x[59:130].count(x[j])/71.0
		factor2 = math.log(factor1)-math.log(x.count(x[j])*71.0/178/178)
		mi_value = mi_value + factor1*factor2
	for j in range(130,177):
		factor1 = x[130:177].count(x[j])/48.0
		factor2 = math.log(factor1)-math.log(x.count(x[j])*48.0/178/178)
		mi_value = mi_value + factor1*factor2
	return mi_value
mi_list = []
for i in range(0,13):	
	x = wine.ix[:,i+1]
	new_x = []
	for i in x:
		new_x.append(i)
	mi_list.append(mi(new_x,y))
print "----------------------mutual information--------------"
maxtomin(mi_list,-1000,-1000)
for i in range(0,len(mi_list)):
	print max_index[i],max_list[i]
#Wrapper

#decision tree, the adaptability function
def decision_tree(personal = []):
	del_fea = []
	for index,i in enumerate(personal):
		if(i==0):
			del_fea.append(index)
	data = load_wine()
	X_train = data.data
	y_train = data.target
	X_train = np.delete(X_train, del_fea, axis=1)


	dtree = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
	dtree.fit(X_train, y_train)
	acc_dtree = cross_val_score(estimator = dtree, X = X_train, y = y_train, cv = 10)
	dtree_acc_mean = acc_dtree.mean()
	dtree_std = acc_dtree.std()
	return dtree_acc_mean

number = 1000
population = np.random.randint(0,2,(number,13))
def genetic_algorithm(population):
	adaptability = []
	for personal in population:
		adaptability.append(decision_tree(personal))
	return maxtomin(adaptability)
print "------------------genetic_algorithm---------------"
print "------------------There'll be a long time if you set a large number--------------"
#product 1000 personals
#cross
def cross(f):
	a = random.randint(0,len(f)-1)
	b = random.randint(0,len(f)-1)
	if(a!=b):
		c = random.randint(0,12)
		t = []
		for i in range(c,13):
			t.append(f[a][i])
		for i in range(c,13):
			f[a][i] = f[b][i]
			f[b][i] = t[i-c]
	return f
def topNsum(n,e_list):
	'''select the top N elements and get their sum'''
	topnsum = 0
	for x in xrange(0,n):
		topnsum = topnsum + e_list[x]
	return topnsum	
#variation
def variation(f):
	a = random.randint(0,len(f)-1)
	b = random.randint(0,12)
	if(f[a][b])==0:
		f[a][b]==1
	else:
		f[a][b]==0
	return f

for iters in xrange(0,5):
	index_gene,gene_adp= genetic_algorithm(population)
	lunpan = []
	for ig in gene_adp:
		lunpan.append(ig/sum(gene_adp))

	#select the best personal 1000
	population_2 = []
	for i in range(0,len(index_gene)):	
		select = random.random()
		for i in xrange(1,len(lunpan)+1):
			if(topNsum(i-1,lunpan)<=select<=topNsum(i,lunpan)):
				population_2.append(population[index_gene[i-1]])
				break
	for i in xrange(0,int(number*0.6)):
		population_2 = cross(population_2)
	for i in xrange(0,int(number*0.05)):
		population_2 = variation(population_2)
	population = population_2
index_gene,gene_adp = genetic_algorithm(population)
print gene_adp
print population[0]

#Global search
#Heuristic search
#Random search
#GA
#SA

#Embedded
#Regularization
#L1
#L2

#Decision tree
# def gini(y):
# 	y1 = float(y.count(1))
# 	y2 = float(y.count(2))
# 	y3 = float(y.count(3))
# 	gini = 1-pow(y1/len(y),2)-pow(y2/len(y),2)-pow(y3/len(y),2)
# 	return gini

# def decision_tree(x,y):
# 	for index,i in enumerate(x):
# 		jun = (max(i)-min(i))/5.0
# 		hou = []
# 		for j in range(0,5):
# 			hou.append(min(i)+jun)
# 		for j in hou:
# 			index2_list = []
# 			for index2,iy in enumerate(y):
# 				if(i)

#Deep learning

