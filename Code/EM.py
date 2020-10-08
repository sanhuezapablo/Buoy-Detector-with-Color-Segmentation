import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import scipy
import random as rd
from scipy.stats import norm

np.set_printoptions(suppress=True)

def calc(val):
	mean=np.mean(val)
	variance=np.var(val)
	return mean,variance

def gaussian(x, mu, sig):
    return ((1/(sig*math.sqrt(2*math.pi)))*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))


np.random.seed(0)
x=np.linspace(-5,5,num=50)
x1=x*np.random.rand(len(x))
x2=x*np.random.rand(len(x))-10
x3=x*np.random.rand(len(x))+10
x_total=np.concatenate((x1,x2,x3))
#print(x_total.shape)
pic=[1/3,1/3,1/3]
y1=[]
y2=[] 	
y3=[]
mu1=-8
mu2=8
mu3=5
std1=5
std2=3
std3=1

# xa1=np.linspace((-5*std1+mu1),(5*std1+mu1),150)
# xa2=np.linspace((-5*std2+mu2),(5*std2+mu2),150)
# xa3=np.linspace((-5*std3+mu3),(5*std3+mu3),150)
xa1=np.linspace(-25,25,1000)
ya1=gaussian(xa1,mu1,std1)
ya2=gaussian(xa1,mu2,std2)
ya3=gaussian(xa1,mu3,std3)

for i in range(0,len(x_total)):
 	y1.append(gaussian(x_total[i],mu1,std1)*pic[0])
 	y2.append(gaussian(x_total[i],mu2,std2)*pic[1])
 	y3.append(gaussian(x_total[i],mu3,std3)*pic[2])


y_total=np.transpose(np.vstack((y1,y2,y3)))


for i in range(len(y_total)):
    y_total[i] = y_total[i]/(np.sum(pic)*np.sum(y_total,axis=1)[i])
print(y_total.shape)    

for i in range(len(y_total)):
		plt.scatter(x_total[i],0,color=np.array([y_total[i][0],y_total[i][1],y_total[i][2]]),s=100)
plt.plot(xa1,ya1,c='red')
plt.plot(xa1,ya2,c='green')
plt.plot(xa1,ya3,c='blue')
plt.show()

###############
	#M Step#
###############
j=0
iterations=[]
log_like_pi=[]
log_like_m=[[mu1,mu2,mu3]]
log2=[]
while (j<100):

	prob_temp=[]
	for c in range(len(y_total[0])):
		m=np.sum(y_total[:,c])
		prob_temp.append(np.sum(y_total[:,c]))


	new_pi = []
	for m in prob_temp:
		new_pi.append(m/np.sum(prob_temp)) 

	# New Mean
	new_m=np.sum(x_total.reshape(len(x_total),1)*y_total,axis=0)/prob_temp

	#New Std
	new_std=[]
	
	for c in range(3):
	    new_std.append(np.sqrt((1/prob_temp[c])*np.dot(((np.array(y_total[:,c]).reshape(150,1))*(x_total.reshape(len(x_total),1)-new_m[c])).T,(x_total.reshape(len(x_total),1)-new_m[c]))))

	ya1=gaussian(xa1,new_m[0],new_std[0][0][0])
	ya2=gaussian(xa1,new_m[1],new_std[1][0][0])
	ya3=gaussian(xa1,new_m[2],new_std[2][0][0])

	iterations.append(j)
	log_like_pi.append(new_pi)
	log2.append(new_m)
	log_like_m.append(new_m)
	
	if abs(log_like_m[len(log_like_m)-1][0]-log_like_m[len(log_like_m)-2][0])<0.0001 and abs(log_like_m[len(log_like_m)-1][1]-log_like_m[len(log_like_m)-2][1])<0.0001 and abs(log_like_m[len(log_like_m)-1][2]-log_like_m[len(log_like_m)-2][2])<0.0001: 
		break
	if abs(new_pi[0]-new_pi[1]) <0.001 and abs(new_pi[1]-new_pi[2])<0.001 and abs(new_pi[2]-new_pi[0])<0.001:
		break
	#Update 
	y1=[]
	y2=[]
	y3=[]

	for i in range(0,len(x_total)):
	 	y1.append(gaussian(x_total[i],new_m[0],new_std[0][0][0])*new_pi[0])
	 	y2.append(gaussian(x_total[i],new_m[1],new_std[1][0][0])*new_pi[1])
	 	y3.append(gaussian(x_total[i],new_m[2],new_std[2][0][0])*new_pi[2])
	
	
	y_total=np.transpose(np.vstack((y1,y2,y3)))
	
	for i in range(len(y_total)):
	    y_total[i] = y_total[i]/(np.sum(pic)*np.sum(y_total,axis=1)[i])

	new_m=[]
	new_std=[]
	new_pi=[]
	prob_temp=[]
	print (j)
	j=j+1

	


#Plot
for i in range(len(y_total)):
		plt.scatter(x_total[i],0,color=np.array([y_total[i][0],y_total[i][1],y_total[i][2]]),s=100)
plt.plot(xa1,ya1,c='red')
plt.plot(xa1,ya2,c='green')
plt.plot(xa1,ya3,c='blue')
plt.show()


plt.title('Convergence')
plt.plot(iterations,log_like_pi)
plt.show()
plt.title('Mean Convergence')
plt.plot(iterations,log2)
plt.show()