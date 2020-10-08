"""
Created on Thu Apr  4 19:15:33 2019

@author: psanh
"""
import numpy as np
import cv2



def gaussian(x, mu, cov_matrix, d):
    temp = np.zeros([np.shape(x)[0],np.shape(x)[1]])
    
    for i in range(np.shape(x)[0]):
        for j in range(np.shape(x)[1]):
            x_t = np.reshape(x[i,j,:],(1,-1))
            temp[i,j] = (1/(((2*np.pi)**(d/2))*np.sqrt(np.linalg.norm(cov_matrix))))*np.exp((-1/2)*np.matmul(np.matmul((x_t-mu),np.linalg.inv(cov_matrix)),np.transpose(x_t-mu)))
    return temp


def EM(x,K, mu, cov_matrix, d):
    
    pic = []

    
    for i in range(K):
        pic.append(1/K)
        
    r = np.zeros([np.shape(x)[0]*np.shape(x)[1],K])
        
    for i in range(K):
        cov = cov_matrix[:,:,i]
        mu_ = mu[i,:]
        r[:,i] = np.reshape((pic[i]*gaussian(x, mu_, cov, d)),(1,-1))/np.sum(np.reshape((pic[i]*gaussian(x, mu_, cov, d)),(1,-1)),axis = 1)

    
    ################
    #	#M Step#
    ################
    
    L = 0
    
    while L<15:
        m_c = []  
        for c in range(K):
            m_c.append(np.sum(r[:,c]))
        
        
        new_pic = []
        new_pic = r.sum(axis = 0)/r.sum()
#        for m in range(K):
#            new_pic.append(m/np.sum(m_c))
        
        new_muc = []
        new_cov = np.zeros([np.shape(cov_matrix)[0], np.shape(cov_matrix)[1], K]) 
    
        for c in range(K):
            temp = np.reshape(x,(np.shape(x)[0]*np.shape(x)[1],-1))
            r_temp = np.reshape(r[:,i],(-1,1))
            temp_muc = (1/m_c[c])*np.sum(temp*r_temp, axis = 0)
            
            new_muc.append((1/m_c[c])*np.sum(temp*r_temp, axis = 0))
            
            new_cov[:,:,c]= (1/m_c[c])*np.matmul(r_temp.T*(temp-temp_muc).T, (temp-temp_muc))
        
        L+=1
        r = np.zeros([np.shape(r)[0],K])
        
        for i in range(K):
            cov = new_cov[:,:,i]
            mu_ = new_muc[i]
            r[:,i] = np.reshape((new_pic[i]*gaussian(x, mu_, cov, d)),(1,-1))/np.sum(np.reshape((new_pic[i]*gaussian(x, mu_, cov, d)),(1,-1)),axis = 1)
    
            print("New mean: \n",new_muc)
            #print("New Cov: ",new_cov)
    return new_muc, new_cov


RR_ALL = np.array([],dtype=np.uint8)
RG_ALL = np.array([],dtype=np.uint8)
RB_ALL = np.array([],dtype=np.uint8)

N = 41


for i in range(0,N):
    green = cv2.imread('Green/green_%d.jpg' %i )
    yellow = cv2.imread('Yellow/yellow_%d.jpg' %i)
    red = cv2.imread('Red/red_%d.jpg' %i)
    
    if(i == 0):
        red_stack = red
        yellow_stack = yellow
        green_stack = green
    else: 
        red_stack = np.vstack((red_stack,red))
        yellow_stack = np.vstack((yellow_stack,yellow))
        green_stack = np.vstack((green_stack,green))
    
    

#cov_matrix = np.array([[715,700,600],[500,675,765],[543,420,520]])
#cov_matrix = np.reshape(cov_matrix, (np.shape(cov_matrix)[0], np.shape(cov_matrix)[1], -1)) 
#
#mu = np.reshape(np.array([200,150,235]),(-1,1))
        
cov_matrix = np.array([[715,0,0],[0,675,0],[0,0,430]])
cov_matrix = np.reshape(cov_matrix, (np.shape(cov_matrix)[0], np.shape(cov_matrix)[1], -1)) 

cov_matrix1 = np.array([[505,400,204],[100,275,400],[230,675,210]])
#cov_matrix1 = np.reshape(cov_matrix1, (np.shape(cov_matrix)[0], np.shape(cov_matrix)[1], -1)) 
cov1 = np.dstack((cov_matrix,cov_matrix1))
mu = np.reshape(np.array([200,150,230]),(1,-1))
#mu = np.array([200,150,230])
mu1 = np.array([220,160,215])
mu2 = np.vstack((mu,mu1))
#new_yellow = np.dstack((yellow_stack[:,:,1],yellow_stack[:,:,2]))
print("Training Started")
test1 = EM(green_stack, 1, mu, cov_matrix, 3)
print("===========================================================")
print("\nGREEN TRAINED\n")
print("===========================================================")

test = EM(red_stack, 1, mu, cov_matrix, 3)
print("===========================================================")
print("\nRED TRAINED\n")
print("===========================================================")

test = EM(yellow_stack, 1, mu, cov_matrix, 3)
print("===========================================================")
print("\nYELLOW TRAINED\n")
print("===========================================================")
print("\nAll BUOYS TRAINED\n")

#############################################################

# USE NEW TRAINED VALUES FOR BUOY TESTING IN ORIGINAL SCRIPT

#############################################################