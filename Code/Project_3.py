import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
import sys
 

def gaussian(x, mu, sig):
    return (1/(sig*np.sqrt(2*np.pi)))*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def add_gaussians(x,mu, std):
    X = np.zeros([])
    for i in range(len(mu)):
        X = X + gaussian(x,mu[i], std[i])
    return X

def x_data(hist,K):
    X = []
    for i in range(256):
        if int(hist[i])>0:
            x = []
            x = [i]*int(hist[i])
            for i in range(len(x)):
                X.append(x[i])
                
    mu = []
    index = np.random.randint(0,len(X),K*1)
    for i in range(K):
        mu.append(X[index[i]])
    
    return np.array(X), mu


def EM(x,K, mu, std):
    
    pic = []
    
    for i in range(K):
        pic.append(1/K)
    
    r = np.zeros([len(x),K])
    
    for i in range(0,K):
        for j in range(0,len(x)):
            r[j][i] = gaussian(x[j],mu[i],std[i])*pic[i]
       
    for i in range(len(r)):
        r[i] = r[i]/(np.sum(pic)*np.sum(r,axis=1)[i])
         
    ################
    #	#M Step#
    ################
    
    L=0
    
    iterations=[]
    log_like_pi=[]
    log_like_m=[mu]
    log2=[]
    
    while (L<500):
        
        new_m=[]
        new_std=[]
        new_pi=[]
        prob_temp=[]
    
        prob_temp=[]
        for c in range(len(r[0])):
        	m=np.sum(r[:,c])
        	prob_temp.append(np.sum(r[:,c]))
        
        
        for m in prob_temp:
            new_pi.append(m/np.sum(prob_temp)) 
        
        # New Mean
        new_m=np.sum(x.reshape(len(x),1)*r,axis=0)/prob_temp
        
        #New Std        	
        for c in range(K):
            new_std.append(np.sqrt((1/prob_temp[c])*np.dot(((np.array(r[:,c]).reshape(len(x),1))*(x.reshape(len(x),1)-new_m[c])).T,(x.reshape(len(x),1)-new_m[c]))))
        
        
        iterations.append(j)
        log_like_pi.append(new_pi)
        log2.append(new_m)
        log_like_m.append(new_m)
        
        for i in range(K):
            adheesh=np.sum(log_like_m,axis=1)/(K)
        if abs((adheesh[len(adheesh)-1])-(adheesh[len(adheesh)-2]))<0.001:
            break
       
        r = np.zeros([len(x),K])
        
        for i in range(0,K):
            for j in range(0,len(x)):
                r[j][i] = gaussian(x[j],new_m[i],new_std[i][0][0])*new_pi[i]
            
        for i in range(len(r)):
            r[i] = r[i]/(np.sum(new_pi)*np.sum(r,axis=1)[i])
       
        L+=1
    
   
    return new_m, new_std

new_hg_r = np.zeros([256,1],dtype=np.float32)
new_hg_g = np.zeros([256,1],dtype=np.float32)
new_hg_b = np.zeros([256,1],dtype=np.float32)

new_hy_r = np.zeros([256,1],dtype=np.float32)
new_hy_g = np.zeros([256,1],dtype=np.float32)
new_hy_b = np.zeros([256,1],dtype=np.float32)

new_hr_r = np.zeros([256,1],dtype=np.float32)
new_hr_g = np.zeros([256,1],dtype=np.float32)
new_hr_b = np.zeros([256,1],dtype=np.float32)

pdf_gg = np.zeros([16,16],dtype=np.uint8)
pdf_rr = np.zeros([16,16],dtype=np.uint8)
pdf_yy = np.zeros([16,16],dtype=np.uint8)

c_yellow =  np.zeros([16,16],dtype=np.uint8)
c_red =  np.zeros([16,16],dtype=np.uint8)
c_green =  np.zeros([16,16],dtype=np.uint8)

Avg_Red = np.zeros([16,16,3],dtype=np.uint8)
Avg_Green = np.zeros([16,16,3],dtype=np.uint8)
Avg_Yellow = np.zeros([16,16,3],dtype=np.uint8)



N = 41
new_G = []
new_R = []
new_Y = []

for i in range(0,N):
    green = cv2.imread('Green/green_%d.jpg' %i )
    yellow = cv2.imread('Yellow/yellow_%d.jpg' %i)
    red = cv2.imread('Red/red_%d.jpg' %i)

    green = cv2.GaussianBlur(green,(5,5),0) 
    yellow = cv2.GaussianBlur(yellow,(5,5),0) 
    red = cv2.GaussianBlur(red,(5,5),0) 
    
    #hg_r = cv2.calcHist([green],[2],None,[256],[0,300])
    
    # hg_g = cv2.calcHist([green],[1],None,[256],[0,300])
    # hg_b = cv2.calcHist([green],[0],None,[256],[0,300])
    
    hg_r,bins = np.histogram(green[:,:,2],256,[0,256])
    hg_r=np.reshape(hg_r,(-1,1))
    hg_g,bins = np.histogram(green[:,:,1],256,[0,256])
    hg_g=np.reshape(hg_g,(-1,1))
    hg_b,bins = np.histogram(green[:,:,0],256,[0,256])
    hg_b=np.reshape(hg_b,(-1,1))
    

    new_hg_r[:] = new_hg_r[:] + hg_r[:]
    new_hg_g[:] = new_hg_g[:] + hg_g[:]
    new_hg_b[:] = new_hg_b[:] + hg_b[:]
    
    hy_r,bins = np.histogram(yellow[:,:,2],256,[0,256])
    hy_r=np.reshape(hy_r,(-1,1))
    hy_g,bins = np.histogram(yellow[:,:,1],256,[0,256])
    hy_g=np.reshape(hy_g,(-1,1))
    hy_b,bins = np.histogram(yellow[:,:,0],256,[0,256])
    hy_b=np.reshape(hy_b,(-1,1))
    

    new_hy_r[:] = new_hy_r[:] + hy_r[:]
    new_hy_g[:] = new_hy_g[:] + hy_g[:]
    new_hy_b[:] = new_hy_b[:] + hy_b[:]
    
    hr_r,bins = np.histogram(red[:,:,2],256,[0,256])
    hr_r=np.reshape(hr_r,(-1,1))
    hr_g,bins = np.histogram(red[:,:,1],256,[0,256])
    hr_g=np.reshape(hr_g,(-1,1))
    hr_b,bins = np.histogram(red[:,:,0],256,[0,256])
    hr_b=np.reshape(hr_b,(-1,1))
    

    new_hr_r[:] = new_hr_r[:] + hr_r[:]
    new_hr_g[:] = new_hr_g[:] + hr_g[:]
    new_hr_b[:] = new_hr_b[:] + hr_b[:]
      
    RR_Red = red[:,:,2]
    RR_Green = red[:,:,1]
    RR_Blue = red[:,:,0]
    
    GG_Red = green[:,:,2]
    GG_Green = green[:,:,1]
    GG_Blue = green[:,:,0]
    
    YY_Red = yellow[:,:,2]
    YY_Green = yellow[:,:,1]
    YY_Blue = yellow[:,:,0]
    
    
    GG = green[:,:,1]
    YY = (yellow[:,:,1].astype(int)+yellow[:,:,2].astype(int))/2
    
    Avg_Red = Avg_Red.astype(int) + red.astype(int)
    Avg_Green = Avg_Green.astype(int) + green.astype(int)
    Avg_Yellow = Avg_Yellow.astype(int) + yellow.astype(int)
            
    
    
    
print("Training Buoys")    
new_hg_r[:]=new_hg_r[:]/N
new_hg_g[:]=new_hg_g[:]/N
new_hg_b[:]=new_hg_b[:]/N

new_hy_r[:]=new_hy_r[:]/N
new_hy_g[:]=new_hy_g[:]/N
new_hy_b[:]=new_hy_b[:]/N

new_hr_r[:]=new_hr_r[:]/N
new_hr_g[:]=new_hr_g[:]/N
new_hr_b[:]=new_hr_b[:]/N

Avg_Red = Avg_Red/N
Avg_Green = Avg_Green/N
Avg_Yellow = Avg_Green/N

Avg_Red_RedChannel = np.mean(Avg_Red[:,:,2])  
Avg_Green_GreenChannel = np.mean(Avg_Green[:,:,1])  




xa = np.linspace(0,255,1000)


"""GREEN"""
std_green_green = [1,2,1]   
std_green_red = [1,3]
std_green_blue = [1,2]    

XG_G, mu_green_green = x_data(new_hg_g,3)
XG_R, mu_green_red = x_data(new_hg_r,2)
XG_B ,mu_green_blue = x_data(new_hg_b,2)

mu_green_r, std_green_r = EM(XG_R,2,mu_green_red,std_green_red)
mu_green_g, std_green_g = EM(XG_G,3,mu_green_green,std_green_green)
mu_green_b, std_green_b = EM(XG_B,2,mu_green_blue ,std_green_blue)


GREEN_RED = add_gaussians(xa,mu_green_r, std_green_r)
GREEN_GREEN = add_gaussians(xa,mu_green_g, std_green_g)
GREEN_BLUE = add_gaussians(xa,mu_green_b, std_green_b)
print("Green Trained")
# plt.figure(0)
# plt.fill(new_hg_b/256,color='b')
# plt.plot(xa,GREEN_BLUE.T, color='blue',  linestyle='dashed',linewidth=2)
# plt.fill(new_hg_r/256,color='r')
# plt.plot(xa, GREEN_RED.T, color='red',  linestyle='dashed',linewidth=2)
# plt.fill(new_hg_g/256,color='g')
# plt.plot(xa,GREEN_GREEN.T, color='green',  linestyle='dashed',linewidth=2)
# plt.title("Green Buoy")
# plt.show()


# """RED"""

std_red_green = [1,2]   
std_red_red = [1,3,2]
std_red_blue = [1,2]    

XR_G, mu_red_green = x_data(new_hr_g,2)
XR_R, mu_red_red = x_data(new_hr_r,3)
XR_B ,mu_red_blue = x_data(new_hr_b,2)

mu_red_r, std_red_r = EM(XR_R,3,mu_red_red,std_red_red)
mu_red_g, std_red_g = EM(XR_G,2,mu_red_green,std_red_green)
mu_red_b, std_red_b = EM(XR_B,2,mu_red_blue ,std_red_blue)


RED_RED = add_gaussians(xa,mu_red_r, std_red_r)
RED_GREEN = add_gaussians(xa,mu_red_g, std_red_g)
RED_BLUE = add_gaussians(xa,mu_red_b, std_red_b)
print("Red Trained")
# plt.figure(1)
# plt.fill(new_hr_r/256,color='r')
# plt.plot(xa, RED_RED.T, color='red',  linestyle='dashed',linewidth=2)
# plt.fill(new_hr_g/256,color='g')
# plt.plot(xa,RED_GREEN.T, color='green',  linestyle='dashed',linewidth=2)
# plt.fill(new_hr_b/256,color='b')
# plt.plot(xa,RED_BLUE.T, color='blue',  linestyle='dashed',linewidth=2)
# plt.title("Red Buoy")
# plt.show()

"""YELLOW"""

std_yellow_red = [1,2,1]
std_yellow_green = [1,2,1]   
std_yellow_blue = [1,2]  
 
XY_R, mu_yellow_red = x_data(new_hy_r, 3)
XY_G, mu_yellow_green = x_data(new_hy_g, 3)
XY_B ,mu_yellow_blue = x_data(new_hy_b, 2)

mu_yellow_r, std_yellow_r = EM(XY_R, 3, mu_yellow_red, std_yellow_red)
mu_yellow_g, std_yellow_g = EM(XY_G, 3, mu_yellow_green, std_yellow_green)
mu_yellow_b, std_yellow_b = EM(XY_B, 2, mu_yellow_blue, std_yellow_blue)


YELLOW_RED = add_gaussians(xa, mu_yellow_r, std_yellow_r)
YELLOW_GREEN = add_gaussians(xa, mu_yellow_g, std_yellow_g)
YELLOW_BLUE = add_gaussians(xa, mu_yellow_b, std_yellow_b)
print("Yellow Trained")
# plt.figure(1)
# plt.fill(new_hr_r/256,color='r')
# plt.plot(xa,  YELLOW_RED.T, color='red',  linestyle='dashed',linewidth=3)
# plt.fill(new_hy_g/256,color='g')
# plt.plot(xa, YELLOW_GREEN.T, color='green',  linestyle='dashed',linewidth=3)
# plt.fill(new_hy_b/256,color='b')
# plt.plot(xa, YELLOW_BLUE.T, color='blue',  linestyle='dashed',linewidth=3)
# plt.title("Yellow Buoy")
# plt.show()
mu_green_green=np.array(mu_green_g)
mu_red_red=np.array(mu_red_r)
mu_yellow_red=np.array(mu_yellow_r)
mu_yellow_green=np.array(mu_yellow_g)
std_green_green=np.array([std_green_g[0][0][0],std_green_g[1][0][0],std_green_g[2][0][0]])
std_red_red=np.array([std_red_r[0][0][0],std_red_r[1][0][0],std_red_r[2][0][0]])
std_yellow_red=np.array([std_yellow_r[0][0][0],std_yellow_r[1][0][0],std_yellow_r[2][0][0]])
std_yellow_green=np.array([std_yellow_g[0][0][0],std_yellow_g[1][0][0],std_yellow_g[2][0][0]])

print("All Buoys Trained")
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
count=0
cap=cv2.VideoCapture('detectbuoy.avi')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('test1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while(cap.isOpened()):
    
    ret,frame=cap.read()
    if ret==True:
        image=copy.deepcopy(frame)
        image2=copy.deepcopy(frame)
        image_g=image[:,:,1]
        image_r=image[:,:,2]
        img_out=np.zeros(image_r.shape, dtype = np.uint8)
        image = cv2.GaussianBlur(image,(5,5),100) 
        image = cv2.medianBlur(image,5) 
        grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
        prob_green = np.zeros([np.shape(image)[0],np.shape(image)[1]])
        prob_red = np.zeros([np.shape(image)[0],np.shape(image)[1]])
        prob_yellow = np.zeros([np.shape(image)[0],np.shape(image)[1]])
        
        prob_green = add_gaussians(image[:,:,1], mu_green_green, std_green_green)
        prob_red = add_gaussians(image[:,:,2], mu_red_red, std_red_red)
        prob_yellow = (add_gaussians(image[:,:,1], mu_yellow_green, std_yellow_green)+add_gaussians(image[:,:,2],mu_yellow_red,std_yellow_red))/2
        #prob_yellow = add_gaussians(image[:,:,2], new_mu_yellow, new_std_yellow)+add_gaussians(image[:,:,1], new_mu_yellow, new_std_yellow)
        
        for i in range(0,image.shape[0]):
            for j in range(0,image.shape[1]):
                if prob_red[i][j]>0.001 and image_g[i][j]<170:
                    img_out[i][j]=255
                else:
                    img_out[i][j]=0  

        ret_red, threshold_red = cv2.threshold(img_out, 240, 255, cv2.THRESH_BINARY)   
        closing_red = cv2.dilate(threshold_red,kernel,iterations = 1)
        closing_red = cv2.morphologyEx(closing_red, cv2.MORPH_OPEN, kernel)
        _,contours_red, hierarchy= cv2.findContours(closing_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours_red:
            
            if cv2.contourArea(contour) >  10:
                (x,y),radius = cv2.minEnclosingCircle(contour)
                center = (int(x),int(y))
                radius = int(radius)
                if radius > 10 and radius < 500:
                    cv2.circle(image2,center,radius,(0,0,255),2)

        for i in range(0,image.shape[0]):
            for j in range(0,image.shape[1]):
                
                if prob_green[i][j]>0.05 and image_r[i][j]<180 and i<400:
                    #print(ans_b[y], 'b')
                    img_out[i][j]=255
                else:
                    img_out[i][j]=0  

        ret_green, threshold_green = cv2.threshold(img_out, 230, 255, cv2.THRESH_BINARY)                    
        #opening = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel)   
        closing_green = cv2.dilate(threshold_green,kernel,iterations = 2)
        #closing_green = cv2.morphologyEx(threshold_green, cv2.MORPH_CLOSE, kernel)
        #closing1 = cv2.morphologyEx(closing1, cv2.MORPH_OPEN, kernel)
        
        _,contours_green, hierarchy= cv2.findContours(closing_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        for contour in contours_green:
            #print(contour)
            if cv2.contourArea(contour) >  5:
                (x,y),radius = cv2.minEnclosingCircle(contour)
                center = (int(x),int(y))
                radius = int(radius)
                if radius > 9 and radius < 16:
                    cv2.circle(image2,center,radius,(0,255,0),2)
            
        for i in range(0,image.shape[0]):
            for j in range(0,image.shape[1]):
                
                #if prob_green[i][j]>0.00000001 and prob_red[i][j]>0.0000001 and image_r[i][j]<240 and image_g[i][j]<250:
                #if prob_yellow[i][j]>0.04 and prob_red[i][j]<0.002 and prob_green[i][j]<0.05 and image_r[i][j]<230 and image_g[i][j]<230 :
                if prob_yellow[i][j]>0.17 and prob_red[i][j]>0.01 and image_r[i][j]<240 and j<450:
                
                    img_out[i][j]=255
                else:
                    img_out[i][j]=0  

        #closing_yellow = cv2.adaptiveThreshold(img_out,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,5,2)                    
        closing_yellow = cv2.adaptiveThreshold(img_out,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)                    
        #closing_yellow = cv2.morphologyEx(closing_yellow, cv2.MORPH_TOPHAT, kernel)   
        #closing_yellow = cv2.morphologyEx(closing_yellow, cv2.MORPH_OPEN, kernel)  
        closing_yellow = cv2.morphologyEx(closing_yellow, cv2.MORPH_CLOSE, kernel)  
        
        closing_yellow = cv2.dilate(closing_yellow,kernel,iterations = 1)    
        #closing_yellow = cv2.dilate(threshold_yellow,kernel,iterations = 2)    
        #_,contours_yellow, hierarchy= cv2.findContours(closing_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        _,contours_yellow, hierarchy= cv2.findContours(closing_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for contour in contours_yellow:
            if cv2.contourArea(contour) >  30:
                (x,y),radius = cv2.minEnclosingCircle(contour)
                center = (int(x),int(y))
                radius = int(radius)
                if radius > 11 and radius < 45:
                    cv2.circle(image2,center,radius,(0,255,255),2)

        cv2.imshow("red - green - yellow",np.hstack((closing_red,closing_green,closing_yellow)))
        cv2.moveWindow("red - green - yellow", 0,0)
        cv2.imshow('test',image2) 
        cv2.moveWindow("test", 300,300)
        out.write(image2)
        count=count+1
        """YELLOW"""
        # prob_yellow = (add_gaussians(image[:,:,1], mu_yellow_green, std_yellow_green)+add_gaussians(image[:,:,2],mu_yellow_red,std_yellow_red))/2
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        break
print("done")
cap.release()
cv2.destroyAllWindows()