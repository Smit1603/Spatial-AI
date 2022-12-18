from scipy.spatial import distance,cKDTree
import numpy as np
import cv2
from PIL import Image as im
import time
import os 



Imagepath=[]
Folder_names={0:'RGBMap',1:'MDEMap',2:'StereoMap'}


for i in range(3):
    folderpath=os.path.dirname(os.path.abspath(__file__))+"/"+Folder_names[i]
    
    
    try:
        filename=os.listdir(folderpath)[0]
        Imagepath.append(folderpath+"/"+filename)
    except: 
        raise Exception("Corresponding image has not been correctly stored inside {} folder".format(Folder_names[i]))

stereoImg=np.asarray(cv2.imread(Imagepath[2],cv2.IMREAD_GRAYSCALE))
monoImg=np.asarray(cv2.imread(Imagepath[1],cv2.IMREAD_GRAYSCALE))
rgbImg=np.asarray(cv2.imread(Imagepath[0]))


def fusion(orgimg,Zs,Zm):
    Zs=Zs+1e-7
    Zm=Zm+1e-7
   
    maxZs=np.max(Zs)
    maxZm_scale=255
    ratio=maxZs/maxZm_scale
    Z_final=np.zeros(Zs.shape)
    Wc=get_Wc(orgimg)
    Nzm=Zm/maxZm_scale
    Nzs=Zs/maxZs
    Ws=np.where(Nzs>Nzm,Nzm/Nzs,Nzs/Nzm)
    
    Z_final=np.where(Zs==0,Zm*ratio,Wc*Zs+(1-Wc)*((1-Ws)*Zm*ratio+Ws*Zs))
    return Z_final

def get_Wc(frame):
    
    
    img=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    

    img = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)
    v = np.median(img)
    sigma = 0.33
    lower_thresh = int(max(0, (1.0 - sigma) * v))
    upper_thresh = int(min(255, (1.0 + sigma) * v))
    imgCanny=cv2.Canny(img,lower_thresh,upper_thresh)
    imgCanny=np.asarray(imgCanny)
    cv2.imshow("iCanny",imgCanny)
    cv2.waitKey(1)
    edges_coords=np.where(imgCanny==255)
    edges_coords=np.column_stack((edges_coords[0],edges_coords[1]))
    tree = cKDTree(edges_coords)
    
    coords=np.where(stereoImg>-1)
    coords=np.column_stack((coords[0],coords[1]))

    min_distance_arr=tree.query(coords,distance_upper_bound=5)[0].reshape(256,256)
    wc= 1/(1+np.exp(0.25*min_distance_arr))
    return wc

start_time=time.time()
Z_final=fusion(rgbImg,stereoImg,monoImg)

Z_final=cv2.normalize(np.float32(Z_final),None,0,1,norm_type=cv2.NORM_MINMAX)
Z_final = cv2.medianBlur(Z_final, 5)
Z_final=(Z_final*255).astype(np.uint8)
Z_final=cv2.applyColorMap(Z_final,cv2.COLORMAP_MAGMA)
final_time=time.time()




stereoImg=cv2.applyColorMap(stereoImg,cv2.COLORMAP_MAGMA)


monoImg=cv2.applyColorMap(monoImg,cv2.COLORMAP_MAGMA)




cv2.imshow("Fused Image",Z_final)
cv2.waitKey(1)

cv2.imshow("Stereo Image",stereoImg)
cv2.waitKey(1)

cv2.imshow("MIDAS Image",monoImg)
cv2.waitKey(1)

cv2.imshow("Original RGB Image",rgbImg)
cv2.waitKey(0)





