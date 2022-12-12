from scipy.spatial import distance,cKDTree
import numpy as np
import cv2
from PIL import Image as im
import time

stereoImg=np.asarray(cv2.imread("/home/raghav/new100stereo.png",cv2.IMREAD_GRAYSCALE))
monoImg=np.asarray(cv2.imread("/home/raghav/new100mono.png",cv2.IMREAD_GRAYSCALE))
rgbImg=np.asarray(cv2.imread("/home/raghav/new100rgb.png"))


def fusion(orgimg,Zs,Zm):
    Zs=Zs+1e-7
    Zm=Zm+1e-7
   
    # Zs=cv2.cvtColor(Zs,cv2.COLOR_RGB2GRAY)+1e-7
    # cv2.imshow("Zs",Zs)
    # cv2.waitKey(1)
    # Zm=cv2.cvtColor(Zm,cv2.COLOR_BGR2GRAY)+1e-7
    maxZs=np.max(Zs)
    maxZm_scale=255
    ratio=maxZs/maxZm_scale
    Z_final=np.zeros(Zs.shape)
    # print(Zm.shape)
    # print(Zs.shape)
    # print(orgimg.shape)
    #print(maxZs,maxZm)
    Wc=get_Wc(orgimg)
    # Wc=(Wc*255).astype(np.uint8)
    # cv2.imshow("wc",Wc)
    # cv2.waitKey(1)
    # print(Wc)
    Nzm=Zm/maxZm_scale
    Nzs=Zs/maxZs
    Ws=np.where(Nzs>Nzm,Nzm/Nzs,Nzs/Nzm)
    
    Z_final=np.where(Zs==0,Zm*ratio,Wc*Zs+(1-Wc)*((1-Ws)*Zm*ratio+Ws*Zs))
    #print(np.max(Z_final))
    # print(Z_final)
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
    # print(imgCanny.shape)
    edges_coords=np.where(imgCanny==255)
    edges_coords=np.column_stack((edges_coords[0],edges_coords[1]))
    # print(edges_coords)
    # print(imgCanny[0][42])
    tree = cKDTree(edges_coords)
    
    coords=np.where(stereoImg>-1)
    coords=np.column_stack((coords[0],coords[1]))

    min_distance_arr=tree.query(coords,distance_upper_bound=5)[0].reshape(256,256)
    
    
    # min_distance_arr=np.min(distance.cdist(np.atleast_1d(coords), edges_coords,'cityblock'),axis=1).reshape(256,256)
    # for y in range(img.shape[0]):
    #     for x in range(img.shape[1]):
    #         min_distance_arr[y,x]=min_distance([[y,x]],edges_coords)
           
    # print(min_distance_arr)
    
    #min_distance_arr=np.where(min_distance_arr>5,5,min_distance_arr)
   
    #print(min_distance_arr[0][0])
    wc= 1/(1+np.exp(0.25*min_distance_arr))
    return wc

start_time=time.time()
Z_final=fusion(rgbImg,stereoImg,monoImg)

Z_final=cv2.normalize(np.float32(Z_final),None,0,1,norm_type=cv2.NORM_MINMAX)
Z_final = cv2.medianBlur(Z_final, 5)
Z_final=(Z_final*255).astype(np.uint8)
Z_final=cv2.applyColorMap(Z_final,cv2.COLORMAP_MAGMA)
final_time=time.time()

print("INFERENCE TIME : " ,final_time-start_time)
#Z_final=(fusion(rgbImg,stereoImg,monoImg)*2)
#image=im.fromarray(Z_final)


# stereoImg=cv2.normalize(np.float32(stereoImg),None,0,1,norm_type=cv2.NORM_MINMAX)
# stereoImg=(stereoImg*255).astype(np.uint8)
stereoImg=cv2.applyColorMap(stereoImg,cv2.COLORMAP_MAGMA)


# monoImg=cv2.normalize(np.float32(monoImg),None,0,1,norm_type=cv2.NORM_MINMAX)
# monoImg=(monoImg*255).astype(np.uint8)
monoImg=cv2.applyColorMap(monoImg,cv2.COLORMAP_MAGMA)




cv2.imshow("Fused Image",Z_final)
cv2.waitKey(1)

cv2.imshow("Stereo Image",stereoImg)
cv2.waitKey(1)

cv2.imshow("MIDAS Image",monoImg)
cv2.waitKey(1)

cv2.imshow("Original RGB Image",rgbImg)
cv2.waitKey(0)





