import Augmentor
import glob
import cv2
import os
import shutil



def cleanImages(imagesFolder):
  
  for image in os.listdir(imagesFolder):
    
    fullImagePath=os.path.join(imagesFolder,image)
    image=cv2.imread(fullImagePath)
    
    if image is None:
      os.remove(fullImagePath)
      continue
    if image.size==0:
      os.remove(fullImagePath)
      continue
        

def Augment(photos_path):
  photosCount=len(glob.glob((photos_path + '/*.jpg')))
  photos=Augmentor.Pipeline(photos_path)
  
  photos.crop_random(probability=0.3,percentage_area=0.8,randomise_percentage_area=True)
  photos.flip_random(probability=0.5)
  photos.random_brightness(probability=0.4,min_factor=0.7,max_factor=1.3)
  photos.random_color(probability=0.3,min_factor=0.7,max_factor=1.3)
  photos.rotate_random_90(probability=0.2)
  photos.gaussian_distortion(probability=0.2,grid_width=4,grid_height=4,magnitude=8,corner='bell',method='in',mex=0.5,mey=0.5)
  
  photos.sample(int(max(600-photosCount,0.3*photosCount)))
  


def combineImages(originalFolder,generatedFolder):
  
    for image in os.listdir(generatedFolder):
      image_source_path = os.path.join(generatedFolder, image)
      image_destination_path = os.path.join(originalFolder, image)
      
      shutil.move(image_source_path,image_destination_path)
        
    os.rmdir(generatedFolder)
    
    
def  AugmentData(datasetMainFolder):
  
    for classFolder in os.listdir(datasetMainFolder):  
      fullClassPath=os.path.join(datasetMainFolder,classFolder)
      cleanImages(fullClassPath)
      Augment(fullClassPath)
      combineImages(fullClassPath,os.path.join(fullClassPath,'output'))
    

AugmentData(r'I:\4th year\first term\ML\Project\ML-Project\dataset')