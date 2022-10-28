import nibabel as nib
import cv2
import numpy as np
from torchvision.transforms import functional as F
from torchvision import transforms
import torchvision.transforms
from skimage import io
from PIL import Image
import SimpleITK as sitk
def splitToPng(inputDir, outputDir):
    idx = 0
    maxn=0
    for i in range(20):
        image = nib.load(inputDir + f"myops_test_{201+i}_T2.nii.gz")
        img_affine = image.affine
        maxn=max(maxn,image.get_fdata().max())
        for j in range(image.shape[2]):
            
            single_synthesis_image = image.get_fdata()[..., j]
            single_synthesis_image=(single_synthesis_image-0.0)/(6506.0)
            single_synthesis_image*=255
            """
            pic = np.zeros((l.shape[0], l.shape[1], 3))
            pic[l == 1] = (51, 255, 51)
            pic[l == 2] = (178, 102, 255)
            pic[l == 3] = (102, 178, 255)
            pic[l == 4] = (255, 178, 102)
            pic[l == 5] = (255, 51, 51)
            cv2.imwrite(outputDir + f"{84+idx}.png", pic)
            """
            idx += 1
            #print(idx)
            PIL_image = Image.fromarray(single_synthesis_image)
            PIL_image = F.resize(PIL_image, [256,256], torchvision.transforms.InterpolationMode.NEAREST)
            img = np.array(PIL_image)
            rgbimg=np.stack(3 * [img], axis=2)
            io.imsave(outputDir+f"{idx}.png", rgbimg)
            #nib.Nifti1Image(single_synthesis_image,img_affine).to_filename( outputDir+f"{idx}.nii.gz")
            #nib.save(single_synthesis_image, outputDir + f"{idx}.nii.gz")

    return maxn
def recover_size(inputDir, outputDir,inputDir2, outputDir2):
    idx = 0
    for i in range(25):
        image = nib.load(inputDir + f"myops_training_{101+i}_C0.nii.gz")
        for j in range(image.shape[2]):
            # += 1
            single_synthesis_image = image.get_fdata()[..., j]
            img_size.append((single_synthesis_image.shape[0],single_synthesis_image.shape[1],image.shape[2]))
    
    i=1
    num=0
    #for i in range(1,73):
    
    while i<=102:
        nii_img = np.zeros([img_size[i-1][0], img_size[i-1][1], img_size[i-1][2]], dtype='float32')
        for j in range(img_size[i-1][2]):
            image2 = Image.open(inputDir2 + f"{i+j}_fake_B.png")
            #image2 = F.rotate(image2,90)
            PIL_image = F.resize(image2, [img_size[i+j-1][0],img_size[i+j-1][1]], torchvision.transforms.InterpolationMode.NEAREST)
            nii_img[:,:,j]=np.array(PIL_image, order='C')[:,:,0]
            #print(nii_img[:,:,j].shape)
            #nii_img[:,:,j]=nii_img[:,:,j].astype(np.float32)
            """
            nii_img[:,:,j].dtype=np.float32
            print(nii_img[:,:,j].dtype)
            nii_img[:,:,j]/=255.0
            nii_img[:,:,j]*=6506.0
            """
        #nii_img.dtype=np.float32
        #print(nii_img.dtype)
        nii_img/=255.0
        nii_img*=6506.0
        #nii_img.dtype=np.int16
        a = nib.load(inputDir + f"myops_training_{101+num}_C0.nii.gz")
        img_affine = a.affine
        new_image = nib.Nifti1Image(nii_img, img_affine.squeeze())
        
        num+=1
        nib.save(new_image, outputDir2+f"{num}.nii.gz")
        i+=img_size[i-1][2]
        #PIL_image.save(outputDir2+f"{i}.png")
    
if __name__ == "__main__":
    inputDir = "./train25/"
    outputDir = "./train25_C0/"
    inputDir2 = "D:/Medical-Transform/images/"
    outputDir2 = "D:/Medical-Transform/recover_images/"
    img_size=[]
    #maxn=splitToPng(inputDir, outputDir)
    #image2 = Image.open("/public/home/chenzheng/pytorch-CycleGAN-and-pix2pix/results/myops_pix2pix/test_latest/images/1_fake_B.png")
    recover_size(inputDir, outputDir,inputDir2,outputDir2)
    print(img_size)