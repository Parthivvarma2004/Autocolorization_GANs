import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pickle
import matplotlib.pyplot as plt
import cv2
import fastai
import io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# rf_loaded = pickle.load(open("model.h5","rb"), map_location = 'cpu')
rf_loaded =  CPU_Unpickler(open("model.h5","rb")).load()

# rf_loaded = torch.load('model.h5',map_location=torch.device('cpu'))

def transform_multiply(mul):
    def fn(arr):
        arr = arr * mul
        return arr
    return fn

def plot_image(img_batch, figsize=(8,3), cmap=None, title=None):
    if len(img_batch.shape)==3:
        img_batch = np.expand_dims(img_batch, axis=0)
    N = len(img_batch)
    fig = plt.figure(figsize=figsize)
    for i in range(N):
        img = img_batch[i]
#         img = np.transpose(img, [1,0,2])
        plt.subplot(1,N,i+1)
        plt.imshow(img, cmap=cmap)
    if title is not None:
        plt.title(f"{title}")
    plt.show()

# Function to preprocess the image for model input
def preprocess_image(image):
    grayscale_image = Image.fromarray(np.uint8(image))
    grayscale_image = grayscale_image.convert("L")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(grayscale_image).unsqueeze(0)
    return img_tensor.to(device)

rf_loaded.eval()

def autocolorize(image):
    preprocessed_img = preprocess_image(image)
    
    # Ensure model is also on the same device
    rf_loaded.to(device)
    
    # Perform model inference
    with torch.no_grad():
        colored_image = rf_loaded(preprocessed_img)
    
    
    # Move the output tensor to CPU for post-processing
    colored_image = colored_image.cpu().squeeze(0).numpy()
    colored_image = np.transpose(colored_image, (1, 2, 0))
    

    # Post-processing to convert the output to RGB
    preprocessed_img_cpu = preprocessed_img.cpu().squeeze(0).numpy()
    preprocessed_img_cpu = np.transpose(preprocessed_img_cpu, (1, 2, 0))

    colored_image = transform_multiply(255.0)(colored_image)
    
    preprocessed_img_cpu = transform_multiply(255.0)(preprocessed_img_cpu)
    
    lab_pred = np.concatenate((preprocessed_img_cpu, colored_image), axis=2)
    
    rgb_pred = cv2.cvtColor((lab_pred.astype("uint8")), cv2.COLOR_LAB2RGB)
    # print(lab_pred)
    
    # Plotting the resulting RGB image
    # plot_image(rgb_pred, figsize=(10,10), title="RGB Actual")
    data = Image.fromarray(rgb_pred) 
    return data

st.title('Black & White Image Autocolorization')

uploaded_file = st.file_uploader("Choose a black & white image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Original Image',  width=300)

    if st.button('Autocolorize'):
        # Perform autocolorization
        colored_image = autocolorize(image)
        st.image(colored_image, caption='Autocolorized Image',  width=300)