import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random

def get_random_patch(image, patch_size=32):

    height, width = image.shape[:2]
    
    # Generate random top-left corner for the patch
    y = np.random.randint(0, height - patch_size + 1)
    x = np.random.randint(0, width - patch_size + 1)

    # Extract the patch
    patch = image[y:y + patch_size, x:x + patch_size]

    return patch


def isPatchGreen( patch, edge_threshold = 0.01):
    # Ensure the patch is in uint8 format
    patch = patch.permute(1, 2, 0).numpy()
    patch = (patch * 255).clip(0, 255).astype("uint8")
    
    # Convert the patch to grayscale for edge detection
    gray_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray_patch, threshold1=50, threshold2=200)
    
    # Count the number of edge pixels
    edge_pixel_count = np.sum(edges > 0)
    total_pixels = patch.shape[0] * patch.shape[1]

    # If the proportion of edge pixels is below the threshold, return True
    return edge_pixel_count / total_pixels < edge_threshold


def display_patch_and_edges(patch):
    # Ensure the patch is in uint8 format
 
    patch2 = (patch * 255).clip(0, 255).astype("uint8")
    
    # Convert the patch to grayscale for edge detection
    gray_patch = cv2.cvtColor(patch2, cv2.COLOR_RGB2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray_patch, threshold1=150, threshold2=250)

    # Create a subplot to display both the original and edge-detected patches
    plt.figure(figsize=(5, 3))

    # Display the original patch
    plt.subplot(1, 2, 1)
    plt.imshow(patch)
    plt.title("Original Patch")
    plt.axis('off')

    # Display the edge-detected patch
    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title("Edge-detected Patch (Canny)")
    plt.axis('off')

    # Show the plots
    plt.tight_layout()
    plt.show()




def display_edge_prediction(model, sample,device):
    grayscale_patch = sample['gray'].unsqueeze(0).to(device)  # Aggiungi batch dimension
    target_edges = sample['edge'].unsqueeze(0).to(device)  # Ground truth

    # Imposta il modello in modalità di valutazione
    model.eval()

    # Passa la patch grigia attraverso il modello
    with torch.no_grad():
        predicted_edge = model(grayscale_patch)

    # Converti le uscite in formato numpy per il plot (se necessario)
    predicted_edge = predicted_edge.squeeze().cpu().numpy()  # Rimuovi batch e canale se presenti
    target_edges = target_edges.squeeze().cpu().numpy()  # Ground truth

    # Visualizza la patch grigia, il risultato predetto e la ground truth
    fig, axes = plt.subplots(1, 3, figsize=(10, 7))

    # Mostra la patch grigia
    axes[0].imshow(grayscale_patch.squeeze().cpu().numpy(), cmap='gray')
    axes[0].set_title('Grayscale Patch')

    # Mostra il risultato predetto
    axes[1].imshow(predicted_edge, cmap='gray')
    axes[1].set_title('Predicted Edge')

    # Mostra la ground truth
    axes[2].imshow(target_edges, cmap='gray')
    axes[2].set_title('Ground Truth')

    for ax in axes:
        ax.axis('off')

    plt.show()



def create_encoding_df(model, patches):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encodings = []
    model.eval()
    with torch.no_grad():
        for i in range(len(patches)):
            p = patches[i]
            a = p[1].to(device).unsqueeze(0) 
            label = p[0]
            a_enc = model(a)
            encodings.append([label] + list(a_enc.squeeze().cpu().detach().numpy()))
        
        # Convert to DataFrame
        encodings = pd.DataFrame(encodings)
        column_names = ['class'] + [f'x{i}' for i in range(1, encodings.shape[1])]
        encodings.columns = column_names
    return encodings


def total_patches(triplet_dataset):
    total_patches = []
    for p in triplet_dataset:
        total_patches.append((1, p['anchor']))
        total_patches.append((1, p['positive']))
        total_patches.append((0, p['negative']))
    return total_patches


def plot_classes(df):
    # Extract rows for class 1 and 0
    class_1 = df[df['class'] == 1]
    class_0 = df[df['class'] == 0]

    # Plotting class 1 (stars)
    plt.scatter(class_1['x1'], class_1['x2'], marker='*', color='blue', label='With Ball',alpha=0.5)

    # Plotting class 0 (squares)
    plt.scatter(class_0['x1'], class_0['x2'], marker='s', color='red', label='No Ball',alpha=0.5)

    # Adding labels, title, and legend
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Class Distribution')
    plt.legend()
    plt.grid(True)
    plt.show()



def apply_pca_to_dataframe(df, n_components=2):
    """
    Reduces the features in the DataFrame to n_components using PCA.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame with a 'class' column and feature columns.
    - n_components (int): Number of principal components to keep (default is 2).
    
    Returns:
    - pd.DataFrame: A DataFrame with 'class' and the top n_components principal components as columns.
    """
    # Separate features and class column
    class_col = df['class']
    features = df.drop(columns=['class'])
    
    # Standardize the features
    scaler = StandardScaler()
    #features = scaler.fit_transform(features)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(features)
    
    # Create a new DataFrame with the principal components
    pca_columns = [f'x{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(principal_components, columns=pca_columns)
    
    # Add the class column back
    pca_df['class'] = class_col.reset_index(drop=True)
    
    # Reorder columns: 'class', 'x1', 'x2', ...
    reordered_columns = ['class'] + pca_columns
    return pca_df[reordered_columns]


def display_colorization_prediction(model, colorization_dataset):


    # Randomly select an index to retrieve an image from the dataset
    index = random.randint(0, len(colorization_dataset))  
    # Load a random grayscale and color image pair from the dataset
    sample = colorization_dataset[index]
    grayscale_image = sample['gray']
    color_image = sample['original']

    # Convert the images to tensors
    grayscale_image_tensor = grayscale_image.unsqueeze(0)  # Add the batch dimension to the grayscale image.
    color_image_tensor = color_image.unsqueeze(0)  # Add the batch dimension to the color image.

    # Evaluate the image
    with torch.no_grad():  # Disable gradient computation during evaluation
        output_image = model(grayscale_image_tensor)  # Generate the colorized image using the model.

    # Convert tensors to numpy arrays for visualization
    grayscale_image_np = grayscale_image_tensor.squeeze().cpu().numpy()  # Convert the grayscale image to numpy.
    output_image_np = output_image.squeeze().cpu().numpy().transpose(1, 2, 0)  # Convert from CHW to HWC format for display.
    color_image_np = color_image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)  # Convert the ground truth image to HWC.

    # Visualize the images: Grayscale, Output, and Target
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  

    # Grayscale input image
    axs[0].imshow(grayscale_image_np, cmap='gray') 
    axs[0].set_title("Grayscale Patch")  
    axs[0].axis('off')  

    # Colorized output image
    axs[1].imshow(output_image_np)  
    axs[1].set_title("Colorized Patch")  
    axs[1].axis('off') 

    # Ground truth color image (target)
    axs[2].imshow(color_image_np)  
    axs[2].set_title("Target Patch (Real Colors)") 
    axs[2].axis('off') 

    plt.show() 
