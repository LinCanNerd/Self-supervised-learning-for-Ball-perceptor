<<<<<<< HEAD
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.TripletModel import TripletBallPerceptor
from src.EdgeModel import EdgeDetection
from src.ColorizationModel import ColorizationModel
from src.TripletBallPatchesDataset import TripletBallPatchesDataset
from src.EdgeDataset import Edgedataset
from src.ColorizationDataset import ColorizationDataset
from src.BallPatchesDataset import BallPatchesDataset
from torchvision import transforms
from torch.utils.data import random_split
import tqdm


def train_edge_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):

    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm.tqdm(train_loader):
            # Load data
            grayscale_patches = batch['gray'].to(device)  # Grayscale patches
            target_edges = batch['edge'].to(device)  # Ground truth (edge map)

            # Forward pass
            outputs = model(grayscale_patches)
            optimizer.zero_grad()

            # Compute the loss
            loss = criterion(outputs, target_edges)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        train_loss /= len(train_loader)  

        # Switch model to evaluation mode
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in val_loader:
                # Load validation data
                grayscale_patches = batch['gray'].to(device)  # Grayscale patches
                target_edges = batch['edge'].to(device)  # Ground truth (edge map)
                # Forward pass for validation
                output = model(grayscale_patches)
                loss = criterion(output, target_edges)
                val_loss += loss.item()

        val_loss /= len(val_loader)    



        # Log training and validation results
        print(f'Epoch [{epoch+1}/{epochs}], '  # Display current epoch number
                f'Train Loss: {train_loss:.4f}, '  # Average training loss for this epoch
                f'Val Loss: {val_loss:.4f}')  # Average validation loss for this epoch

    return model





def train_triplet_loss(model, distance_function, train_loader, val_loader, epochs, optimizer=None, margin=1.0, device='cpu'):
    

    # Training step
    model.to(device)
    criterion = nn.TripletMarginWithDistanceLoss(margin=margin, distance_function=distance_function)

    for epoch in range(epochs):
        model.train() 
        train_loss = 0.0

        for batch in tqdm.tqdm(train_loader):
            anchor_patches = batch['anchor'].to(device)
            positive_patches = batch['positive'].to(device)
            negative_patches = batch['negative'].to(device)

            optimizer.zero_grad()

            anchor_embeddings = model(anchor_patches)
            positive_embeddings = model(positive_patches)
            negative_embeddings = model(negative_patches)

            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation step
        model.eval()  
        val_loss = 0.0

        with torch.no_grad():  
            for batch in val_loader:
                anchor_patches = batch['anchor'].to(device)
                positive_patches = batch['positive'].to(device)
                negative_patches = batch['negative'].to(device)

                anchor_embeddings = model(anchor_patches)
                positive_embeddings = model(positive_patches)
                negative_embeddings = model(negative_patches)

                loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)

                val_loss += loss.item()
                        # Save the model weights if validation loss improves

        val_loss /= len(val_loader)



        print(f'Epoch [{epoch+1}/{epochs}], '  # Display current epoch number
                f'Train Loss: {train_loss:.4f}, '  # Average training loss for this epoch
                f'Val Loss: {val_loss:.4f}')  # Average validation loss for this epoch    

     

    return model


def train_colorization_model(model, train_loader, val_loader, criterion, optimizer, device, epochs):

    model.to(device)

    # Training loop over the number of epochs
    for epoch in range(epochs):

        model.train()
        train_loss = 0.0 
        for batch in tqdm.tqdm(train_loader):  # Iterate through the training data loader
            grayscale_patch = batch['gray'].to(device)  
            color_patch = batch['original'].to(device)  
            
            # Forward pass: Pass the grayscale input through the model to get the predicted colorized image
            optimizer.zero_grad()  
            predicted_image = model(grayscale_patch)  # Perform forward pass and get predicted output

            # Compute the loss between the predicted image and the ground truth (colorized image)
            loss = criterion(predicted_image, color_patch)
            loss.backward() 
            optimizer.step()  

            train_loss += loss.item() 

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0  
        with torch.no_grad(): 
            for batch in val_loader:  
                grayscale_patch = batch['gray'].to(device)  
                color_patch = batch['original'].to(device)  

                predicted_image = model(grayscale_patch)  

                # Compute the loss for the validation batch
                loss = criterion(predicted_image, color_patch)
                val_loss += loss.item()  

        val_loss /= len(val_loader)


        print(f'Epoch [{epoch+1}/{epochs}], '  # Display current epoch number
                f'Train Loss: {train_loss:.4f}, '  # Average training loss for this epoch
                f'Val Loss: {val_loss:.4f}')  # Average validation loss for this epoch

    return model






def split_dataset(dataset):

    train_ratio = 0.8  # 80% for training, 20% for validation

    # Calculate sizes for train and validation datasets
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    batch_size = 5012

    # Perform the split
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader



annotations_path = 'spqr_dataset/raw/merged.csv'
images_path = 'spqr_dataset/images/'
epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),          # Convert image to PyTorch tensor
])

dataset = BallPatchesDataset(annotations_path, images_path, transform=transform) 

distance_function = lambda x, y: torch.sqrt(torch.sum((x - y) ** 2, dim=-1))

margin = 1.0

edge_model = EdgeDetection()
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(edge_model.parameters(), lr=0.001)
edgedataset = Edgedataset(dataset)
train_edge_loader, val_edge_loader = split_dataset(edgedataset)

print("\nTraining EdgeModel (1/3 epoch)...")
edge_model = train_edge_model(edge_model, train_edge_loader, val_edge_loader, criterion, optimizer, epochs//3, device)


print("\nLoading Feature Extractor on TripletModel...")
edge_weights = edge_model.state_dict()
feature_extractor_weights = {key: value for key, value in edge_weights.items() if "decoder" not in key}


triplet_model = TripletBallPerceptor()
triplet_model.load_state_dict(feature_extractor_weights, strict=False)
optimizer = optim.Adam(triplet_model.parameters(), lr=0.001)
triplet_dataset = TripletBallPatchesDataset(dataset, triplet_model, k=20, batch_size=256)
train_triplet_loader, val_triplet_loader = split_dataset(triplet_dataset)


print("\nTraining TripletModel (2/3 epoch)..")
triplet_model = train_triplet_loss(triplet_model, distance_function, train_triplet_loader, val_triplet_loader, epochs//3, optimizer, margin=margin, device = device)


print("\nLoading Feature Extractor on ColorizationModel...")
triplet_weights = triplet_model.state_dict()
feature_extractor_weights = {key: value for key, value in triplet_weights.items() if "decoder" not in key}


colorization_model = ColorizationModel()
colorization_model.load_state_dict(feature_extractor_weights, strict=False)

optimizer = optim.Adam(colorization_model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()
colorization_dataset = ColorizationDataset(dataset)
train_colorization_dataset, val_colorization_dataset =  split_dataset(colorization_dataset)

print("\nTraining ColorizationModel (3/3 epoch).")

colorization_model = train_colorization_model(colorization_model, train_colorization_dataset, val_colorization_dataset, criterion, optimizer, device, epochs = epochs//3)

print("\nLoading Weigths to a file")
torch.save(colorization_model.state_dict(), "Maml.pt")

=======
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.TripletModel import TripletBallPerceptor
from src.EdgeModel import EdgeDetection
from src.ColorizationModel import ColorizationModel
from src.TripletBallPatchesDataset import TripletBallPatchesDataset
from src.EdgeDataset import Edgedataset
from src.ColorizationDataset import ColorizationDataset
from src.BallPatchesDataset import BallPatchesDataset
from torchvision import transforms
from torch.utils.data import random_split
import tqdm


def train_edge_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
   

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm.tqdm(train_loader):
            # Load data
            grayscale_patches = batch['gray'].to(device)  # Grayscale patches
            target_edges = batch['edge'].to(device)  # Ground truth (edge map)

            # Forward pass
            outputs = model(grayscale_patches)
            optimizer.zero_grad()

            # Compute the loss
            loss = criterion(outputs, target_edges)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Switch model to evaluation mode
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in val_loader:
                # Load validation data
                grayscale_patches = batch['gray'].to(device)  # Grayscale patches
                target_edges = batch['edge'].to(device)  # Ground truth (edge map)
                # Forward pass for validation
                output = model(grayscale_patches)
                val_loss += criterion(output, target_edges)
            val_loss /= len(val_loader)
            
            # Save the model weights if validation loss improves
            if val_loss < best_val:
                torch.save(model.state_dict(), 'edgeTest.pt')
                best_val = val_loss
                print("Saved Weights")

        # Log training and validation results
        print(f'Epoch [{epoch+1}/{epochs}], '  # Display current epoch number
                f'Train Loss: {train_loss/len(train_loader):.4f}, '  # Average training loss for this epoch
                f'Val Loss: {val_loss/len(val_loader):.4f}')  # Average validation loss for this epoch

    return model





def train_triplet_loss(model, distance_function, train_loader, val_loader, epochs, optimizer=None, margin=1.0, device='cpu'):

    # Training step
    model.to(device)
    model.train() 
    criterion = nn.TripletMarginWithDistanceLoss(margin=margin, distance_function=distance_function)
    train_loss = 0.0

    for epoch in range(epochs):

        for batch in train_loader:
            anchor_patches = batch['anchor'].to(device)
            positive_patches = batch['positive'].to(device)
            negative_patches = batch['negative'].to(device)

            optimizer.zero_grad()

            anchor_embeddings = model(anchor_patches)
            positive_embeddings = model(positive_patches)
            negative_embeddings = model(negative_patches)

            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()


        # Validation step
        model.eval()  
        val_loss = 0.0

        with torch.no_grad():  
            for batch in val_loader:
                anchor_patches = batch['anchor'].to(device)
                positive_patches = batch['positive'].to(device)
                negative_patches = batch['negative'].to(device)

                anchor_embeddings = model(anchor_patches)
                positive_embeddings = model(positive_patches)
                negative_embeddings = model(negative_patches)

                loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)

                val_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], '  # Display current epoch number
                f'Train Loss: {train_loss/len(train_loader):.4f}, '  # Average training loss for this epoch
                f'Val Loss: {val_loss/len(val_loader):.4f}')  # Average validation loss for this epoch    

     

    return model


def train_colorization_model(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    best_val_loss = float('inf')  # Initialize the best validation loss to infinity to ensure the first validation loss will be saved

    # Training loop over the number of epochs
    for epoch in range(epochs):

        model.train()
        train_loss = 0.0 
        for batch in tqdm.tqdm(train_loader):  # Iterate through the training data loader
            grayscale_patch = batch['gray'].to(device)  
            color_patch = batch['original'].to(device)  
            
            # Forward pass: Pass the grayscale input through the model to get the predicted colorized image
            optimizer.zero_grad()  
            predicted_image = model(grayscale_patch)  # Perform forward pass and get predicted output

            # Compute the loss between the predicted image and the ground truth (colorized image)
            loss = criterion(predicted_image, color_patch)
            loss.backward() 
            optimizer.step()  

            train_loss += loss.item() 

        model.eval()
        val_loss = 0.0  
        with torch.no_grad(): 
            for batch in val_loader:  
                grayscale_patch = batch['gray'].to(device)  
                color_patch = batch['original'].to(device)  

                predicted_image = model(grayscale_patch)  

                # Compute the loss for the validation batch
                loss = criterion(predicted_image, color_patch)
                val_loss += loss.item()  

        # If the current validation loss is lower than the best validation loss, save the model's weights
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), 'weights/colorizationMSETest.pt')  # Save the model's weights to a file (change the name accordingly)
            best_val_loss = val_loss  # Update the best validation loss
            print("Saved Weights")  # Indicate that the model weights were saved due to a lower validation loss

        print(f'Epoch [{epoch+1}/{epochs}], '  # Display current epoch number
                f'Train Loss: {train_loss/len(train_loader):.4f}, '  # Average training loss for this epoch
                f'Val Loss: {val_loss/len(val_loader):.4f}')  # Average validation loss for this epoch

    return model






def split_dataset(dataset):

    train_ratio = 0.8  # 80% for training, 20% for validation

    # Calculate sizes for train and validation datasets
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    batch_size = 5012

    # Perform the split
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader



annotations_path = 'spqr_dataset/raw/merged.csv'
images_path = '/spqr_dataset/images/'
epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),          # Convert image to PyTorch tensor
])

dataset = BallPatchesDataset(annotations_path, images_path, transform=transform) 

distance_function = lambda x, y: torch.sqrt(torch.sum((x - y) ** 2, dim=-1))

margin = 1.0

edge_model = EdgeDetection()
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(edge_model.parameters(), lr=0.001)
edgedataset = Edgedataset(dataset)
train_edge_loader, val_edge_loader = split_dataset(edgedataset)

print("\nTraining EdgeModel (1/3 epoch)...")
edge_model = train_edge_model(edge_model, train_edge_loader, val_edge_loader, criterion, optimizer, epochs//3, device)


print("\Loading Feature Extractor on TripletModel...")
edge_weights = edge_model.state_dict()
feature_extractor_weights = {key: value for key, value in edge_weights.items() if "decoder" not in key}


triplet_model = TripletBallPerceptor(feature_extractor_weights)
optimizer = optim.Adam(triplet_model.parameters(), lr=0.001)
triplet_dataset = TripletBallPatchesDataset(dataset, triplet_model, k=20, batch_size=256)
train_triplet_loader, val_triplet_loader = split_dataset(triplet_dataset)


print("\\nTraining EdgeModel (2/3 epoch)..")
triplet_model = train_triplet_loss(triplet_model, distance_function, train_triplet_loader, val_triplet_loader, epochs//3, optimizer, margin=margin, device = device)


print("\Loading Feature Extractor on ColorizationModel...")
triplet_weights = triplet_model.state_dict()
feature_extractor_weights = {key: value for key, value in triplet_weights.items() if "decoder" not in key}


print("\nTraining of ColorizationModel...")
colorization_model = ColorizationModel(feature_extractor_weights)
optimizer = optim.Adam(colorization_model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()
colorization_dataset = ColorizationDataset(dataset)
train_colorization_dataset, val_colorization_dataset =  split_dataset(colorization_dataset)

colorization_model = train_colorization_model(colorization_model, train_colorization_dataset, val_colorization_dataset, criterion, optimizer, device, epochs = epochs//3)

print("Loading Weigths on a file")
torch.save(colorization_model.state_dict(), "Maml.pt")

>>>>>>> 0bbb47733b79819b5b1d92074f6970d78c740fdc
print("\nCompleted Training!")