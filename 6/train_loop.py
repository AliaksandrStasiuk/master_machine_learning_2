from tqdm import tqdm
import torch

def train(model, train_loader, test_loader, num_epochs, 
    optimizer, criterion, logs_writer):
    for epoch in tqdm(range(num_epochs)):
        
        for i, (images, labels) in enumerate(train_loader):
            # Load images
            images = images.requires_grad_()

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output
            outputs = model(images)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            logs_writer.add_scalar('Itearation Loss/train', loss, epoch*len(train_loader) + i)

        # Calculate Accuracy         
        correct = 0
        total = 0
        # Iterate through test dataset
        for images, labels in test_loader:
            # Load images
            images = images.requires_grad_()

            # Forward pass only to get logits/output
            outputs = model(images)

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)

            # Total number of labels
            total += labels.size(0)

            # Total correct predictions
            correct += (predicted == labels).sum()

        accuracy = 100 * correct / total

        # Print Loss
        logs_writer.add_scalar('Accuracy/train', accuracy, epoch)