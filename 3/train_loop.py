from tqdm import tqdm
import torch

def train(model, train_loader, test_loader, num_epochs, 
    optimizer, criterion, logs_writer, device):
    for epoch in tqdm(range(num_epochs)):
        
        for i, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs_writer.add_scalar('Itearation Loss/train', loss, epoch*len(train_loader) + i)
        
        accuracy = calculate_accuracy(model, test_loader, device)
        # Print Loss
        logs_writer.add_scalar('Accuracy/train', accuracy, epoch)

def calculate_accuracy(model, test_loader, device):
    # Calculate Accuracy         
    correct = 0
    total = 0
    # Iterate through test dataset
    for images, labels in test_loader:
        # Load images
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass only to get logits/output
        outputs = model(images)

        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)

        # Total number of labels
        total += labels.size(0)

        # Total correct predictions
        correct += (predicted == labels).sum()

    accuracy = correct.cpu().numpy() / total
    return accuracy