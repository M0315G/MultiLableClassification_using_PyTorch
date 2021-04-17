import torch

from tqdm import tqdm

# training function
def train(model, dataloader, optimizer, criterion, train_data, device):
    
    print("Starting Training...")
    model.train()
    counter = 0
    train_running_loss = 0.0
    for i, data in tqdm( enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        counter += 1
        data, target = data['image'].to(device), data['label'].to(device)
        optimizer.zero_grad()
        outputs = model(data)
        
        # apply sigmoid activation to get all the outputs between 0 and 1
        outputs = torch.sigmoid(outputs)
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        
        # backpropogation
        loss.backward()
        # update optimizer parameters
        optimizer.step()
        
    train_loss = train_running_loss/counter
    
    return train_loss