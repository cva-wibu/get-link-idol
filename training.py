

from facenet_pytorch import MTCNN, training
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms, models
import numpy as np
import torch
import time
import os

dir = os.getcwd()

dir = os.path.join(dir, 'data')
list = os.listdir(dir)
num_classes = len(os.listdir(dir))

print(f'num_classes is {num_classes}')

# select gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print(f'device is {device}')



# use MTCNN to crop

batch_size = 15
epochs = 8
workers = 0 if os.name == 'nt' else 4

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.6, 0.7], factor=0.789, prewhiten=True,
    device=device)

dataset = datasets.ImageFolder(dir, transform=transforms.Resize((1024, 1024)))
dataset.samples = [(p, p.replace(dir, dir + '_cropped')) for p, _ in dataset.samples]

# crop image
loader = DataLoader(dataset, num_workers=workers, batch_size=16, collate_fn=training.collate_pil)

for i, (x, y) in enumerate(loader):
    print('\rImages processed: {:8d} of {:8d}'.format(i + 1, len(loader)), end='')
    mtcnn(x, save_path=y)
    print((x, y))

# Remove mtcnn to reduce GPU memory usage
del mtcnn



num_workers = 0

# load data and split
batch_size = 20

train_transforms = transforms.Compose([transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                                       transforms.RandomRotation(degrees=15),
                                       transforms.ColorJitter(),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.CenterCrop(size=224),  # Image net standards
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])
                                       ])

valid_size = 0.2
test_size = 0.1
train_size = 1 - valid_size - test_size

dataset = datasets.ImageFolder(dir + '_cropped', transform=train_transforms)
img_inds = np.arange(len(dataset))
np.random.shuffle(img_inds)

train_inds = img_inds[:int((train_size) * len(img_inds))]
val_inds = img_inds[int(test_size * len(img_inds)): int((test_size + valid_size) * len(img_inds))]
test_inds = img_inds[:int(test_size * len(img_inds))]

train_sampler = SubsetRandomSampler(train_inds)
valid_sampler = SubsetRandomSampler(val_inds)
test_sampler = SubsetRandomSampler(test_inds)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          sampler=test_sampler, num_workers=num_workers)



# load vgg for classification


model = models.vgg19(pretrained=True)

# Freeze training for all layers
for param in model.features.parameters():
    param.require_grad = False
# change last layer
num_features = model.classifier[6].in_features
model.classifier[6] = torch.nn.Linear(num_features, num_classes)
model.to(device)

# set hyperparameters
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.003)




# trainning
def train(model, train_loader, valid_loader, criterion, optimizer, num_epochs=10):
    valid_loss_min = np.Inf
    for epoch in range(num_epochs):

        start = time.time()

        # run on training set
        model.train()

        train_loss = 0.0
        valid_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0
        for inputs, labels in train_loader:

            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            train_acc += torch.mean(equals.type(torch.FloatTensor)).item()
        # train on validation set
        model.eval()

        with torch.no_grad():

            for inputs, labels in valid_loader:

                inputs, labels = inputs.to(device), labels.to(device)
                output = model.forward(inputs)
                batch_loss = criterion(output, labels)
                valid_loss += batch_loss.item()

                ps = torch.exp(output)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                val_acc += torch.mean(equals.type(torch.FloatTensor)).item()

        # calculate average losses and acc
        train_loss = train_loss / len(train_loader)
        valid_loss = valid_loss / len(valid_loader)
        train_accuracy = train_acc / len(train_loader)
        valid_accuracy = val_acc / len(valid_loader)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} '
              '\tTrain Accuracy: {:6f} \tValidation Accuracy: {:.6f}'.format(
            epoch + 1, train_loss, valid_loss, train_accuracy, valid_accuracy))

        # save model
        if valid_loss <= valid_loss_min:

            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            model_save_name = "get_idol.pt"
            path = os.path.join(dir, model_save_name)
            torch.save(model.state_dict(), path)
            valid_loss_min = valid_loss

        print(f"Time per epoch: {(time.time() - start):.3f} seconds")


train(model, train_loader, valid_loader, criterion, optimizer, num_epochs=15)



# load model
model.load_state_dict(torch.load(os.path.join(dir, 'get_idol.pt')))


def test(model, criterion, loader):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.
    for batch_idx, (data, target) in enumerate(loader):
        # move to GPU
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)

        print('Test Loss: {:.6f}\n'.format(test_loss))
    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))


test(model, criterion, test_loader)




