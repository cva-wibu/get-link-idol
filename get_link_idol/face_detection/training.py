import numpy as np
import time
import os
from datetime import date

import torch
from torchvision.models.vgg import vgg19
from facenet_pytorch import MTCNN

from get_link_idol.face_detection.data_manager import get_dataloader
from control import Config


def train(model,
          train_loader,
          valid_loader,
          criterion,
          optimizer,
          cfg,
          model_dir):
    valid_loss_min = np.Inf
    epoch_counter = 0
    for epoch in range(cfg.epoch):
        if epoch_counter > cfg.train_patience:
            print('Model not improve after {} epochs.'.format(cfg.train_patience))
            print('Training stop!')
            break

        start = time.time()

        # run on training set
        model.train()

        train_loss = 0.0
        valid_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0
        for inputs, labels in train_loader:
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)

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
                inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)
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
            torch.save(model.state_dict(), model_dir)
            valid_loss_min = valid_loss
        else:
            epoch_counter += 1

        print(f"Time per epoch: {(time.time() - start):.3f} seconds")


def test(model,
         criterion,
         loader,
         model_dir):
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


if __name__ == '__main__':
    cfg = Config()

    mtcnn = MTCNN(
        image_size=cfg.image_size,
        margin=cfg.margin,
        min_face_size=cfg.min_face_size,
        thresholds=cfg.threshold,
        factor=cfg.factor,
        prewhiten=cfg.prewhiten,
        device=cfg.device)

    train_loader, valid_loader, test_loader = get_dataloader(mtcnn,
                                                             batch_size=cfg.batch_size,
                                                             test_ratio=cfg.test_ratio,
                                                             valid_ratio=cfg.valid_ratio,
                                                             random_state=cfg.seed,
                                                             num_workers=cfg.num_workers,
                                                             pin_memory=cfg.pin_memory)

    del mtcnn

    model = vgg19(pretrained=True).to(cfg.device)

    num_features = model.classifier[6].in_features
    num_classes = train_loader.dataset.classes
    model.classifier[6] = torch.nn.Linear(num_features, num_classes)
    model.to(cfg.device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=cfg.lr)

    model_dir = os.path.join(os.getcwd(), 'models', date.today().strftime('%Y%m%d'))
    train(model, train_loader, valid_loader, criterion, optimizer, cfg, model_dir)
    test(model, criterion, test_loader, model_dir)
