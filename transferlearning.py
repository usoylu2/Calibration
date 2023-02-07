import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pyplot as plt
from netmodel import MNet_bce, MNetdense_bce
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import statistics
import time
from denoising import Denoising
from read_rf import read

Training_Visual = False
Test_Visual = False
Normalization_FLAG = True
Noise_FLAG = False
Freq_mismatch = False
Focus_mismatch = False
Power_mismatch = False
ResNet = True
DenseNet = False
PATH_base = './saved_models/base_resnet.pth'
PATH_fine = './saved_models/resnet_focus_tuned.pth'

Noise_low = 6
Noise_high = -60
learning_rate = [2e-6]*8
denoising_func = Denoising(low=Noise_low, high=Noise_high)
train_images = [2, 4, 6, 8, 10, 20, 40, 100]
valid_images = [2, 4, 6, 8, 10, 20, 40, 100]
test_images = [2, 4, 6, 8, 10, 20, 40, 100]
epochs = [20]*8
repetition = 10
Depth = 9
Batch_size = 128
Start_pixel = 540

# Get cpu or gpu device for training.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
m = nn.Sigmoid()


def extract_all_patchess(volume, num):
    indices = np.random.permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    depth, channels, frames = volume.shape

    if channels == 512:
        volume = volume[:, ::2, :]

    start_depth = Start_pixel
    patch_size = 200
    jump = 100

    patches = []
    depth_list = []
    flag = True

    frame_counter = 0
    depth_counter = 0

    while flag:
        for jj in range(9):
            patches.append(volume[start_depth + depth_counter * jump:start_depth + patch_size + depth_counter * jump,
                           10 + 26 * jj:36 + 26 * jj, frame_counter])
            depth_list.append(depth_counter)

        depth_counter += 1
        if depth_counter == Depth:
            frame_counter += 1
            depth_counter = 0
            # patches.pop()
            # depth_list.pop()

        if start_depth + patch_size + depth_counter * jump >= depth:
            frame_counter += 1
            depth_counter = 0
            patches.pop()
            depth_list.pop()

        if frame_counter == frames:
            flag = False

    return np.array(patches), np.array(depth_list)


def test_function(x_test, y_test, depth_test, mean_data, std_data, PATH_fine):

    # net = AlexNet(4).to(device)
    if ResNet:
        net = MNet_bce().to(device)
    elif DenseNet:
        net = MNetdense_bce().to(device)
    else:
        print("Network Error")
    #print(net)
    # parameter_number = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # print(f"Number of trainable parameters:{parameter_number}")

    net.load_state_dict(torch.load(PATH_fine))

    # mean = np.mean(x_test, axis=0)
    # # std = np.std(x_test, axis=0)
    if Normalization_FLAG:
        x_test = (x_test - mean_data) / std_data

    x_test_gpu = torch.from_numpy(x_test[:, np.newaxis, :, :]).float().to(device)
    y_test_gpu = torch.from_numpy(y_test).float().to(device)
    depth_test_gpu = torch.from_numpy(depth_test).to(device)

    dataset = TensorDataset(x_test_gpu, y_test_gpu, depth_test_gpu)
    test_loader = DataLoader(dataset, batch_size=Batch_size, pin_memory=False, shuffle=True)

    # prepare to count predictions for each class
    classes = ["phantom1", "phantom2"]
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    data_matrix = np.zeros((2, 2))
    depth_matrix = np.zeros((Depth, 3))
    net.eval()
    auc = []
    # again no gradients needed

    # filter_aug = Firwin_test()

    with torch.no_grad():
        for data in test_loader:
            inputs, labels, depth = data

            outputs = net(inputs)
            auc.append(roc_auc_score(labels.cpu().detach().numpy(), m(outputs)[:, 0].cpu().detach().numpy()))
            predictions = ((m(outputs) > 0.5) * 1)[:, 0]
            if predictions.shape[0] != labels.shape[0]:
                print("Error in label shape")
            # collect the correct predictions for each class
            for index in range(predictions.shape[0]):
                if int(labels[index]) == int(predictions[index]):
                    correct_pred[classes[int(labels[index])]] += 1
                    depth_matrix[int(depth[index]), 0] = depth_matrix[int(depth[index]), 0] + 1
                else:
                    depth_matrix[int(depth[index]), 1] = depth_matrix[int(depth[index]), 1] + 1
                total_pred[classes[int(labels[index])]] += 1
                if int(labels[index]) == 0:
                    data_matrix[0, int(predictions[index])] += 1
                elif int(labels[index]) == 1:
                    data_matrix[1, int(predictions[index])] += 1
                # elif label == 2:
                #     data_matrix[2, prediction] += 1
                # elif label == 3:
                #     data_matrix[3, prediction] += 1

    # print accuracy for each class
    total = 0
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        total = total + accuracy
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))

    print("Average Accuracy is: {:.1f} %".format(total / 2))
    print("AUC is : {:.1f} %".format(sum(auc)/len(auc)))
    if Test_Visual:
        print("Confusion Matrix:")
        print(data_matrix)
        plt.matshow(data_matrix)
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.show()

        depth_matrix[:, 2] = depth_matrix[:, 0]/(depth_matrix[:, 0]+depth_matrix[:, 1])
        depth_list = [1.02, 1.21, 1.4, 1.59, 1.78, 1.98, 2.17, 2.36, 2.55, 2.75, 2.94, 3.13]
        # depth_list = [1.02, 1.21, 1.4, 1.59, 1.78, 1.98, 2.17, 2.36, 2.55, 2.75, 2.94, 3.13, 3.32, 3.52]

        print("Depth Matrix:")
        print(depth_matrix)
        # plt.plot(depth_list, depth_matrix[:14, 2])
        plt.plot(depth_list, depth_matrix[:, 2])
        plt.title("Accuracy vs Patch Depth")
        plt.xlabel("Depth in cm")
        plt.ylabel("Classification Accuracy")
        plt.grid()
        plt.show()
    return total/2, sum(auc)/len(auc)


def train_function(x_train, x_valid, y_train, y_valid, depth_train, depth_valid, PATH_fine, PATH_base, epoch_num, LR):
    accuracies = []
    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # Calculate Mean
    mean_data = np.mean(x_train, axis=0)
    std_data = np.std(x_train, axis=0)

    # z-score normalization or standardization
    if Normalization_FLAG:
        x_train = (x_train-mean_data)/std_data
        x_valid = (x_valid-mean_data)/std_data

    x_train_gpu = torch.from_numpy(x_train[:, np.newaxis, :, :]).float().to(device)
    y_train_gpu = torch.from_numpy(y_train).float().to(device)
    depth_train_gpu = torch.from_numpy(depth_train).to(device)

    dataset = TensorDataset(x_train_gpu, y_train_gpu, depth_train_gpu)
    train_loader = DataLoader(dataset, batch_size=Batch_size, pin_memory=False, shuffle=True)

    x_valid_gpu = torch.from_numpy(x_valid[:, np.newaxis, :, :]).float().to(device)
    y_valid_gpu = torch.from_numpy(y_valid).float().to(device)
    depth_valid_gpu = torch.from_numpy(depth_valid).to(device)

    dataset = TensorDataset(x_valid_gpu, y_valid_gpu, depth_valid_gpu)
    valid_loader = DataLoader(dataset, batch_size=Batch_size, pin_memory=False, shuffle=True)

    # for X, y in train_loader:
    #     print("Shape of X [N, C, H, W]: ", X.shape, X.dtype)
    #     print("Shape of y: ", y.shape, y.dtype)
    #     break

    if ResNet:
        net = MNet_bce().to(device)
    elif DenseNet:
        net = MNetdense_bce().to(device)
    else:
        print("Network Error")
    net.load_state_dict(torch.load(PATH_base))
    # net = AlexNet(4).to(device)
    # print(net)
    parameter_number = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Number of trainable parameters:{parameter_number}")

    criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.8)
    optimizer = optim.Adam(net.parameters(), lr=LR)
    loss_epoch = []
    accuracies = []
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 40], gamma=0.5)
    hflipper = T.RandomHorizontalFlip(p=0.5)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epoch_num):# loop over the dataset multiple times
        start_time = time.time()

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, depth = data

            #Data Augmentation  https://pytorch.org/vision/stable/transforms.html
            inputs = hflipper(inputs)

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
                outputs = net(inputs)
                loss = criterion(outputs[:, 0], labels)

            # Scales the loss, and calls backward()
            # to create scaled gradients
            scaler.scale(loss).backward()

            # Unscales gradients and calls
            # or skips optimizer.step()
            scaler.step(optimizer)

            # Updates the scale for next iteration
            scaler.update()

            # print statistics
            # loss_epoch.append(loss.item())
            running_loss += loss.item()
        # scheduler.step()
        loss_epoch.append(running_loss)
        print('[%d] loss: %.3f'%(epoch + 1, running_loss))

        if (epoch+1) % 5 == 0:
            classes = ["phantom1", "phantom2"]
            correct_pred = {classname: 0 for classname in classes}
            total_pred = {classname: 0 for classname in classes}
            data_matrix = np.zeros((2, 2))
            net.eval()

            # again no gradients needed
            with torch.no_grad():
                for data in valid_loader:
                    inputs, labels, depth = data
                    outputs = net(inputs)
                    predictions = ((m(outputs) > 0.5)*1)[:, 0]
                    if predictions.shape[0] != labels.shape[0]:
                        print("Error in label shape")
                    # collect the correct predictions for each class
                    for index in range(predictions.shape[0]):
                        if int(labels[index]) == int(predictions[index]):
                            correct_pred[classes[int(labels[index])]] += 1
                        total_pred[classes[int(labels[index])]] += 1
                        if int(labels[index]) == 0:
                            data_matrix[0, int(predictions[index])] += 1
                        elif int(labels[index]) == 1:
                            data_matrix[1, int(predictions[index])] += 1
                        # elif label == 2:
                        #     data_matrix[2, prediction] += 1
                        # elif label == 3:
                        #     data_matrix[3, prediction] += 1
            # print accuracy for each class
            total = 0
            print("Epoch:")
            print(epoch)
            for classname, correct_count in correct_pred.items():
                accuracy = 100 * float(correct_count) / total_pred[classname]
                total = total + accuracy
                print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
            print("Average Accuracy is: {:.1f} %".format(total / 2))
            accuracies.append(total/2)
        print("Execution time is %s seconds" % (time.time() - start_time))
    print('Finished Training')

    if Training_Visual:
        plt.plot(loss_epoch)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid()
        plt.show()

        plt.plot(accuracies)
        plt.title("Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid()
        plt.show()
    torch.save(net.state_dict(), PATH_fine)

    return mean_data, std_data, accuracies


def read_training_files():

    if Freq_mismatch:
        fileName = '//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/TF_paper_data/freehand/phantomb/ufuk1.rf'
    elif Focus_mismatch:
        fileName = '//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/TF_paper_data/freehand/focus_specific/phantomb9MHz2cm.rf'
    elif Power_mismatch:
        fileName = '//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/TF_paper_data/freehand/power_specific/phantomb9MHz0db.rf'
    else:
        print("Error in mismatch type")
    volume1 = read(fileName)
    if Noise_FLAG:
        volume1 = denoising_func(volume1)

    if Freq_mismatch:
        fileName = '//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/TF_paper_data/freehand/liver_ph2/ufuk1.rf'
    elif Focus_mismatch:
        fileName = '//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/TF_paper_data/freehand/focus_specific/liver9MHz2cm.rf'
    elif Power_mismatch:
        fileName = '//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/TF_paper_data/freehand/power_specific/liver9MHz0db.rf'
    else:
        print("Error in mismatch type")
    volume2 = read(fileName)
    if Noise_FLAG:
        volume2 = denoising_func(volume2)

    return volume1, volume2


def train_split(vol1, vol2, num):
    num = int(num//2)

    class1, depth1 = extract_all_patchess(vol1, num)
    indices = np.random.permutation(class1.shape[0])
    class1 = class1[indices, :, :]
    depth1 = depth1[indices]

    class2, depth2 = extract_all_patchess(vol2, num)
    indices = np.random.permutation(class2.shape[0])
    class2 = class2[indices, :, :]
    depth2 = depth2[indices]

    x = np.concatenate((class1, class2), axis=0)
    y = np.concatenate((np.zeros(class1.shape[0]), np.zeros(class2.shape[0]) + 1))
    depth = np.concatenate((depth1, depth2), axis=0)

    x_train, x_test, y_train, y_test, depth_train, depth_test = train_test_split(x, y, depth,
                                                                                 test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test, depth_train, depth_test


def read_test_files():

    if Freq_mismatch:
        fileName = '//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/TF_paper_data/freehand/phantomb/ufuk9.rf'
    elif Focus_mismatch:
        fileName = '//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/TF_paper_data/freehand/focus_specific/phantomb9MHz1cm3cm.rf'
    elif Power_mismatch:
        fileName = '//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/TF_paper_data/freehand/power_specific/phantomb9MHz6db.rf'
    else:
        print("Error in mismatch type")
    volume1 = read(fileName)
    if Noise_FLAG:
        volume1 = denoising_func(volume1)

    if Freq_mismatch:
        fileName = '//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/TF_paper_data/freehand/liver_ph2/ufuk9.rf'
    elif Focus_mismatch:
        fileName = '//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/TF_paper_data/freehand/focus_specific/liver9MHz1cm3cm.rf'
    elif Power_mismatch:
        fileName = '//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/TF_paper_data/freehand/power_specific/liver9MHz6db.rf'
    else:
        print("Error in mismatch type")
    volume2 = read(fileName)
    if Noise_FLAG:
        volume2 = denoising_func(volume2)

    return volume1, volume2


def test_split(vol1, vol2, num):
    num = int(num//2)

    class1, depth1 = extract_all_patchess(vol1, num)
    indices = np.random.permutation(class1.shape[0])
    class1 = class1[indices, :, :]
    depth1 = depth1[indices]

    class2, depth2 = extract_all_patchess(vol2, num)
    indices = np.random.permutation(class2.shape[0])
    class2 = class2[indices, :, :]
    depth2 = depth2[indices]

    x = np.concatenate((class1, class2), axis=0)
    y = np.concatenate((np.zeros(class1.shape[0]), np.zeros(class2.shape[0]) + 1))
    depth = np.concatenate((depth1, depth2), axis=0)
    #
    # x_valid, x_test, y_valid, y_test, depth_train, depth_test = train_test_split(x, y, depth,
    #                                                                              test_size=0.5, random_state=42)
    return x, y, depth


if __name__ == '__main__':
    print("Using {} device".format(device))
    # train_vol1, train_vol2 = read_training_files()
    test_vol1, test_vol2 = read_test_files()
    print(test_vol1.shape, test_vol2.shape)
    for epoch, train_num, valid_num, test_num, LR in zip(epochs, train_images, valid_images, test_images, learning_rate):
        c1 = []
        c2 = []
        for i in range(repetition):
            print("Trial:")
            print(i+1)
            # x_train, x_val, y_train, y_val, depth_train, depth_val = train_split(train_vol1, train_vol2, train_num)
            x_train, y_train, depth_train = test_split(test_vol1[:, :, ::3], test_vol2[:, :, ::3], train_num)
            x_valid, y_valid, depth_valid = test_split(test_vol1[:, :, 1::3], test_vol2[:, :, 1::3], valid_num)
            x_test, y_test, depth_test = test_split(test_vol1[:, :, 2::3], test_vol2[:, :, 2::3], test_num)
            print(x_train.shape, x_valid.shape, y_train.shape, y_valid.shape)
            print(x_test.shape, y_test.shape)
            mean_data, std_data, acc = train_function(x_train, x_valid, y_train, y_valid, depth_train,
                                                      depth_valid, PATH_fine, PATH_base, epoch, LR)
            print("Testing:")
            print(x_test.shape, y_test.shape)
            temp, auc = test_function(x_test, y_test, depth_test, mean_data, std_data, PATH_fine)
            print(temp, auc)
            c1.append(temp)
            c2.append(auc)

        print("Results:")
        print("accuracy")
        print(c1)
        print(sum(c1)/len(c1))
        print(statistics.pstdev(c1))
        print("auc")
        print(c2)
        print(sum(c2)/len(c2))
        print(statistics.pstdev(c2))


