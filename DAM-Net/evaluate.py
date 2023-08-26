import torch
from sklearn.metrics import accuracy_score, confusion_matrix
import torch.optim as optim

best_accuracy = 0.0
learning_rate = 0.0001
save_path = 'D:\OASIS\\Netmodel\\new_model.pth'



def evaluate_model(model, dataloader, flag=True):
    global best_accuracy
    model.eval()
    epoch_predictions = []
    epoch_labels = []
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    with torch.no_grad():
        for images, labels in dataloader:
            optimizer.zero_grad()
            if flag == False:
                main_outputs = model(images)
            else:
                main_outputs, _, _ = model(images)
            print("main_outputs.data", main_outputs.data)
            _, predicted_labels = torch.max(main_outputs.data, 1)

            label_map = {'CN': 0, 'AD': 1}
            # 将标签转换为数字编码
            numeric_labels = [label_map[label] for label in labels]
            # 将数字编码的标签转换为Tensor类型
            labels = torch.tensor(numeric_labels)

            epoch_predictions.extend(predicted_labels.cpu().numpy())
            epoch_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(epoch_labels, epoch_predictions)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        # 保存模型
        torch.save(model.state_dict(), save_path)
    print('epoch_labels:\n', epoch_labels)
    print('epoch_predictions:\n', epoch_predictions)
    # 计算混淆矩阵
    confusion = confusion_matrix(epoch_labels, epoch_predictions)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    print("Accuracy:", accuracy)
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
    print('best_accuracy:', best_accuracy)
    return accuracy, sensitivity, specificity
