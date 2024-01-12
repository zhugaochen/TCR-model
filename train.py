import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader
from model import Tudui
from data_preprocess import load_data
from model import TCRDataset

# 定义训练的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data, labels = load_data('../train1')
# data = np.array(data)
# data = torch.tensor(data, dtype=torch.float32)
# labels = np.array(labels)
# labels = labels.reshape(-1, 1)
# labels = torch.tensor(labels, dtype=torch.float32)
data_tensor = np.array(data)
data_tensor = torch.tensor(data_tensor, dtype=torch.float32)
print(data_tensor.shape)

labels = np.array(labels)
labels = labels.reshape(-1, 1)
labels = torch.tensor(labels, dtype=torch.float32)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_ids, test_ids) in enumerate(kf.split(data_tensor)):
    print(f"FOLD {fold}")
    print("--------------------------------")
    # 分割数据
    train_data = data_tensor[train_ids]
    test_data = data_tensor[test_ids]
    train_labels = labels[train_ids]
    test_labels = labels[test_ids]

    train_data_size = len(train_data)
    test_data_size = len(test_data)

    # 创建数据集
    train_dataset = TCRDataset(train_data, train_labels)
    test_dataset = TCRDataset(test_data, test_labels)

    # 创建 DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8)

    # 初始化模型
    tudui = Tudui().to(device)

    # 损失函数
    loss_fn = nn.BCELoss().to(device)

    # 优化器
    optimizer = torch.optim.Adam(tudui.parameters(), lr=1e-3)

    # 设置训练网络的一些参数
    # 记录训练的次数
    total_train_step = 0
    # 记录测试的次数
    total_test_step = 0
    # 训练的轮数
    epoch = 10

    # 初始化最佳准确率
    best_accuracy = 0.0

    # 训练和验证模型
    for i in range(epoch):
        print(f"Epoch {i + 1} / {epoch}")
        print("--------------------------")

        # 训练模型
        tudui.train()
        correct = 0
        total = 0
        total_train_loss = 0
        for data in train_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device).float()
            outputs = tudui(imgs)

            predicted = outputs.round()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            loss = loss_fn(outputs, targets)
            total_train_loss += loss.item()
            # 优化器优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step += 1
            if total_train_step % 10 == 0:
                print("训练次数: {}, Loss: {}".format(total_train_step, loss.item()))

        train_accuracy = correct / total
        print("第 {} 轮训练结束, 准确率: {:.2f}%".format(i + 1, train_accuracy * 100))

        all_outputs = []
        all_targets = []

        # 测试步骤开始
        tudui.eval()
        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for data in test_dataloader:
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device).float()
                outputs = tudui(imgs)
                outputs1 = torch.where(outputs < 0.5, torch.tensor(0.0).to(device), torch.tensor(1.0).to(device))
                loss = loss_fn(outputs, targets)
                total_test_loss += loss.item()
                total += targets.size(0)
                accuracy = (outputs1 == targets).sum()
                total_accuracy += accuracy

                # 保存预测和标签用于后续的AUC计算
                all_outputs.extend(outputs.sigmoid().cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # 计算当前epoch的准确率
        current_accuracy = total_accuracy / test_data_size
        current_auc = roc_auc_score(all_targets, all_outputs)
        print(f"Epoch {i + 1} test loss: {total_test_loss}, accuracy: {current_accuracy:.4f}, AUC: {current_auc:.4f}")

        print(current_accuracy)
        print(best_accuracy)
        # 如果当前epoch的准确率是最好的，保存模型
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            # 保存模型
            # torch.save(tudui.state_dict(), f'model_fold_{fold}_best.pth')
            print(f"Model saved at epoch {i + 1} with accuracy: {current_accuracy:.4f}")



