import os
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import torchvision.models as models

# 定义数据集路径
dataset_path = 'data_sex'

# 定义数据预处理的转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 加载训练集数据
train_dataset = ImageFolder(os.path.join(dataset_path, 'train'), transform=transform)
# 加载测试集数据
test_dataset = ImageFolder(os.path.join(dataset_path, 'test'), transform=transform)

# 创建特征提取器模型
resnet = models.resnet101(pretrained=True)
resnet = resnet.eval()

# 提取训练集特征向量
train_features = []
train_labels = []

for image, label in train_dataset:
    feature_vector = resnet(image.unsqueeze(0)).detach().numpy().flatten()
    train_features.append(feature_vector)
    train_labels.append(label)

train_features = np.array(train_features)
train_labels = np.array(train_labels)

# 提取测试集特征向量
test_features = []
test_labels = []

for image, label in test_dataset:
    feature_vector = resnet(image.unsqueeze(0)).detach().numpy().flatten()
    test_features.append(feature_vector)
    test_labels.append(label)

test_features = np.array(test_features)
test_labels = np.array(test_labels)

# 创建贝叶斯分类器对象
classifier = GaussianNB(var_smoothing=0.3)

# 在训练集上训练分类器
classifier.fit(train_features, train_labels)

# 在训练集上进行预测
train_pred = classifier.predict(train_features)

# 计算训练集的准确率
train_accuracy = accuracy_score(train_labels, train_pred)
print("训练集准确率:", train_accuracy)

# 在测试集上进行预测
test_pred = classifier.predict(test_features)

# 计算测试集的准确率
test_accuracy = accuracy_score(test_labels, test_pred)
print("测试集准确率:", test_accuracy)
#####
scores = cross_val_score(classifier, test_features, test_pred, cv=5)  # 5折交叉验证

# 打印每折交叉验证的准确率
print("每折交叉验证的准确率：", scores)

# 打印平均准确率
print("平均准确率:", scores.mean())