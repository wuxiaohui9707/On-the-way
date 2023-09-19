import torch
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_truth_vs_prediction(model, loader,save_path):
    model.eval()

    truths = []
    predictions = []

    for batch in loader:
        inputs, labels = batch
        with torch.no_grad():# 关闭PyTorch的梯度计算功能
            outputs = model(inputs)
        truths.extend(labels.cpu().numpy())
        predictions.extend(outputs.cpu().numpy())

    truths = np.array(truths)
    predictions = np.array(predictions)

    plt.scatter(truths, predictions, alpha=1)
    plt.xlabel('Truths')
    plt.ylabel('Predictions')
    plt.title('Truths vs Predictions')
    plt.plot([truths.min(), truths.max()], [truths.min(), truths.max()], 'k--', lw=2)
    #file_path = os.path.join(save_path, "truth_vs_pred.png")
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path)  # 保存图片到指定目录
    plt.show()

