import numpy as np
import pandas as pd

def create_indicators(scores,top,nege_per):
    # 计算score的排名
    sorted_indices = np.argsort(scores[:, 1])[::-1]
    total_rows = len(scores)
    top_10_percent = int(total_rows * top)
    negative_percent = nege_per
    print("sorted_indices",sorted_indices)

    # 创建label向量
    label = np.zeros(total_rows, dtype=bool)
    label[sorted_indices[:top_10_percent]] = True

    # 创建negative向量
    negative = np.zeros(total_rows, dtype=bool)
    remaining_indices = sorted_indices[top_10_percent:]
    for i in range(5):
        start_index = int(len(remaining_indices) * (i / 5))
        end_index = int(len(remaining_indices) * ((i + 1) / 5))
        selected_indices = np.random.choice(remaining_indices[start_index:end_index],
                                             size=int(negative_percent * total_rows),
                                             replace=False)
        negative[selected_indices] = True

    # 创建train和test向量
    train = np.zeros(total_rows, dtype=bool)
    test = np.zeros(total_rows, dtype=bool)

    # 将label中等于true的行前80%赋予train，后20%赋予test
    label_true_indices = np.where(label)[0]
    label_true_split_index = int(len(label_true_indices) * 0.8)
    train[label_true_indices[:label_true_split_index]] = True
    test[label_true_indices[label_true_split_index:]] = True


    # 将negative中等于true的行前80%赋予train，后20%赋予test
    negative_true_indices = np.where(negative)[0]
    negative_true_split_index = int(len(negative_true_indices) * 0.8)
    train[negative_true_indices[:negative_true_split_index]] = True
    test[negative_true_indices[negative_true_split_index:]] = True

    return label, negative, train, test

# 示例向量
name = "lastfm_asia"
scores = np.array(pd.read_csv(f"E:\Documents\Python Scripts\DataSet\\{name}-graph_SIR.txt",sep=' ',header=None))
print(scores)
# 创建指示向量
top = 0.05
bucket = top * 2 / 5
label, negative, train, test = create_indicators(scores, top, bucket)

# 打印结果
print("Label:", np.where(label)[0],np.count_nonzero(label))
print("Negative:", np.where(negative)[0],np.count_nonzero(negative))
print("Train:", np.where(train)[0],np.count_nonzero(train))
print("Test:", np.where(test)[0],np.count_nonzero(test))

import pickle
path = f'E:\Documents\Python Scripts\SIR_add_exp\\{name}\\'
times = 5
# 保存为二进制文件
with open(path+name+f"-labeled_{times}.labeled_{times}", "wb") as f:
    pickle.dump(label, f)

with open(path+name+f"-train_mask_{times}.train_mask_{times}", "wb") as f:
    pickle.dump(train, f)

with open(path+name+f"-test_mask_{times}.test_mask_{times}", "wb") as f:
    pickle.dump(test, f)