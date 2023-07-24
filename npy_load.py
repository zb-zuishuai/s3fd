import numpy as np

output = np.load("S3FD.npy", encoding="latin1", allow_pickle=True)  # 加载文件'ASCII', 'latin1', 'bytes'
with open("output.txt", "a") as f:  # 打开一个存储文件，并依次写入
    print(output, file=f)  # 将打印内容写入文件中

