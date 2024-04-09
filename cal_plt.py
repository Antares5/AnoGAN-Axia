import os
import re
import matplotlib.pyplot as plt

# 初始化数组
precision_values = []
recall_values = []
maa_values = []
f1_measure_values = []

# 获取当前工作目录
current_directory = os.getcwd()

# 遍历当前目录及其子目录中的所有文件
for root, dirs, files in os.walk(current_directory):
    for file in files:
        # 仅处理log文件
        if file.endswith("log_file.log"):
            file_path = os.path.join(root, file)

            # 打开文件并读取最后一行
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1]

                    # 使用正则表达式提取信息
                    match = re.search(
                        r'Precision: (\d+\.\d+), Recall: (\d+\.\d+), Majority Accuracy: (\d+\.\d+), F1_Measure: (\d+\.\d+)',
                        last_line)

                    if match:
                        # 提取匹配的值并添加到相应的数组中
                        precision_values.append(float(match.group(1)))
                        recall_values.append(float(match.group(2)))
                        maa_values.append(float(match.group(3)))
                        f1_measure_values.append(float(match.group(4)))

# 打印结果
print("Precision Values:", precision_values)
print("Recall Values:", recall_values)
print("Majority Accuracy Values:", maa_values)
print("F1 Measure Values:", f1_measure_values)


# 创建并保存 Precision 图表
plt.plot(precision_values, label='Precision', marker='o')
plt.title('Precision Evolution Over Log Files')
plt.xlabel('Log Files')
plt.ylabel('Precision Values')
plt.legend()
plt.savefig('pics/precision_plot.png')
plt.clf()  # 清除图表，以便创建下一个图表

# 创建并保存 Recall 图表
plt.plot(recall_values, label='Recall', marker='o')
plt.title('Recall Evolution Over Log Files')
plt.xlabel('Log Files')
plt.ylabel('Recall Values')
plt.legend()
plt.savefig('pics/recall_plot.png')
plt.clf()

# 创建并保存 Majority Accuracy 图表
plt.plot(maa_values, label='Majority Accuracy', marker='o')
plt.title('Majority Accuracy Evolution Over Log Files')
plt.xlabel('Log Files')
plt.ylabel('Majority Accuracy Values')
plt.legend()
plt.savefig('pics/maa_plot.png')
plt.clf()

# 创建并保存 F1 Measure 图表
plt.plot(f1_measure_values, label='F1 Measure', marker='o')
plt.title('F1 Measure Evolution Over Log Files')
plt.xlabel('Log Files')
plt.ylabel('F1 Measure Values')
plt.legend()
plt.savefig('pics/f1_measure_plot.png')
plt.clf()

# 绘制 Precision 曲线
plt.plot(precision_values, label='Precision', marker='o')
# 绘制 Recall 曲线
plt.plot(recall_values, label='Recall', marker='o')
# 绘制 Majority Accuracy 曲线
plt.plot(maa_values, label='Majority Accuracy', marker='o')
# 绘制 F1 Measure 曲线
plt.plot(f1_measure_values, label='F1 Measure', marker='o')

# 设置图表标题和轴标签
plt.title('Metrics Evolution Over Log Files')
plt.xlabel('Log Files')
plt.ylabel('Metric Values')

# 添加图例
plt.legend()
plt.savefig('pics/all_plot.png')
