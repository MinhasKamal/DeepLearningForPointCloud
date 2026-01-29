import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

plt.rcParams["font.family"] = "Serif"
plt.figure(figsize=(16, 8))


data1 = [
    # model-name,    year,       OvAc, color,  sym, type,                      offsetX, offsetY
    # ["3DShapeNets",  2015+ 7/12, 77.0, '#f55', 'o', 'Convolutional Network', 0.0,  0.0],
    ["MVCNN",        2015+ 9/12, 90.1, '#f55', 'o', '_Convolutional Network',  0.0,  0.0],
    ["PointNet",     2017+ 4/12, 89.2, '#55f', '^', 'Multi-Layer Perceptron',  0.0,  0.0],
    ["PointNet++",   2017+ 6/12, 90.7, '#55f', '^', '_Multi-Layer Perceptron', 0.0,  0.0],
    ["PointCNN",     2018+11/12, 92.2, '#f55', 'o', '_Convolutional Network',  0.0,  0.0],
    ["PointConv",    2020+11/12, 92.5, '#f55', 'o', '_Convolutional Network',  0.0,  0.0],
    ["GeomGCNN",     2021+ 3/12, 95.9, '#5f5', 's', 'Graph Conv. Network',     0.0,  0.0],
    ["PointTrans.",  2021+ 9/12, 93.7, '#5ff', 'D', 'Transformer',            -1.9,  0.0],
    ["PointViewGCN", 2021+ 9/12, 95.4, '#5f5', 's', '_Graph Conv. Network',    0.0,  0.0],
    ["PointBERT",    2022+ 6/12, 93.8, '#5ff', 'D', '_Transformer',            0.0, -0.3],
    ["PointMLP",     2022+11/12, 94.5, '#55f', '^', '_Multi-Layer Perceptron',-1.6,  0.0],
    ["PointNext",    2022+10/12, 94.0, '#55f', '^', '_Multi-Layer Perceptron', 0.0,  0.0],
    ["ReCon",        2023+ 5/12, 94.7, '#5ff', 'D', '_Transformer',            0.0, -0.1],
    ["OmniVec",      2023+11/12, 96.6, '#5ff', 'D', '_Transformer',            0.0,  0.0],
    ["ReCon++",      2024+ 3/12, 95.0, '#5ff', 'D', '_Transformer',            0.0,  0.0]
]

plt.subplot(2, 2, 1)
plt.title("Classification on ModelNet40", fontsize=15, fontweight='normal')
plt.xlabel("Year", labelpad=10, loc="center", fontsize=12, fontweight='normal')
plt.ylabel("OA (%)", labelpad=10, loc="center", fontsize=12, fontweight='normal')
plt.xlim(2015, 2026)
plt.ylim(88, 97)
plt.grid(True)

# labels = []
for entry in data1:
  plt.plot(entry[1], entry[2], label=entry[5], color=entry[3], marker=entry[4], markersize=7)
  plt.annotate(entry[0], (entry[1]+entry[6]+0.13, entry[2]+entry[7]-0.16), fontsize=10)


####


data2 = [
    # model-name  ,  year,       OvAc,  color,  sym, type,                      offsetX, offsetY
    ["3DShapeNets",  2015+ 7/12, 00.00, '#f55', 'o', 'Convolutional Network',   0.0,  0.0],
    ["MVCNN",        2015+ 9/12, 00.00, '#f55', 'o', '_Convolutional Network',  0.0,  0.0],
    ["PointNet",     2017+ 4/12, 68.20, '#55f', '^', 'Multi-Layer Perceptron',  0.0,  0.0],
    ["PointNet++",   2017+ 6/12, 77.90, '#55f', '^', '_Multi-Layer Perceptron', 0.0, -0.6],
    ["PointCNN",     2018+11/12, 78.50, '#f55', 'o', '_Convolutional Network',  0.0,  0.0],
    ["PointConv",    2020+11/12, 00.00, '#f55', 'o', '_Convolutional Network',  0.0,  0.0],
    ["GeomGCNN",     2021+ 3/12, 00.00, '#5f5', 's', 'Graph Conv. Network',     0.0,  0.0],
    ["PointTrans.",  2021+ 9/12, 00.00, '#5ff', 'D', 'Transformer',             0.0,  0.0],
    ["PointViewGCN", 2021+ 9/12, 85.50, '#5f5', 's', '_Graph Conv. Network',   -1.9,  0.0],
    ["PointBERT",    2022+ 6/12, 83.07, '#5ff', 'D', '_Transformer',            0.0,  0.0],
    ["PointMLP",     2022+11/12, 85.70, '#55f', '^', '_Multi-Layer Perceptron', 0.0,  0.0],
    ["PointNext",    2022+10/12, 87.70, '#55f', '^', '_Multi-Layer Perceptron', 0.0,  0.0],
    ["ReCon",        2023+ 5/12, 91.26, '#5ff', 'D', '_Transformer',            0.0,  0.0],
    ["OmniVec",      2023+11/12, 96.10, '#5ff', 'D', '_Transformer',            0.0,  0.6],
    ["ReCon++",      2024+ 3/12, 95.25, '#5ff', 'D', '_Transformer',            0.0,  0.0]
]

plt.subplot(2, 2, 2)
plt.title("Classification on ScanObjectNN", fontsize=15, fontweight='normal')
plt.xlabel("Year", labelpad=10, loc="center", fontsize=12, fontweight='normal')
plt.ylabel("OA (%)", labelpad=10, loc="center", fontsize=12, fontweight='normal')
plt.xlim(2017, 2026)
plt.ylim(67, 98)
plt.grid(True)

for entry in data2:
  plt.plot(entry[1], entry[2], label=entry[5], color=entry[3], marker=entry[4], markersize=7)
  plt.annotate(entry[0], (entry[1]+entry[6]+0.13, entry[2]+entry[7]-0.5), fontsize=10)


####


data3 = [
    # model-name  ,  year,       mIoU,  color,  sym, type,                      offsetX, offsetY
    ["3DShapeNets",  2015+ 7/12, 00.00, '#f55', 'o', 'Convolutional Network',   0.0,  0.0],
    ["MVCNN",        2015+ 9/12, 00.00, '#f55', 'o', '_Convolutional Network',  0.0,  0.0],
    ["PointNet",     2017+ 4/12, 83.70, '#55f', '^', 'Multi-Layer Perceptron',  0.0,  0.0],
    ["PointNet++",   2017+ 6/12, 85.10, '#55f', '^', '_Multi-Layer Perceptron', 0.0,  0.0],
    ["PointCNN",     2018+11/12, 86.10, '#f55', 'o', '_Convolutional Network',  0.0,  0.0],
    ["PointConv",    2020+11/12, 85.70, '#f55', 'o', '_Convolutional Network',  0.0,  0.0],
    ["GeomGCNN",     2021+ 3/12, 89.10, '#5f5', 's', 'Graph Conv. Network',     0.0,  0.0],
    ["PointTrans.",  2021+ 9/12, 86.60, '#5ff', 'D', 'Transformer',             0.0,  0.0],
    ["PointViewGCN", 2021+ 9/12, 00.00, '#5f5', 's', '_Graph Conv. Network',    0.0,  0.0],
    ["PointBERT",    2022+ 6/12, 85.60, '#5ff', 'D', '_Transformer',            0.0,  0.0],
    ["PointMLP",     2022+11/12, 86.10, '#55f', '^', '_Multi-Layer Perceptron', 0.0, -0.1],
    ["PointNext",    2022+10/12, 87.10, '#55f', '^', '_Multi-Layer Perceptron', 0.0,  0.0],
    ["ReCon",        2023+ 5/12, 86.40, '#5ff', 'D', '_Transformer',            0.0,  0.0],
    ["OmniVec",      2023+11/12, 00.00, '#5ff', 'D', '_Transformer',            0.0,  0.0],
    ["ReCon++",      2024+ 3/12, 00.00, '#5ff', 'D', '_Transformer',            0.0,  0.0]
]

plt.subplot(2, 2, 3)
plt.title("Part Segmentation on ShapeNet-Part", fontsize=15, fontweight='normal', pad=10)
plt.xlabel("Year", labelpad=10, loc="center", fontsize=12, fontweight='normal')
plt.ylabel("mIoU (%)", labelpad=10, loc="center", fontsize=12, fontweight='normal')
plt.xlim(2017, 2026)
plt.ylim(83, 90)
plt.grid(True)

for entry in data3:
  plt.plot(entry[1], entry[2], label=entry[5], color=entry[3], marker=entry[4], markersize=7)
  plt.annotate(entry[0], (entry[1]+entry[6]+0.13, entry[2]+entry[7]-0.15), fontsize=10)

####

data4 = [
    # model-name  ,  year,       mIoU,  Color,  Sym, type,                      offsetX, offsetY
    ["3DShapeNets",  2015+ 7/12, 00.00, '#f55', 'o', 'Convolutional Network',   0.0,  0.0],
    ["MVCNN",        2015+ 9/12, 00.00, '#f55', 'o', '_Convolutional Network',  0.0,  0.0],
    ["PointNet",     2017+ 4/12, 47.60, '#55f', '^', 'Multi-Layer Perceptron',  0.0,  0.0],
    ["PointNet++",   2017+ 6/12, 54.50, '#55f', '^', '_Multi-Layer Perceptron', 0.0,  0.0],
    ["PointCNN",     2018+11/12, 65.40, '#f55', 'o', '_Convolutional Network',  0.0,  0.0],
    ["DGCNN",        2019+ 6/12, 56.10, '#5f5', 's', '_Graph Conv. Network',    0.0,  0.0],
    ["PointConv",    2020+11/12, 00.00, '#f55', 'o', '_Convolutional Network',  0.0,  0.0],
    ["GeomGCNN",     2021+ 3/12, 70.10, '#5f5', 's', 'Graph Conv. Network',     0.0,  0.0],
    ["PointTrans.",  2021+ 9/12, 73.50, '#5ff', 'D', 'Transformer',             0.0, -0.5],
    ["PointViewGCN", 2021+ 9/12, 00.00, '#5f5', 's', '_Graph Conv. Network',    0.0,  0.0],
    ["PointBERT",    2022+ 6/12, 00.00, '#5ff', 'D', '_Transformer',            0.0,  0.0],
    ["PointMLP",     2022+11/12, 00.00, '#55f', '^', '_Multi-Layer Perceptron', 0.0,  0.0],
    ["PointNext",    2022+10/12, 74.90, '#55f', '^', '_Multi-Layer Perceptron', 0.0, -0.5],
    ["ReCon",        2023+ 5/12, 00.00, '#5ff', 'D', '_Transformer',            0.0,  0.0],
    ["OmniVec",      2023+11/12, 75.90, '#5ff', 'D', '_Transformer',            0.0,  0.0],
    ["ReCon++",      2024+ 3/12, 00.00, '#5ff', 'D', '_Transformer',            0.0,  0.0],
    ["Sonata + PTV3",2025+ 3/12, 82.30, '#5ff', 'D', '_Transformer',           -2.0,  0.0]
]

plt.subplot(2, 2, 4)
plt.title("Semantic Segmentation on S3DIS", fontsize=15, fontweight='normal', pad=10)
plt.xlabel("Year", labelpad=10, loc="center", fontsize=12, fontweight='normal')
plt.ylabel("mIoU (%)", labelpad=10, loc="center", fontsize=12, fontweight='normal')
plt.xlim(2017, 2026)
plt.ylim(46, 84)
plt.grid(True)

for entry in data4:
  plt.plot(entry[1], entry[2], label=entry[5], color=entry[3], marker=entry[4], markersize=7)
  plt.annotate(entry[0], (entry[1]+entry[6]+0.13, entry[2]+entry[7]-0.7), fontsize=10)

####


plt.legend(loc='lower center', bbox_to_anchor=(-0.1, -0.5), title="Backbone Network Structures",
           labelspacing=1, shadow=False, ncol=4, fontsize=10, title_fontsize=12)

plt.subplots_adjust(hspace=0.4, wspace=0.2)
plt.savefig("my_plot.pdf", format="pdf", bbox_inches="tight")

