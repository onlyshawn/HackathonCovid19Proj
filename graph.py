import matplotlib.pyplot as plt

# MCC F1_score precision recall kappa balanced_acc
ldam_reweight = [0.983,0.992,0.986,0.994,0.980,0.983]
ldam = [0.967,0.984,0.981,0.987,0.967,0.981]
focal_reweight = [0.976,0.989,0.985,0.994,0.975,0.980]
focal = [0.971,0.987,0.985,0.990,0.971,0.978]
cel_reweight = [0.963, 0.983, 0.973, 0.993, 0.963, 0.954]
cel = [0.959, 0.982, 0.975, 0.989, 0.959, 0.960]
baseline = [0.942, 0.975, 0.966, 0.971, 0.957, 0.960]

labels = ['MCC', 'F1_score', 'Precision', 'Recall', 'Kappa', 'Balanced_acc']

plt.rcParams['axes.labelsize'] = 16  # xy轴label的size
plt.rcParams['xtick.labelsize'] = 12  # x轴ticks的size
plt.rcParams['ytick.labelsize'] = 14  # y轴ticks的size

# 设置柱形的间隔
width = 0.1  # 柱形的宽度
x1_list = []
x2_list = []
x3_list= []
x4_list= []
x5_list= []
x6_list= []
x7_list= []
offset = 0
for i in range(len(ldam_reweight)):
    x1_list.append(i+offset)
    x2_list.append(i+width+offset)
    x3_list.append(i+2*width+offset)
    x4_list.append(i + 3 * width+offset)
    x5_list.append(i + 4 * width+offset)
    x6_list.append(i + 5 * width+offset)
    x7_list.append(i + 6 * width + offset)
# 创建图层
fig, ax1 = plt.subplots()
ax1.set_ylabel('Performance')
ax1.set_ylim(0.94, 1)
ax1.bar(x1_list,ldam_reweight, width=width, label="LDAM_Reweight",color='mediumspringgreen', align='edge')
ax1.bar(x2_list,ldam, width=width, label="LDAM",color='cornflowerblue', align='edge', tick_label=labels)
ax1.bar(x3_list,focal_reweight, width=width, label="Focal_Reweight",color='aqua', align='edge', tick_label=labels)
ax1.bar(x4_list,focal, width=width, label="Focal",color='mediumslateblue', align='edge', tick_label=labels)
ax1.bar(x5_list,cel_reweight, width=width, label="CEL_Reweight",color='lavender', align='edge', tick_label=labels)
ax1.bar(x6_list,cel, width=width, label="CEL",color='burlywood', align='edge', tick_label=labels)
ax1.bar(x7_list,baseline, width=width, label="Baseline",color='brown', align='edge', tick_label=labels)

ax1.legend(loc="upper right",prop={'size': 8})

# ax6.set_ylim(0, 0.3)
# ax6.set_ylabel('Y')
a_2 = [0,0,0.13]
b_2 = [0,0,0.18]
c_2 = [0,0,0.18]



plt.show()