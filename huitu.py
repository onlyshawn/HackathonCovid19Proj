import matplotlib.pyplot as plt

import matplotlib.font_manager as fm #字体管理器




#准备数据

data = [254, 684, 125, 45, 2]
print(sum(data))
data = [d/sum(data) for d in data]

#准备标签

labels = ['CT-0','CT-1','CT-2','CT-3','CT-4']


#将横、纵坐标轴标准化处理,保证饼图是一个正圆,否则为椭圆

plt.axes(aspect='equal')


#控制X轴和Y轴的范围(用于控制饼图的圆心、半径)

plt.xlim(0,8)

plt.ylim(0,8)


#不显示边框

plt.gca().spines['right'].set_color('none')

plt.gca().spines['top'].set_color('none')

plt.gca().spines['left'].set_color('none')

plt.gca().spines['bottom'].set_color('none')

colors = ['tomato', 'lightskyblue', 'goldenrod', 'green', 'y']
#绘制饼图

plt.pie(x=data, #绘制数据

labels=labels,#添加编程语言标签

colors = colors,

autopct='%.3f%%',#设置百分比的格式,保留3位小数

shadow=False,

pctdistance=0.5, #设置百分比标签和圆心的距离

labeldistance=0.9,#设置标签和圆心的距离

startangle=180,#设置饼图的初始角度

center=(4,4),#设置饼图的圆心(相当于X轴和Y轴的范围)

radius=3.8,#设置饼图的半径(相当于X轴和Y轴的范围)

counterclock= False,#是否为逆时针方向,False表示顺时针方向

wedgeprops= {'linewidth':1,'edgecolor':'green'},#设置饼图内外边界的属性值

textprops= {'fontsize':12,'color':'black'},#设置文本标签的属性值

frame=1) #是否显示饼图的圆圈,1为显示



#不显示X轴、Y轴的刻度值

plt.xticks(())

plt.yticks(())


#添加图形标题

plt.title('Dataset Distribution')

#显示图形

plt.savefig("./huitu.jpg")