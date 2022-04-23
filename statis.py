import xlrd
#打开excel
wb = xlrd.open_workbook('dataset_registry.xlsx')
#按工作簿定位工作表
sh = wb.sheet_by_name('MosMedData COVID19_1110')
print(sh.nrows)#有效数据行数
print(sh.ncols)#有效数据列数
labels = {'CT-0':0,'CT-1':0,'CT-2':0,'CT-3':0,'CT-4':0}

for i in range(sh.nrows):
    if i==0:
        continue
    labels[sh.cell(i,1).value] += 1

print(labels)