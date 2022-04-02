import xlrd

wb = xlrd.open_workbook('dataset_registry.xlsx')

sh = wb.sheet_by_name('MosMedData COVID19_1110')
print(sh.nrows)
print(sh.ncols)
labels = {'CT-0':0,'CT-1':0,'CT-2':0,'CT-3':0,'CT-4':0}

for i in range(sh.nrows):
    if i==0:
        continue
    labels[sh.cell(i,1).value] += 1

print(labels)
