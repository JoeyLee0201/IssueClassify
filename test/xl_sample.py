# -*- coding: UTF-8 -*-
import xlrd
import xlwt

# 打开Excel文件读取数据
data = xlrd.open_workbook("./resource/temp1.xls")

# 获取一个工作表
table = data.sheets()[0]              # 通过索引顺序获取
table = data.sheet_by_index(0)        # 通过索引顺序获取
table = data.sheet_by_name(u'Sheet')  # 通过名称获取

# 获取整行和整列的值（数组）
i = 1
table.row_values(i)
table.col_values(i)

# 获取行数和列数
nrows = table.nrows
ncols = table.ncols

# 循环行列表数据
for i in range(nrows):
    print table.row_values(i)

# 单元格
cell_A1 = table.cell(0, 0).value
cell_C4 = table.cell(2, 3).value

# 使用行列索引
cell_A1 = table.row(0)[0].value
cell_A2 = table.col(1)[0].value


# 新建一个excel文件
f = xlwt.Workbook(encoding='utf-8')
# 新建一个sheet
table = f.add_sheet('info', cell_overwrite_ok=True)
# 写入数据table.write(行,列,value)
table.write(0, 0, 'test')
# 保存文件
f.save('./output/test.xls')
