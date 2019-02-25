# -*- coding: UTF-8 -*-
import xlrd
import xlwt


def read(name, index):
    data = xlrd.open_workbook(name)
    table = data.sheet_by_index(index)

    # 获取行数和列数
    nrows = table.nrows
    ncols = table.ncols

    print "rows:", nrows
    print "cols:", ncols
    res = {}
    for i in range(nrows):
        res[table.row_values(i)[0]] = table.row_values(i)[1]
    del res['labels']
    return res


def read2(name, index):
    data = xlrd.open_workbook(name)
    table = data.sheet_by_index(index)

    # 获取行数和列数
    nrows = table.nrows
    ncols = table.ncols

    print "rows:", nrows
    print "cols:", ncols
    res = {}
    for i in range(nrows):
        if table.row_values(i)[0] in res:
            continue
        res[table.row_values(i)[0]] = {}
    for i in range(nrows):
        res[table.row_values(i)[0]][table.row_values(i)[1]] = table.row_values(i)[2]
    del res['labels']
    return res


def read_range(name, index=0):
    data = xlrd.open_workbook(name)
    table = data.sheet_by_index(index)

    # 获取行数和列数
    nrows = table.nrows
    ncols = table.ncols

    print "rows:", nrows
    print "cols:", ncols
    res = {}
    for i in range(nrows):
        res[table.row_values(i)[0]] = table.row_values(i)[1]
    del res['ID']
    return res


def write(labels, name, title1, title2):
    f = xlwt.Workbook()
    table = f.add_sheet('sheet', cell_overwrite_ok=True)
    # 写入数据table.write(行,列,value)
    table.write(0, 0, title1)
    table.write(0, 1, title2)
    i = 1
    for label, values in labels.items():
        table.write(i, 0, label)
        table.write(i, 1, values)
        i += 1
    f.save(name)


def write(labels, name, title1, title2, title3):
    f = xlwt.Workbook()
    table = f.add_sheet('Sheet1', cell_overwrite_ok=True)
    # 写入数据table.write(行,列,value)
    table.write(0, 0, title1)
    table.write(0, 1, title2)
    table.write(0, 2, title3)
    i = 1
    for label, values in labels.items():
        table.write(i, 0, label)
        table.write(i, 1, values[title2])
        table.write(i, 2, values[title3])
        i += 1
    f.save(name)


def write_temp2(labels, name):
    print labels
    f = xlwt.Workbook()
    table = f.add_sheet('sheet', cell_overwrite_ok=True)
    # 写入数据table.write(行,列,value)
    table.write(0, 0, 'labels')
    table.write(0, 1, 'contain_label')
    table.write(0, 2, 'num')
    i = 1
    for label, label_dict in labels.items():
        for contain_label, value in label_dict.items():
            table.write(i, 0, label)
            table.write(i, 1, contain_label)
            table.write(i, 2, value)
            i += 1
    table = f.add_sheet('sheet2', cell_overwrite_ok=True)
    table.write(0, 0, 'labels')
    table.write(0, 1, 'total_num')
    i = 1
    for label, label_dict in labels.items():
        total_num = 0
        for contain_label, value in label_dict.items():
            total_num += value
        table.write(i, 0, label)
        table.write(i, 1, total_num)
        i += 1
    f.save(name)



