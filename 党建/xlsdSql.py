import xlrd
workbook = xlrd.open_workbook('./2015级获奖信息统计.xlsx')
booksheet = workbook.sheet_by_name('Sheet1')
rowstr = "";
helper =MySQLHelper("localhost", "root", "123456")
helper.setDB("partycreatestudy")
for row in range(booksheet.nrows):
    rowM=[]
    for col in range(booksheet.ncols):
        cel = booksheet.cell(row, col)
        val = cel.value
        rowstr = rowstr + "   " + str(val)
        row.append( str(val))
    print(rowstr)
    helper.insertsql(
        "insert into stuInfo(id,name,number,grade,dangyuan,date,reward) values(null,'" + str(row[0]) + "','" + str(
            123) + "','" + str(row[2]) + "','" + row[3] + "','" + row[4] + "','" + str(123) + "')")
    rowstr = "";

