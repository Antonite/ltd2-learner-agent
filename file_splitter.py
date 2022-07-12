from csv import reader

f = open('../ltd2-game-parser/data.csv', 'r')
csvfile = f.readlines()
header = csvfile[0]
filename = "data/{0}.csv"
n = 1

for i in range(1, len(csvfile)-1, 330000):
    tempCsv = open(filename.format(n), 'w+')
    tempCsv.write(header)
    tempCsv.writelines(csvfile[i:i+330000])
    n += 1
    tempCsv.close()

f.close()
