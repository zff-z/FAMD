import csv

file1 = "E:\\scientific_research\\android\\experiment\\Few-shot-chatGPT\\dataset\\with_suspiciousper_siamese.csv"
file2 = "E:\\scientific_research\\android\\experiment\\Few-shot-chatGPT\\dataset\\3_gram.csv"
output = "E:\\scientific_research\\android\\experiment\\Few-shot-chatGPT\\dataset\\with_opcode_siamese.csv"
# 读取第一个csv文件，获取hash值和family label
hash_dict = {}
with open(file1, 'r') as f1:
    reader = csv.reader(f1)
    for row in reader:
        hash_dict[row[1]] = row[-1]

# 遍历第二个csv文件，查找对应的hash值并将family label写入文件
with open(file2, 'r') as f2, open(output, 'w', newline='') as f_out:
    reader = csv.reader(f2)
    writer = csv.writer(f_out)
    for row in reader:
        if row[-1] in hash_dict:
            row.append(hash_dict[row[-1]])
        writer.writerow(row)