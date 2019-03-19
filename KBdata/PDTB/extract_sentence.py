with open('train_pdtb.txt', 'r', encoding='utf-8') as f:
    list = f.readlines()

with open('sentence_entity.txt', 'w', encoding='utf-8') as f:
    for line in list:
        line_split = line.strip()
        line_list = line_split.split('|||')
        line_new = line_list[1] + '\n' + line_list[2] + '\n'
        f.write(line_new)

with open('train.txt', 'w', encoding='utf-8') as f:
    for line in list:
        line_split = line.strip()
        line_list = line_split.split('|||')
        line_new = line_list[1] + '\t' + line_list[2] + '\t' + line_list[0] + '\n'
        f.write(line_new)
