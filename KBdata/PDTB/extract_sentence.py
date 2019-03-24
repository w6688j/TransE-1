import uuid

with open('train_pdtb.txt', 'r', encoding='utf-8') as f:
    list = f.readlines()

sentence_dict = {}
with open('sentence_entity.txt', 'w', encoding='utf-8') as f:
    for line in list:
        line_split = line.strip()
        line_list = line_split.split('|||')
        line_new = ''
        if line_list[1]:
            key1 = str(uuid.uuid4())
            sentence_dict[key1] = line_list[1].lower()
            line_new += key1 + '\t' + line_list[1].lower() + '\n'

        if line_list[2]:
            key2 = str(uuid.uuid4())
            sentence_dict[key2] = line_list[2].lower()
            line_new += key2 + '\t' + line_list[2].lower() + '\n'

        f.write(line_new)


def find_key(dict, value):
    for v in dict.items():
        if value == v[1]:
            return v[0]


with open('train.txt', 'w', encoding='utf-8') as f:
    for line in list:
        line_split = line.strip()
        line_list = line_split.split('|||')
        if line_list[0] and line_list[1]:
            key1 = find_key(sentence_dict, line_list[1].lower())
            key2 = find_key(sentence_dict, line_list[2].lower())
            line_new = key1 + '\t' + key2 + '\t' + line_list[0] + '\n'
            f.write(line_new)
