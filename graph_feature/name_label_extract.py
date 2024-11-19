import json

f = open('./reentrancy/train.json', 'r', encoding='utf8')
outputName = open('./reentrancy/contract_name_train.txt', 'a')
outputLabel = open('./reentrancy/label_by_experts_train.txt', 'a')
json_data = json.load(f)
for d in json_data:
    print(d['filename'])
    outputName.write(d['filename'] + '\n')
    print(d['Reentrancy'])
    outputLabel.write(d['Reentrancy'] + '\n')
