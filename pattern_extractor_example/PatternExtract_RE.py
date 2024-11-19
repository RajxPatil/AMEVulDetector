import re
import os
import torch
import numpy as np
from MLP_layer import MLP

patterns = {"pattern1": 1, "pattern2": 2, "pattern3": 3}

patterns_flag = {"100", "010", "001"}

# split all functions of contracts
def split_function(filepath):
    function_list = []
    f = open(filepath, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()
    flag = -1

    for line in lines:
        text = line.strip()
        if len(text) > 0 and text != "\n":
            if text.split()[0] == "function" or text.split()[0] == "constructor":
                function_list.append([text])
                flag += 1
            elif len(function_list) > 0 and ("function" or "constructor" in function_list[flag][0]):
                function_list[flag].append(text)

    return function_list


# Position the call.value to generate the graph
def extract_pattern(filepath):
    allFunctionList = split_function(filepath)  # Store all functions
    callValueList = []  # Store all functions that call call.value
    otherFunctionList = []  # Store functions other than the functions that contain call.value
    pattern_list = []

    # Store functions other than W functions (with .call.value)
    for i in range(len(allFunctionList)):
        flag = 0
        for j in range(len(allFunctionList[i])):
            text = allFunctionList[i][j]
            if '.call.value' in text:
                callValueList.append(allFunctionList[i])
                flag += 1
        if flag == 0:
            otherFunctionList.append(allFunctionList[i])

    ################   pattern 1  #######################
    if len(callValueList) != 0:
        pattern_list.append(1)
    else:
        pattern_list.append(0)
        pattern_list.append(0)
        pattern_list.append(0)

    ################   pattern 2  #######################
    for i in range(len(callValueList)):
        CallValueFlag1 = 0

        if len(pattern_list) > 1:
            break

        for j in range(len(callValueList[i])):
            text = callValueList[i][j]
            if '.call.value' in text:
                CallValueFlag1 += 1
            elif CallValueFlag1 != 0:
                text = text.replace(" ", "")
                if "-" in text or "-=" in text or "=0" in text:
                    pattern_list.append(1)
                    break
                elif j + 1 == len(callValueList[i]) and len(pattern_list) == 1:
                    pattern_list.append(0)

    ################   pattern 3  #######################
    for i in range(len(callValueList)):
        CallValueFlag2 = 0
        param = None

        if len(pattern_list) > 2:
            break

        for j in range(len(callValueList[i])):
            text = callValueList[i][j]
            if '.call.value' in text:
                CallValueFlag2 += 1
                param = re.findall(r".call.value\((.+?)\)", text)[0]
            elif CallValueFlag2 != 0:
                if param in text:
                    pattern_list.append(1)
                    break
                elif j + 1 == len(callValueList[i]) and len(pattern_list) == 2:
                    pattern_list.append(0)

    return pattern_list


def extract_feature_with_fc(outputPathFC, pattern1, pattern2, pattern3):
    pattern1 = torch.Tensor(pattern1)
    pattern2 = torch.Tensor(pattern2)
    pattern3 = torch.Tensor(pattern3)
    model = MLP(4, 100, 250)

    pattern1FC = model(pattern1).detach().numpy().tolist()
    pattern2FC = model(pattern2).detach().numpy().tolist()
    pattern3FC = model(pattern3).detach().numpy().tolist()
    pattern_final = np.array([pattern1FC, pattern2FC, pattern3FC])

    np.savetxt(outputPathFC, pattern_final, fmt="%.6f")


if __name__ == "__main__":
    label_list = []  # List to store all labels
    inputFileDir = "../data_example/reentrancy/source_code/validation/"
    outputfeatureDir = "../pattern_feature/feature_zeropadding/reentrancy/validation/"
    outputfeatureFCDir = "../pattern_feature/feature_FNN/reentrancy/validation/"
    outputlabelDir = "../pattern_feature/label_by_extractor/reentrancy/validation/"
    dirs = os.listdir(inputFileDir)
    for file in dirs:
        pattern1 = [1, 0, 0]
        pattern2 = [0, 1, 0]
        pattern3 = [0, 0, 1]

        print(file)
        inputFilePath = inputFileDir + file
        name = file.split(".")[0]
        pattern_list = extract_pattern(inputFilePath)
        if len(pattern_list) == 3:
            if pattern_list[0] == 1:
                if pattern_list[1] == 1 and pattern_list[2] == 1:
                    label = 1
                else:
                    label = 0
            else:
                label = 0
        else:
            print("The extracted patterns are error!")

        pattern1.append(pattern_list[0])
        pattern2.append(pattern_list[1])
        pattern3.append(pattern_list[2])

        outputPathFC = outputfeatureFCDir + name + ".txt"
        extract_feature_with_fc(outputPathFC, pattern1, pattern2, pattern3)

        pattern1 = np.array(pattern1)
        pattern1 = np.array(np.pad(pattern1, (0, 246), 'constant'))
        pattern2 = np.array(pattern2)
        pattern2 = np.array(np.pad(pattern2, (0, 246), 'constant'))
        pattern3 = np.array(pattern3)
        pattern3 = np.array(np.pad(pattern3, (0, 246), 'constant'))

        pattern_final = np.array([pattern1, pattern2, pattern3])
        outputPath = outputfeatureDir + name + ".txt"
        np.savetxt(outputPath, pattern_final, fmt="%.6f")

        label_list.append(label)  # Add the label to the list

    # Now write all labels to a single txt file
    output_label_file = outputlabelDir + "validlabels.txt"
    with open(output_label_file, 'w') as f_outlabel:
        for label in label_list:
            f_outlabel.write(f"{label}\n")  # Write each label in a new line
