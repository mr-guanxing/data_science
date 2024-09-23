def bottom_up_max(number):
    dic_1 = {0: [1], 1: [1], 2: [2], 3: [3], 4: [4], 5: [2, 3]} 
    dic_2 = {0: 1, 1: 1, 2: 2, 3: 3, 4: 4, 5: 6} 
    if number <= 5:
        return dic_1[number], dic_2[number]    
    # 动态规划自底向上计算
    for i in range(6, number + 1):
        j_max, max_value = max(((j, dic_2[j] * dic_2[i - j]) for j in range(1, i)), key=lambda x: x[1])
        dic_1[i] = sorted(dic_1[j_max] + dic_1[i - j_max])
        dic_2[i] = max_value
        # 返回组合
    return dic_1[number], dic_2[number]

print(bottom_up_max(2023))


