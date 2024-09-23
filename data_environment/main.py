        
def generate_custom_sequence_backtrack(digits, n):
    def backtrack(curr_sequence):
        nonlocal count
        if len(curr_sequence) == len(digits):
            count += 1
            if count == n:
                result.extend(curr_sequence)
                return
        else:
            for digit in digits:
                if digit not in curr_sequence:
                    curr_sequence.append(digit)
                    backtrack(curr_sequence)
                    curr_sequence.pop()

    result = []
    count = 0
    backtrack([])
    return int(''.join(map(str, result)))

# 输入要生成的个位数字序列和要查找的序列位置
input_digits = input("请输入要生成的个位数字序列（以空格分隔）：")
digits = [int(digit) for digit in input_digits.split()]
n = int(input("请输入要查找的序列位置："))

# 生成并输出序列
sequence = generate_custom_sequence_backtrack(digits, n)
if sequence:
    print(f"第{n}个序列是：{sequence}")
else:
    print("指定的位置超出范围，请输入一个更小的数字。")
