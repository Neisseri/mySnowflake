def read_data_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = file.read().split('\n')
            data = [line.split(',') for line in data]
            # 将所有数据点合并到一个列表中
            data = [float(item) for sublist in data for item in sublist if item.strip()]
            return data
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到")
        return []

def calculate_relative_error(data1, data2):
    if len(data1) != len(data2):
        raise ValueError("两组数据长度不一致")

    relative_errors = []
    for i in range(len(data1)):
        if data1[i] != 0:
            relative_error = abs((data1[i] - data2[i]) / data1[i])
            relative_errors.append(relative_error)

    return relative_errors

def main():
    file1_path = "FP16_acc/result/500.csv"
    file2_path = "FP32/result/500.csv"

    data1 = read_data_from_file(file1_path)
    data2 = read_data_from_file(file2_path)

    if not data1 or not data2:
        return

    relative_errors = calculate_relative_error(data1, data2)

    if relative_errors:
        average_relative_error = sum(relative_errors) / len(relative_errors)
        print(f"两组数据的相对平均误差：{average_relative_error:.4f}")

if __name__ == "__main__":
    main()
