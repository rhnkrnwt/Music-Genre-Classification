import csv
def get_data():
    num = 4
    with open('dataset.csv', 'r') as f:
        lines = csv.reader(f)
        dataset = list(lines)


    D = []
    for i in range(0,len(dataset),15):
        tmp1 = dataset[i:i+15]
        tmp2 = [k for l in tmp1 for k in l]
        D.append(tmp2)

    F = [[]]*num
    for i in range(num):
        F[i] = D[i*100:(i+1)*100]

    return F
