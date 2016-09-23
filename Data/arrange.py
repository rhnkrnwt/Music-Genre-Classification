import csv
def get_data():
    with open('dataset.csv', 'r') as f:
        lines = csv.reader(f)
        dataset = list(lines)

    d = []
    for i in xrange(4):
        d.append([])
    for i, j in enumerate(dataset):
        if i < 1500:
            d[0].append(j)
        elif i < 3000:
            d[1].append(j)
        elif i < 4500:
            d[2].append(j)
        else:
           d[3].append(j)

    return d

