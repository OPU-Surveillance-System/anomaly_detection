from tqdm import tqdm

scenes = {
    '1_1': list(range(482, 617 + 1)),
    '1_2': list(range(1294, 1441 + 1)),
    '2_1': list(range(1756, 1938 + 1)),
    '2_2': list(range(2554, 2679 + 1)),
    '2_3': list(range(3179, 3341 + 1)),
    '2_4': list(range(3907, 4012 + 1)),
    '2_5': list(range(4770, 4928 + 1)),
    '2_6': list(range(5385, 5522 + 1)),
    '3_1': list(range(6144, 6253 + 1)),
    '3_2': list(range(6820, 6917 + 1)),
    '3_3': list(range(7655, 7738 + 1))
}

frames = {
    '1_1': (0, 624),
    '1_2': (625, 1452),
    '2_1': (1453, 2001),
    '2_2': (2002, 2686),
    '2_3': (2687, 3454),
    '2_4': (3455, 4033),
    '2_5': (4034, 4928),
    '2_6': (4929, 5595),
    '3_1': (5596, 6253),
    '3_2': (6254, 6930),
    '3_3': (6931, 7738)
}

sets = {
    'trainset': ['1_1', '2_1', '3_1'],
    'valset': ['2_2', '3_2'],
    'testset': ['1_2', '2_2', '2_3', '2_4', '2_5', '2_6', '3_2', '3_3']
}

for k in sets.keys():
    with open('umn_{}_labels'.format(k), 'w') as f:
        for s in tqdm(sets[k]):
            for i in range(frames[s][0], frames[s][1] + 1):
                label = 0
                if i in scenes[s]:
                    label = 1
                f.write('UMN_frames/frame_{}_{}\t{}\n'.format(s, i, label))
