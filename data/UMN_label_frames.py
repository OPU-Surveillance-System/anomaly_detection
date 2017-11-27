from tqdm import tqdm

scenes = {
    '1_1': list(range(525, 614 + 1)),
    '1_2': list(range(1330, 1439 + 1)),
    '2_1': list(range(1806, 1985 + 1)),
    '2_2': list(range(2605, 2684 + 1)),
    '2_3': list(range(3219, 3428 + 1)),
    '2_4': list(range(3938, 4017 + 1)),
    '2_5': list(range(4807, 4928 + 1)),
    '2_6': list(range(5422, 5595 + 1)),
    '3_1': list(range(6195, 6235 + 1)),
    '3_2': list(range(6883, 6913 + 1)),
    '3_3': list(range(7700, 7719 + 1))
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
    'testset': ['1_2', '2_2', '2_3', '2_4', '2_5', '2_6', '3_2', '3_3']
}

for k in sets.keys():
    with open('umn_{}_labels'.format(k), 'w') as f:
        for s in tqdm(sets[k]):
            for i in range(frames[s][0], frames[s][1] + 1):
                label = 0
                if i in scenes[s]:
                    label = 1
                f.write('UMN_frames/frame_{}-{}\t{}\n'.format(s, i, label))
