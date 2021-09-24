import json
import numpy as np

def color_map(img):
    f = open('dataset/config.json')
    data = json.load(f)
    # rgb_frame = np.random.randint(low=0, high=66, size=(160, 240))
    a = np.zeros(shape=(160,240, 3))
    for i in range(160):
        for j in range(240):
            a[i,j] = data['labels'][img[i][j]]['color']
    f.close()
    return a