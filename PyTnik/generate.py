from random import randint

dots = 10
zlatnik_min = 1
zlatnik_max = 1000

for i in range(0, dots):
    x, y = randint(0, 1000), randint(0, 600)
    print(f"{x}, {y}", end='')
    for dist in range(0, i):
        print(', ', randint(zlatnik_min, zlatnik_max), end='', sep='')
    print('')