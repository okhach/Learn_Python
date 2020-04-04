sum = 0
for n in range(1000000, 10000000):
    for i in range(2, n):
        if n % i == 0:
            break
    else:
        print(n)