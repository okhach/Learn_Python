import pdb

def fib(max):
    a = 0 
    b = 0
    x = 1
    i = 1
    while i <= max:
        pdb.set_trace()
        yield x
        a = b 
        b = x
        x = a + b
        i = i + 1
    return 'done'


for n in fib(20):
    print(n)
