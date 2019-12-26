def func(x,y,z):
    def func2(x,y):
        return x*y
    return func2(x,z)

print(func(1,2,3))