from math import dist


d = dict(
    first = 1,
    sec = 2,
    third = 3
)

a = [d['first'], d['sec'], d['third']]

print(d)
print(a)

d['first'] = 10

print(d)
print(a)