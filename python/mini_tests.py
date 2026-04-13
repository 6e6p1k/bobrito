import bobr

a = bobr.arange(0, 10)
b = bobr.full((10,), 0.5)

c = a + b

c.eval()

print(c)
