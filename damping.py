import math

class SecondOrderDynamic:
    def __init__(self, f, z, r, x0):
        pi = math.pi
        self.k1 = z / (pi * f)
        self.k2 = 1 / ((2 * pi * f) * (2 * pi * f))
        self.k3 = r * z / (2 * pi * f)

        self.xp = x0
        self.y = x0
        self.yd = 0

    def Update(self, T, x, xd = None):
        if(xd == None):
            xd = (x - self.xp) / T
            self.xp = x

        self.y = self.y + T * self.yd
        self.yd = self.yd + T * (x + self.k3 * xd - self.y - self.k1 * self.yd) / self.k2
        return self.y