from casadi import *
# Symbolic representation
x = SX.sym('x')
y = SX.sym('y')
z = SX.sym('z')
f = x**2 + 100*z*2
g = z+(1 - x)**2 - y
P = dict(f = f, g = g, x = vertcat(x, y, z))

# Create solver instance 
F = nlpsol('F','ipopt',P)

# Solve the problem
r = F(x0 = [2.5, 3.0, 0.75], ubg = 0, lbg = 0)
print(r['x'])