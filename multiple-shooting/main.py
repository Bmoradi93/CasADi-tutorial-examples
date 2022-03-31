from casadi import *
# Symbolic representation

# system's states
x = SX.sym('x',2)

# Algebric variable
z = SX.sym('z')

# control action
u = SX.sym('u')

f = vertcat(z*x[0] - x[1] + u, x[0])

# algebric equation
g = x[1]**2 + z - 1
h = x[0]**2 + x[1]**2 + u**2
dae = dict(x = x, p = u, ode = f, z = z, alg = g, quad = h)

# Create solver instance
T = 10 # end time
N = 20 # prediction horizon
op = dict(t0 = 0, tf = T/N)
F = integrator('F', 'idas', dae, op)


# Starting from an empty NLP
w = []
lbw = []
ubw = []
G = []
J = 0

# initial conditions
Xk = MX.sym('X0', 2)
w += [Xk]
lbw += [0,1]
ubw += [0,1]

# Adding decision variables to each iteration over the prediction horizon
for k in range(1, N + 1):
    # Local control
    Uname = 'U' + str(k-1)
    Uk = MX.sym(Uname)
    w += [Uk]
    lbw += [-1]
    ubw += [1]

    # Call integrator
    Fk = F(x0 = Xk,p = Uk)
    J += Fk['qf']

    # New local state
    Xname = 'X' + str(k)
    Xk = MX.sym(Xname, 2)
    w += [Xk]
    lbw += [-25, -inf]
    ubw += [inf, inf]

    # Continuity constraints
    G += [Fk['xf'] - Xk]

# The NLP solver
nlp = dict(f = J, g = vertcat(*G), x = vertcat(*w))

# solution
S = nlpsol('S', 'blocksqp', nlp)

# solve nlp
r = S(lbx = lbw, ubx = ubw, x0 = 0, lbg = 0, ubg = 0)

print(r['x'])