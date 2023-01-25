import casadi as ca
from casadi import *
import numpy as np

# States
# x1 = MX.sym('x1',3,1)  # [x,y,psi] position
# x2 = MX.sym('x2',3,1)  # [u,v,r] velocity vector
x = MX.sym('x', 6, 1)  # [x1, x2]  # state vector
# x =[x1,x2]
# controls
u = MX.sym('u', 3, 1)  # Controls of the otter this is [t1+t2,0,-l1*t1-l2*t2]

# constants
# Rpsi = MX.sym('R(psi)', 3, 3)  # rotation matrix from BODY to NED (north-east-down)
p1 = MX.sym('p1', 1, 6)  # elimination matrix from the to extract yaw-rate
p2 = MX.sym('p2', 6, 3)  # elimination matrix to extract velocity matrix
Minv = MX.sym('Minv', 3, 3)  # inverse mass matrix M^-1
D = MX.sym('D', 3, 3)  # the Otter Damping matrix
C = MX.sym('C', 3, 3)  # corlious matrix C^-1
psi = SX.sym('psi')

# p1 = np.array([0, 0, 1, 0, 0, 0, 0])
# p2 = np.array([[0, 0, 0, 1, 0, 0],
#               [0, 0, 0, 0, 1, 0],
#               [0, 0, 0, 0, 0, 1]])

print(f"x shape: {x.shape}")
print(f"u shape: {u.shape}")

# Rotation Matrix function from BODY to NED
b2ned = ca.vertcat(ca.horzcat(ca.cos(psi), -ca.sin(psi), 0),
                      ca.horzcat(ca.sin(psi), ca.cos(psi), 0),
                      ca.horzcat(0, 0, 1))

Rpsi= Function('R_psi', [psi], [b2ned])

# Function to calculate rigid body and added mass Coriolis and centripetal matrices
        # CRB_CG = [ (m+mp) * Smtrx(nu2)          O3
        #              O3                   -Smtrx(Ig*nu2)  ]
#              (Fossen 2021, Chapter 6)
 CRB_CG = np.zeros((6, 6))
 CRB_CG[0:3, 0:3] = self.m_total * Smtrx(nu[3:6])
 CRB_CG[3:6, 3:6] = -Smtrx(np.matmul(self.Ig, nu[3:6]))
 CRB = self.H_rg.T @ CRB_CG @ self.H_rg  # transform CRB from CG to CO

 CA = m2c(self.MA, nu_r)
 CA[5, 0] = 0  # assume that the Munk moment in yaw can be neglected
 CA[5, 1] = 0  # if nonzero, must be balanced by adding nonlinear damping

 C = CRB + CA
 # skrive denne funksjonen: C(x[:3])


print(x[2].shape)
# Fossen state space ODE p.157
ode = [(Rpsi(x[2])) @ (x[:3]),
       ]
Minv @ (-C(x[:3]) @ (x[:3]) - D @ (x[:3]) + u)

#f = Function('f', {x, u}, {ode}, {'x', 'u'}, {'ode'})
# f([0.2,0.8],0.1)
