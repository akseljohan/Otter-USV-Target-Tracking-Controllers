"""
This is the 3-DOF Otter Model implemented in the Casadi-framework.
The matrix coefficients are collected directly from the Otter by Fossen 2021
"""
#TODO: remove the vehicle as an istance variable, and implement the matrecies directly in the model
# TODO Clean the casadi_otteR_model_3DOF file, to only contin the nessecary details for the 3DOF-model.
import casadi as ca
from casadi import *
import numpy as np
#import otter.otter as otter
#from python_vehicle_simulator.lib import gnc
#from casadi import tools as cat
#from python_vehicle_simulator.lib import gnc  # some functions from Fossen
#import matplotlib
#from matplotlib import pyplot as plt
import utils


# otter = otter.otter(0, 0, 0, 0)

class Casadi3dofOtterModel:
    def __init__(self, fossen_6_dof_otter_model):
        otter = fossen_6_dof_otter_model
        self.ode = None
        self.x = None
        self.u = None
        self.f = None

        # States
        # x1 = MX.sym('x1',3,1)  # [x,y,psi] position
        # x2 = MX.sym('x2',3,1)  # v velocity vector
        # x =[x1,x2]
        x = MX.sym('x', 6, 1)  # [x1, x2]  # state vector where x = [[x,y,psi].T , [u,v,r].T].T

        # controls
        u = MX.sym('u', 3, 1)  # Controls of the otter this is [t1+t2,0,-l1*t1-l2*t2]

        # symbolic representations
        # Rpsi = MX.sym('R(psi)', 3, 3)  # rotation matrix from BODY to NED (north-east-down)
        # p1 = MX.sym('p1', 1, 6)  # elimination matrix from the to extract yaw-rate
        # p2 = MX.sym('p2', 6, 3)  # elimination matrix to extract velocity matrix
        # M = MX.sym('M', 3,3) # mass matrix
        # Minv = MX.sym('Minv', 3, 3)  # inverse mass matrix M^-1
        # MRB = MX.sym('MRB', 3,3) # rigid body mass
        # MA = MX.sym('MA',3,3)   # Mass matrix added mass
        # D = num.sym('D', 3, 3)  # the Otter Damping matrix

        psi = MX.sym('psi')  # heading angle
        m_tot = MX.sym('m_tot')  # total mass of otter
        SO3 = MX.sym('SO3', 3, 3)  #
        r = MX.sym('r')  # angular velocity yaw = x[5]
        g_x = MX.sym('g_x')  # vector from the CO to CG chap. 3.3 in Fossen 2021
        Xh = SX.sym('Xh')
        Yh = SX.sym('Yh')
        Nh = SX.sym('Nh')
        Xu = SX.sym('Xh')
        Yv = SX.sym('Yv')
        Nr = SX.sym('Nr')
        nu_c = MX.sym('nu_c', 6, 1)  # symbol for currents
        nu_r = MX.sym('nu_r', 6, 1)  # relative velocity taking currents into considerations

        # Define constants:
        m_tot = otter.m_total
        MA = ca.MX(utils.convert_6DOFto3DOF(otter.MA))
        # print(f"MA:{MA}")
        MRB = ca.MX(utils.convert_6DOFto3DOF(otter.MRB))
        M = MA + MRB
        D = ca.MX(utils.convert_6DOFto3DOF(otter.D))  # "dette er "negativ D
        # print(f"D: {D}")
        Minv = ca.MX(np.linalg.inv(utils.convert_6DOFto3DOF(otter.M)))
        # nu_r = x - nu_c
        G = ca.MX((otter.G))  # we dont need this, due to the restoring forces only act in Have, roll and pitch
        g_x = otter.rg[0]  # x-component in the g-matrix
        # print(f"otter rg[]0: {g_x}")
        # functions
        CRB_v = MX.sym('C', 3, 3)  # Rigid body Corlious and centripetal matrix (absolut velocity in the body)
        CA_vr = MX.sym('CA_vr', 3, 3)  # added Coriolis and Centripetal matrix (relative velocity, due to currents)
        D_vr = MX.sym('D_vr', 3, 3)  # Linear surge damping + nonlinear yaw-damping

        # p1 = np.array([0, 0, 1, 0, 0, 0, 0])
        # p2 = np.array([[0, 0, 0, 1, 0, 0],
        #               [0, 0, 0, 0, 1, 0],
        #               [0, 0, 0, 0, 0, 1]])

        # Rotation Matrix function from BODY to NED
        b2ned = ca.vertcat(ca.horzcat(ca.cos(psi), -ca.sin(psi), 0),
                           ca.horzcat(ca.sin(psi), ca.cos(psi), 0),
                           ca.horzcat(0, 0, 1))

        # define function for R(psi)
        Rpsi = Function('R_psi', [psi], [b2ned])

        """
        Function to calculate rigid body and added mass Coriolis and centripetal matrices
         x[5] = yaw-rate and is the sith element in the state matrix
         CRB_v is - Quality controlled twoards eq. 6.99 in Fossen
        """
        CRB_v_matrix = ca.vertcat(ca.horzcat(0, -m_tot * x[5], -m_tot * g_x * x[5]),
                                  ca.horzcat(m_tot * x[5], 0, 0),
                                  ca.horzcat(m_tot * g_x * x[5], 0, 0))
        # CRB_v_matrix = Rpsi(x[2]).T @ CRB_v_matrix @ Rpsi(x[2]) #rotation about
        # defining function to calculate the CRB matrix
        CRB_v = Function('CRB_v', [x], [CRB_v_matrix])

        """
        CRB alternative from fossen:
        CRB_CG = np.zeros((6, 6))
                CRB_CG[0:3, 0:3] = self.m_total * Smtrx(nu[3:6])
                CRB_CG[3:6, 3:6] = -Smtrx(np.matmul(self.Ig, nu[3:6]))
                CRB = self.H_rg.T @ CRB_CG @ self.H_rg  # transform CRB from CG to CO(Body)
        
        CRB_CG = np.zeros((6, 6))
        CRB_CG[0:3, 0:3] = otter.m_total * gnc.Smtrx([0,0,x[5]])
        CRB_CG[3:6, 3:6] = -gnc.Smtrx(np.matmul(otter.Ig, [0,0,x[5]]))
        CRB = otter.H_rg.T @ CRB_CG @ otter.H_rg  # transform CRB from CG to CO(Body)
        CRB_v2 = Function('CRB_v', [x], [CRB])
        
        """

        # Matematically expression to calculate added mass Coriolis and centripetal matrices
        # 3-DOF model (surge, sway and yaw)
        # C = np.zeros( (3,3) )
        #        C[0,2] = -M[1,1] * nu[1] - M[1,2] * nu[2]
        #        C[1,2] =  M[0,0] * nu[0]

        """
        CA_vr assumes that the Munk moment (CA_vr[2,0]) in yaw can be neglected  and that CA[2,1] is zero (due to nonlinear-damping)
        x = [x (x-coord),y (y-coor),psi(heading),u(surge), v(sway), r (yaw rate)]
        CA_vr is - Quality controlled twoards eq. 6.99 in Fossen
        MA = [Xudot(0,0) 
                        Yvdot(1,1) 
                                Zwdot(2,2) 
                                        Kpdot(3,3) 
                                                Mqdot(4,4) 
                                                        Nrdot (5,5)]
        
        CA_vr_mtrx = ca.vertcat(ca.horzcat(0, 0, MA[1, 1] * x[4] + MA[1, 2] * x[5]),
                                ca.horzcat(0, 0, - 0 * MA[0, 0] * x[3]),
                                ca.horzcat((-MA[1, 1] * x[4] - MA[1, 2] * x[5]), (MA[0, 0] * x[3]), 0))"""

        CA_vr_mtrx = utils.m2c(MA, x[3:])
        CA_vr_mtrx[2, 0] = 0  # neglecting Munk moment
        CA_vr_mtrx[2, 1] = 0  # negleting som damping stuff tha need better damping terms in order to be used
        CA_vr = Function('CA_vr', [x], [CA_vr_mtrx])

        # Hydrodynamic linear(surge) damping + nonlinear yaw damping tau_damo[2,2]
        tau_damp = ca.mtimes(D, x[3:])
        tau_damp[2] = (D[2, 2] * x[5]) + (10 * D[2, 2] * ca.fabs(x[5]) * x[5])  # Same expression as in the Otter script
        """
        tau_damp = -np.matmul(self.D, nu_r)
        tau_damp[5] = tau_damp[5] - 10 * self.D[5, 5] * abs(nu_r[5]) * nu_r[5]
                """

        # print(tau_damp)
        Dv = Function('Dn_vr', [x], [tau_damp])

        """#Def symbols
        Yh = ca.SX.sym('Yh')
        Nh = ca.SX.sym('Nh')
        xL = ca.SX.sym('xL')
        Ucf = ca.SX.sym(('Ucf'))
        """

        # Cross-flow drag
        tau_cf_x = ca.Function('tau_cf', [x], [utils.crossFlowDrags3DOF(otter.L, otter.B, otter.T, x)])

        # define input variables:
        eta_0 = ca.SX([0, 0, 0])  # starter i x_n = 1, y_n = 1 psi = 0 (heading rett nord)
        x_0 = ca.MX([0, 0, 0, 0, 0, 0])  # initial states
        u_0 = ca.MX([0, 0, 0])  # initial controls
        nu_c_in = ca.MX([0, 0, 0, 0, 0, 0])  # velocity of current
        x_r = x - nu_c  # relative velocity vector
        Dv_val = ca.evalf(Dv(x_0))
        # print("Checking the ode equations")
        # print(f"val_MA: {MA}, type: {type(MA)}")
        # print(f"val_MRB: {MRB}, type: {type(MRB)}")
        # print(f"val_M: {M}, type: {type(M)}")
        # print(f"val_D: {D}, type: {type(D)}")
        # print(f"val_G: {G}, type: {type(G)}")
        # print(f"val_D(v): {ca.evalf(Dv([0, 0, 0, 10, 10, 10]))}")
        # print(f"CRB([0,0,0,0,0,0]): {ca.evalf(CRB_v([0, 0, 0, 0, 0, 0]))}")
        # print(f"CA(v): {ca.evalf(CA_vr(x_0))}")
        # print(f"C: {ca.evalf(CRB_v(x_0) + CA_vr(x_0))}")

        # Forward time integration method  -> Integrator to discretize the system using casadi integrators
        # x_dot --> x(k+1)

        sampleTime = 0.02
        N = 10000
        simTime = N * sampleTime
        integrator_options = {'tf': sampleTime,
                              'simplify': True,
                              'number_of_finite_elements': 4
                              }
        solver = 'rk'  # ipopt'

        # DAE problem structure
        # x = MX.sym('x')
        # u = MX.sym('u')
        p = MX.sym('p')

        damp_sway = ca.Function('damp_sway', [x], [ca.vertcat(0, -x[4] * 10, 0)])

        ode = ca.vertcat((Rpsi(x[2])) @ (x[3:]),
                         (Minv @ (u - (CRB_v(x) - CA_vr(x)) @ x[3:] - Dv(x) - tau_cf_x(x))))  # tau_cf_cx = crossflow

        f = ca.Function('f_1', [x, u], [ode], ['x', 'u'], ['xdot'])

        self.ode = ode
        self.x = x
        self.u = u
        self.f = f