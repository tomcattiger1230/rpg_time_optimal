from casadi import MX, DM, SX,vertcat, mtimes, Function, inv, cross, sqrt, norm_2
import yaml
from quaternion import *
import casadi as ca
import numpy as np


class Quad:
    def __init__(self, filename=""):
        self.m = 1  # mass in [kg]
        self.l = 1  # arm length
        self.I = DM([(1, 0, 0), (0, 1, 0), (0, 0, 1)])  # Inertia
        self.I_inv = inv(self.I)  # Inertia inverse
        self.T_max = 5  # max thrust [N]
        self.T_min = 0  # min thrust [N]
        self.omega_max = 3  # max bodyrate [rad/s]
        self.ctau = 0.5  # thrust torque coeff.
        self.rampup_dist = 0
        self.T_ramp_start = 5
        self.omega_ramp_start = 3

        self.v_max = None
        self.cd = 0.0

        self.g = 9.801

        if filename:
            self.load(filename)

    def load(self, filename):
        print("Loading track from " + filename)
        with open(filename, 'r') as file:
            quad = yaml.load(file, Loader=yaml.FullLoader)

        if 'mass' in quad:
            self.m = quad['mass']
        else:
            print("No mass specified in " + filename)

        if 'arm_length' in quad:
            self.l = quad['arm_length']
        else:
            print("No arm length specified in " + filename)

        if 'inertia' in quad:
            self.I = DM(quad['inertia'])
            self.I_inv = inv(self.I)
        else:
            print("No inertia specified in " + filename)

        if 'TWR_max' in quad:
            self.T_max = quad['TWR_max'] * 9.81 * self.m / 4
        elif 'thrust_max' in quad:
            self.T_max = quad['thrust_max']
        else:
            print("No max thrust specified in " + filename)

        if 'TWR_min' in quad:
            self.T_min = quad['TWR_min'] * 9.81 * self.m / 4
        elif 'thrust_min' in quad:
            self.T_min = quad['thrust_min']
        else:
            print("No min thrust specified in " + filename)

        if 'omega_max_xy' in quad:
            self.omega_max_xy = quad['omega_max_xy']
        else:
            print("No max omega_xy specified in " + filename)

        if 'omega_max_z' in quad:
            self.omega_max_z = quad['omega_max_z']
        else:
            print("No max omega_z specified in " + filename)

        if 'torque_coeff' in quad:
            self.ctau = quad['torque_coeff']
        else:
            print("No thrust to drag coefficient specified in " + filename)

        if 'v_max' in quad:
            self.v_max = quad['v_max']
            a_max = 4 * self.T_max / self.m
            a_hmax = sqrt(a_max**2 - self.g**2)
            self.cd = a_hmax / self.v_max
        if 'drag_coeff' in quad:
            self.cd = quad['drag_coeff']

        if 'rampup_dist' in quad:
            self.rampup_dist = quad['rampup_dist']
            if 'TWR_ramp_start' in quad and 'omega_ramp_start' in quad:
                self.T_ramp_start = min(
                    quad['TWR_ramp_start'] * 9.81 * self.m / 4, self.T_max)
                self.omega_ramp_start = min(quad['omega_ramp_start'],
                                            self.omega_max_xy)
            else:
                print(
                    "No TWR_ramp_start or omega_ramp_start specified. Disabling rampup"
                )
                rampup_dist = 0

    def get_Lagrangian_casadi(self):
        # set symbolic parameters using casadi
        x = SX.sym('x')
        y = SX.sym('y')
        z = SX.sym('z')
        phi = SX.sym('phi')
        theta = SX.sym('theta')
        psi = SX.sym('psi')
        states_2 = ca.vertcat(x, y, z, phi, theta, psi)
        p = ca.vertcat(x, y, z)
        # angle = ca.vertcat(phi, theta, psi)

        # set states_dot
        d_x = SX.sym('dx')
        d_y = SX.sym('dy')
        d_z = SX.sym('dz')
        d_phi = SX.sym('dphi')
        d_theta = SX.sym('dtheta')
        d_psi = SX.sym('dpsi')
        d_states = ca.vertcat(d_x, d_y, d_z, d_phi, d_theta, d_psi)
        states = ca.vertcat(states_2, d_states)
        p_dot = ca.vertcat(d_x, d_y, d_z)

        Rz = np.array([[ca.cos(psi), -ca.sin(psi), 0],
                       [ca.sin(psi), ca.cos(psi), 0], [0, 0, 1]])
        Ry = np.array([[ca.cos(theta), 0, ca.sin(theta)], [0, 1, 0],
                       [-ca.sin(theta), 0, ca.cos(theta)]])
        Rx = np.array([[1, 0, 0], [0, ca.cos(phi), -ca.sin(phi)],
                       [0, ca.sin(phi), ca.cos(phi)]])
        eRb = ca.mtimes([Rz, Ry, Rx])  # body to inertial
        e3 = np.array([0, 0, 1]).reshape(-1, 1)
        T_matrix = np.array([[1, 0.0, -ca.sin(theta)],
                             [0.0,
                              ca.cos(phi),
                              ca.sin(phi) * ca.cos(theta)],
                             [0.0, -ca.sin(phi),
                              ca.cos(phi) * ca.cos(theta)]])
        bW = ca.mtimes([T_matrix, np.array([d_phi, d_theta, d_psi])])
        # the angle velocity of uav in the frame of world
        Ib = np.array([[self.Ixx, 0, 0], [0, self.Iyy, 0], [0, 0, self.Izz]])

        # Kinetic energy
        K = 1 / 2 * ca.mtimes([self.mass, p_dot.T, p_dot])\
            + 1 / 2 * ca.mtimes([bW.T, Ib, bW])  # kinetic energy of uav

        # potential energy
        U = ca.mtimes([self.mass, self.g_, e3.T, p])

        L = K - U
        # Lagrangian = ca.Function('Lagrangian',[states],[L])
        self.fct_L = ca.Function('fct_L', [states_2, d_states], [L])

        L_ddstates = ca.gradient(L, d_states)
        # L_dstates = ca.gradient(L, states_2)
        self.fct_L_ddstates = ca.Function('fct_L_ddstates',
                                          [states_2, d_states], [L_ddstates])
        # all input forces from motors
        U1 = SX.sym("U1")  # motor 1
        U2 = SX.sym("U2")  # motor 2
        U3 = SX.sym("U3")  # motor 3
        U4 = SX.sym("U4")  # motor 4
        u = ca.vertcat(U1, U2, U3, U4)
        f_local = U1 + U2 + U3 + U4
        f_xyz = ca.mtimes([eRb, e3, f_local])
        moment_x = (U1 + U4 - U2 - U3) * self.l
        moment_y = (U3 + U4 - U1 - U2) * self.l
        moment_z = (U1 + U3 - U2 - U4) * self.ctau
        rhs_f = ca.vertcat(f_xyz, moment_x, moment_y, moment_z)
        # force function
        self.fct_f = ca.Function('fct_f', [states_2, u], [rhs_f])

        self.n_states = states.size()[0]
        self.n_controls = u.size()[0]

    def dynamics(self):
        p = MX.sym('p', 3)
        v = MX.sym('v', 3)
        q = MX.sym('q', 4)
        w = MX.sym('w', 3)
        T = MX.sym('thrust', 4)

        x = vertcat(p, v, q, w)
        u = vertcat(T)

        g = DM([0, 0, -self.g])

        x_dot = vertcat(
            v,
            rotate_quat(q, vertcat(0, 0,
                                   (T[0] + T[1] + T[2] + T[3]) / self.m)) + g -
            v * self.cd, 0.5 * quat_mult(q, vertcat(0, w)),
            mtimes(
                self.I_inv,
                vertcat(self.l * (T[0] - T[1] - T[2] + T[3]),
                        self.l * (-T[0] - T[1] + T[2] + T[3]),
                        self.ctau * (T[0] - T[1] + T[2] - T[3])) -
                cross(w, mtimes(self.I, w))))
        fx = Function('f', [x, u], [x_dot], ['x', 'u'], ['x_dot'])
        return fx
