from casadi import MX, DM, SX, vertcat, horzcat, veccat, norm_2, dot, mtimes, nlpsol, diag, repmat, sum1
import casadi as ca
import numpy as np
import inspect
from copy import deepcopy


class Planner:
    def __init__(self, quad, track, Integrator, options={}):
        # Essentials
        self.quad = quad
        self.track = track
        self.options = options

        # Track
        self.wp = DM(track.gates).T
        if track.end_pos is not None:
            if len(track.gates) > 0:
                self.wp = horzcat(self.wp, DM(track.end_pos))
            else:
                self.wp = DM(track.end_pos)

        if track.init_pos is not None:
            self.p_init = DM(track.init_pos)
        else:
            self.p_init = self.wp[:, -1]
        if track.init_att is not None:
            self.q_init = DM(track.init_att)
        else:
            self.q_init = DM([1, 0, 0, 0]).T

        # Dynamics
        self.quad.get_Lagrangian_casadi()

        # Sizes
        self.NX = 6
        self.NU = self.quad.n_controls
        self.NW = self.wp.shape[1]

        dist = [np.linalg.norm(self.wp[:, 0] - self.p_init)]
        for i in range(self.NW - 1):
            dist += [
                dist[i] + np.linalg.norm(self.wp[:, i + 1] - self.wp[:, i])
            ]

        if 'nodes_per_gate' in options:
            self.NPW = options['nodes_per_gate']
        else:
            self.NPW = 30

        if 'tolerance' in options:
            self.tol = options['tolerance']
        else:
            self.tol = 0.3

        self.N = self.NPW * self.NW
        self.dpn = dist[-1] / self.N
        if self.dpn < self.tol:
            suff_str = "sufficient"
        else:
            suff_str = "insufficient"
        print("Discretization over %d nodes and %1.1fm" % (self.N, dist[-1]))
        print("results in %1.3fm per node, %s for tolerance of %1.3fm." %
              (self.dpn, suff_str, self.tol))

        self.i_switch = np.array(self.N * np.array(dist) / dist[-1], dtype=int)

        # Problem variables
        self.x = []
        self.xg = []
        self.g = []
        self.lb = []
        self.ub = []
        self.J = []

        # Solver
        if 'solver_options' in options:
            self.solver_options = options['solver_options']
        else:
            self.solver_options = {}
            ipopt_options = {}
            ipopt_options['max_iter'] = 10000
            self.solver_options['ipopt'] = ipopt_options
        if 'solver_type' in options:
            self.solver_type = options['solver_type']
        else:
            self.solver_type = 'ipopt'
        self.iteration_plot = []

        # Guesses
        if 't_guess' in options:
            self.t_guess = options['t_guess']
            self.vel_guess = dist[-1] / self.t_guess
        elif 'vel_guess' in options:
            self.vel_guess = options['vel_guess']
            self.t_guess = dist[-1] / self.vel_guess
        else:
            self.vel_guess = 2.5
            self.t_guess = dist[-1] / self.vel_guess

        if 'legacy_init' in options:
            if options['legacy_init']:
                self.i_switch = range(self.NPW, self.N + 1, self.NPW)

    def set_iteration_callback(self, callback):
        self.iteration_callback = callback

    def set_init_pos(self, position):
        self.p_init = DM(position)

    def set_init_att(self, quaternion):
        self.q_init = DM(quaternion)

    def set_t_guess(self, t):
        self.t_guess

    def set_initial_guess(self, x_guess):
        if len(x_guess) > 0 and (self.xg.shape[0] == len(x_guess)
                                 or self.xg.shape[0] == 0):
            self.xg = veccat(*x_guess)

    def init_state_guess(self, ):
        i_wp = 0
        # linearly interpolate max thrust and max omegas
        u0 = np.tile(np.array([self.quad.T_max] * 4),
                     self.N + 1).reshape(-1, self.N + 1)
        mu0 = []
        tau0 = np.tile(np.array([0.0] * self.NW), self.N)
        lambda0 = [1.0] * self.NW
        p_init = self.p_init.full().flatten().tolist()
        q_init = self.q_init.full().flatten().tolist()
        rpy_init = self.quaternion_to_rpy(q_init)
        vel_guess = (self.wp[:, 0].full() - self.p_init.full())
        vel_guess = (self.vel_guess * vel_guess /
                     norm_2(vel_guess)).full().flatten().tolist()
        x0 = [
            p_init[0], p_init[1], p_init[2], rpy_init[0], rpy_init[1],
            rpy_init[2]
        ]

        pos_guess = p_init
        for i in range(self.N):
            # if i >= (1 + i_wp) * self.NPW:
            if i > self.i_switch[i_wp]:
                i_wp += 1
            if i_wp == 0:
                wp_last = self.p_init
            else:
                wp_last = self.wp[:, i_wp - 1]
            wp_next = self.wp[:, i_wp]
            if i_wp > 0:
                interp = (i - self.i_switch[i_wp - 1]) / (
                    self.i_switch[i_wp] - self.i_switch[i_wp - 1])
            else:
                interp = i / self.i_switch[0]
            pos_guess = (1 - interp) * wp_last + interp * wp_next
            vel_guess = self.vel_guess * (wp_next - wp_last) / norm_2(wp_next -
                                                                      wp_last)
            # direction = (wp_next - wp_last)/norm_2(wp_next - wp_last)
            # if interp < 0.5:
            #   vel_guess = interp * 4 * self.vel_guess * direction
            # else:
            #   vel_guess = (1 - interp) * 4 * self.vel_guess * direction
            # pos_guess += self.t_guess / self.N * vel_guess
            x0 = x0 + pos_guess.full().flatten().tolist() + rpy_init.flatten(
            ).tolist()

            if self.quad.rampup_dist > 0:
                T_max = max(
                    min(
                        self.interpolate(0, self.quad.T_ramp_start,
                                         self.quad.rampup_dist,
                                         self.quad.T_max, i * self.dpn),
                        self.quad.T_max), self.quad.T_ramp_start)
                u0[:, i] = np.array([T_max] * 4)

            if ((i_wp == 0) and (i + 1 >= self.i_switch[0])
                ) or i + 1 - self.i_switch[i_wp - 1] >= self.i_switch[i_wp]:
                mu = [0] * self.NW
                mu[i_wp] = 1.0
                mu0 += mu
            else:
                mu0 += [0] * self.NW
            for j in range(self.NW):
                # if i >= ((j+1)*self.NPW - 1):
                if i + 1 >= self.i_switch[j]:
                    lambda0 += [0]
                else:
                    lambda0 += [1]
        print("N: ", self.N)
        print(np.array(x0).shape)
        x0 = np.array(x0).reshape(-1, self.N + 1)
        print("x guess", x0.reshape(-1, 1)[:6])
        lambda0 = np.array(lambda0).reshape(self.N + 1, self.NW)
        self.guess_state = ca.vcat([
            self.t_guess,
            u0.reshape(-1, 1),
            x0.reshape(-1, 1),
            np.array(lambda0).reshape(-1, 1),
            np.array(mu0).reshape(-1, 1),
            np.array(tau0).reshape(-1, 1)
        ])

    def setup(self):
        # Total time variable
        t = SX.sym('t', 1)
        u = SX.sym('u', self.NU, self.N + 1)
        x = SX.sym('x', self.NX, self.N + 1)
        lamg = SX.sym('lamg', self.NW, self.N + 1)
        mu = SX.sym('mu', self.NW, self.N)
        tau = SX.sym('tau', self.NW, self.N)
        J = t
        self.dt = t / self.N

        # DMOC setup
        # discrete lagrange equations
        q_nm1 = SX.sym('qnm1', self.NX)
        q_n = SX.sym('qn', self.NX)
        q_np1 = SX.sym('qnp1', self.NX)
        D2L_d = ca.gradient(
            self.discrete_lagrange(self.dt, q_nm1, q_n, self.quad.func_L), q_n)
        D1L_d = ca.gradient(
            self.discrete_lagrange(self.dt, q_n, q_np1, self.quad.func_L), q_n)
        d_EulerLagrange = ca.Function('dEL', [q_nm1, q_n, q_np1],
                                      [D2L_d + D1L_d])
        q_b = SX.sym('q_b', self.NX)
        q_b_dot = SX.sym('q_b_dot', self.NX)
        D2L = self.quad.func_L_ddstates(q_b, q_b_dot)
        d_EulerLagrange_init = ca.Function('dEl_init',
                                           [q_b, q_b_dot, q_n, q_np1],
                                           [D2L + D1L_d])
        d_EulerLagrange_end = ca.Function('dEl_end',
                                          [q_b, q_b_dot, q_nm1, q_n],
                                          [-D2L + D2L_d])

        g = []

        if self.track.init_pos is not None:
            # if self.track.init_pos is not None and not self.track.ring:
            print('Using start position constraint')
            for i in range(3):
                g.append(x[i, 0] - self.track.init_pos[i])
                # g.append(x[i, -1]-self.wp[i, -1].__float__())
        if self.track.init_vel is not None:
            print('Using start velocity constraint')
            for i in range(3):
                g.append((x[i, 1] - x[i, 0]) / self.dt -
                         self.track.init_vel[i])
        if self.track.init_att is not None:
            print('Using start attitude constraint')
            rpy_des_ = self.quaternion_to_rpy(self.track.init_att)
            for i in range(3):
                g.append(rpy_des_[i] - x[i + 3, 0])
                # g.append(x[i+3, -1])
        else:
            for i in range(3):
                g.append(x[3 + i, 0])
        if self.track.init_omega is not None:
            print('Using start bodyrate constraint')
            d_q_ = x[3:, 1] - x[3:, 0]
            for i in range(3):
                q_dot_ = ca.atan2(ca.sin(d_q_[i]), ca.cos(d_q_[i])) / self.dt
                g.append(q_dot_ - self.track.init_omega[i])
        self.init_state_guess()

        # Bound inital progress variable to 1
        for i in range(self.NW):
            g.append(lamg[i, 0] - 1)
            g.append(lamg[i, -1])

        # For each node ...
        for i in range(1, self.N):
            f_d_nm1 = self.discrete_forces_v2(self.dt, self.quad.func_f,
                                              x[:, i - 1], u[:, i - 1], u[:,
                                                                          i])
            f_d_n = self.discrete_forces_v2(self.dt, self.quad.func_f, x[:, i],
                                            u[:, i], u[:, i + 1])
            sum = d_EulerLagrange(x[:, i - 1], x[:, i],
                                  x[:, i + 1]) + f_d_nm1 + f_d_n
            g.append(sum)

        # boundary condition (x_0, x_end)
        rpy_des_ = self.quaternion_to_rpy(self.track.init_att)
        f_0 = self.discrete_forces_v2(self.dt, self.quad.func_f, x[:, 0],
                                      u[:, 0], u[:, 1])
        g.append(
            d_EulerLagrange_init(self.track.init_pos +
                                 rpy_des_.tolist(), self.track.init_vel +
                                 self.track.init_omega, x[:, 0], x[:, 1]) +
            f_0)

        # f_N_1 = self.discrete_forces_v2(self.dt, self.quad.func_f,
        #                                 x[:, self.N - 1],
        #                                 u[:, self.N - 1],
        #                                 u[:, self.N ])
        # g.append(
        #     d_EulerLagrange_end([self.wp[0, -1].__float__(), self.wp[1, -1].__float__(), self.wp[2, -1].__float__(),0, 0, 0], [0, 0, 0, 0, 0, 0],
        #                         x[:, self.N -1], x[:, self.N]) + f_N_1)

        for i in range(self.N):
            for j in range(self.NW):
                # cc
                mid_result_ = self.average_state(x[:, i], x[:, i + 1])
                diff = mid_result_[0:3] - self.wp[:, j]
                g.append(mu[j, i] * (dot(diff, diff) - tau[j, i]))
                # g.append(mu[j, i]*(ca.norm_2(mid_result_[:3]-self.wp[:, j])-tau[j]))
                # change of mu
                g.append(lamg[j, i] - lamg[j, i + 1] - mu[j, i])
        self.equal_constraint_length = np.shape(ca.vertcat(*g))[0]
        print('Total number of equal constraints {}:'.format(
            self.equal_constraint_length))

        # omega constraints
        for i in range(self.N):
            d_q_ = x[3:, i + 1] - x[3:, i]
            for i in range(3):
                g.append(ca.atan2(ca.sin(d_q_[i]), ca.cos(d_q_[i])) / self.dt)

        for i in range(self.N + 1):
            for j in range(self.NW - 1):
                g.append(lamg[j + 1, i] - lamg[j, i])

        # for i in range(self.N):
            # for j in range(self.NW):
            #     # cc
            #     mid_result_ = self.average_state(x[:, i], x[:, i + 1])
            #     diff = mid_result_[0:3] - self.wp[:, j]
            #     g.append(mu[j, i] * (dot(diff, diff) - tau[j, i]))

        self.unequal_constraint_length = np.shape(
            ca.vertcat(*g))[0] - self.equal_constraint_length
        print('Total number of unequal constraints {}:'.format(
            self.unequal_constraint_length))

        # Reformat
        self.x = ca.vcat([
            t,
            ca.reshape(u, -1, 1),
            ca.reshape(x, -1, 1),
            ca.reshape(lamg, -1, 1),
            ca.reshape(mu, -1, 1),
            ca.reshape(tau, -1, 1)
        ])

        self.g = ca.vertcat(*g)
        self.J = J

        # Construct Non-Linear Program
        self.nlp = {'f': self.J, 'x': self.x, 'g': self.g}

        # constraints
        lbg = []
        ubg = []
        # equality constraints
        for _ in range(self.equal_constraint_length):
            lbg.append(0.0)
            ubg.append(0.0)

        # limit omega
        for i in range(self.N):
            omega_max_xy = self.quad.omega_max_xy
            if self.quad.rampup_dist > 0:
                omega_max_xy = max(
                    min(
                        self.interpolate(0, self.quad.omega_ramp_start,
                                         self.quad.rampup_dist,
                                         self.quad.omega_max_xy, i * self.dpn),
                        self.quad.omega_max_xy), self.quad.omega_ramp_start)
            lbg += [-omega_max_xy, -omega_max_xy, -self.quad.omega_max_z]
            ubg += [omega_max_xy, omega_max_xy, self.quad.omega_max_z]

        for i in range(self.N + 1):
            for j in range(self.NW - 1):
                lbg += [0.0]
                ubg += [1.0]

        # for i in range(self.N):
        #     for j in range(self.NW):
        #         lbg += [0.0]
        #         ubg += [0.01]

        lbx = [0.1]
        ubx = [150]

        # U
        for i in range(self.N + 1):
            T_max = self.quad.T_max
            if self.quad.rampup_dist > 0:
                T_max = max(
                    min(
                        self.interpolate(0, self.quad.T_ramp_start,
                                         self.quad.rampup_dist,
                                         self.quad.T_max, i * self.dpn),
                        self.quad.T_max), self.quad.T_ramp_start)
            lbx += [self.quad.T_min] * self.NU
            ubx += [T_max] * self.NU

        # X
        for i in range(self.N + 1):
            omega_max_xy = self.quad.omega_max_xy
            if self.quad.rampup_dist > 0:
                omega_max_xy = max(
                    min(
                        self.interpolate(0, self.quad.omega_ramp_start,
                                         self.quad.rampup_dist,
                                         self.quad.omega_max_xy, i * self.dpn),
                        self.quad.omega_max_xy), self.quad.omega_ramp_start)

            lbx = lbx + [
                -np.inf,
                -np.inf,
                0.5,
                -np.inf,
                -np.inf,
                -np.inf,
            ]
            ubx = ubx + [
                np.inf,
                np.inf,
                100.0,
                np.inf,
                np.inf,
                np.inf,
            ]

        # lambda
        for _ in range(self.N + 1):
            for _ in range(self.NW):
                lbx.append(0)
                ubx.append(1)
        # mu
        for _ in range(self.N):
            for _ in range(self.NW):
                lbx.append(0)
                ubx.append(1)
        # tau
        for _ in range(self.N):
            for _ in range(self.NW):
                lbx.append(0)
                ubx.append(self.tol**2)

        self.lb_x = lbx
        self.ub_x = ubx
        self.lb_g = lbg
        self.ub_g = ubg

    def solve(self, x_guess=[]):
        self.set_initial_guess(x_guess)

        if hasattr(
                self,
                'iteration_callback') and self.iteration_callback is not None:
            if inspect.isclass(self.iteration_callback):
                callback = self.iteration_callback("IterationCallback")
            elif self.iteration_callback:
                callback = self.iteration_callback

            callback.set_size(self.x.shape[0], self.g.shape[0], self.NPW)
            callback.set_wp(self.wp)
            self.solver_options['iteration_callback'] = callback

        self.solver = nlpsol('solver', self.solver_type, self.nlp,
                             self.solver_options)

        # self.solution = self.solver(x0=self.xg, lbg=self.lb, ubg=self.ub)
        self.solution = self.solver(x0=self.guess_state,
                                    lbg=self.lb_g,
                                    ubg=self.ub_g,
                                    lbx=self.lb_x,
                                    ubx=self.ub_x)
        self.x_sol = self.solution['x'].full().flatten()
        return self.x_sol

    def interpolate(self, x1, y1, x2, y2, x):
        if (abs(x2 - x1) < 1e-5):
            return 0

        return y1 + (y2 - y1) / (x2 - x1) * (x - x1)

    @staticmethod
    def discrete_lagrange(dt, q_n, q_np1, fct_L, num_state=6):
        """
        second-order accurate diescrete Lagrangian
        Args:
            dt: diescrete time step
            q_n: state at n step
            q_np1: state at n+1 step
            fct_L: lagrange function
        Returns:
            symbolic
        """
        q = (q_n + q_np1) / 2.
        # average of euler angles in Rad, [x, y, z, r, p, y]
        for i in range(3, num_state):
            s_ = ca.sin(q_n[i]) + ca.sin(q_np1[i])
            c_ = ca.cos(q_n[i]) + ca.cos(q_np1[i])
            q[i] = ca.atan2(s_, c_)

        q_dot = (q_np1 - q_n) / dt
        for i in range(3, num_state):
            diff_ = q_np1[i] - q_n[i]
            q_dot[i] = ca.atan2(ca.sin(diff_), ca.cos(diff_)) / dt
        L_d = dt * fct_L(q, q_dot)
        return L_d

    @staticmethod
    def average_velocity(state_1, state_2, dt, num_state=6):
        # velocity (q_2 - q_1)/dt
        assert state_1.shape == (num_state,
                                 1) and state_2.shape == (num_state,
                                                          1), state_1.shape
        d_q = (state_2 - state_1)

        if num_state > 3:
            q_dot = deepcopy(state_1)
            for i in range(3):
                q_dot[i] = d_q[i] / dt
                q_dot[i + 3] = ca.atan2(ca.sin(d_q[i + 3]), ca.cos(
                    d_q[i + 3])) / dt

        return q_dot

    @staticmethod
    def quaternion_to_rpy(quaternion):
        q0, q1, q2, q3 = quaternion
        roll_ = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
        pitch_ = np.arcsin(2 * (q0 * q2 - q3 * q1))
        yaw_ = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))
        return np.array([roll_, pitch_, yaw_])

    @staticmethod
    def rpy_to_quaternion(rpy):
        roll_, pitch_, yaw_ = rpy
        cy = np.cos(yaw_ * 0.5)
        sy = np.sin(yaw_ * 0.5)
        cp = np.cos(pitch_ * 0.5)
        sp = np.sin(pitch_ * 0.5)
        cr = np.cos(roll_ * 0.5)
        sr = np.sin(roll_ * 0.5)

        w_ = cr * cp * cy + sr * sp * sy
        x_ = sr * cp * cy - cr * sp * sy
        y_ = cr * sp * cy + sr * cp * sy
        z_ = cr * cp * sy - sr * sp * cy
        return np.array([w_, x_, y_, z_])

    @staticmethod
    def discrete_forces_v2(dt, f, q, u_n, u_np1):
        f_d = 1 / 4. * dt * (f(q, u_n) + f(q, u_np1))
        return f_d

    @staticmethod
    def average_state(state_1, state_2, num_state=6):
        # average whole state
        assert state_1.shape == (num_state,
                                 1) and state_2.shape == (num_state,
                                                          1), state_1.shape
        q = (state_1 + state_2) / 2
        # mean of euler angles
        for i in range(3, num_state):
            s_ = ca.sin(state_1[i]) + ca.sin(state_2[i])
            c_ = ca.cos(state_1[i]) + ca.cos(state_2[i])
            q[i] = ca.atan2(s_, c_)
        return q
