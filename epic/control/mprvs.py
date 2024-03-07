from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import casadi as ca
import casadi.tools as ctools

from epic.utils.veronese import lift_image_points


class MPRVS(object):
    """
    Model Predictive Relative Visual Servoing (MPIBVS) controller.

    This controller takes a set of epipolar lines and a desire range to align
    the two camera systems.

    An estimate of each point depth is expected. This can be fixed or varying.
    """

    def __init__(self, mpc_args):

        # Check validity of mpc_args:
        self.check_args_validity(mpc_args)

        # If proper arguments, then unpack and instantiate problem
        self.model = mpc_args["model"]
        self.dynamics = mpc_args["dynamics"]
        self.N = mpc_args["N"]

        self.Q = mpc_args["Q"]
        self.P = mpc_args["P"]
        self.R = mpc_args["R"]

        self.xub = mpc_args["xub"]
        self.xlb = mpc_args["xlb"]
        self.xf = mpc_args["xf"]

        self.uub = mpc_args["uub"]
        self.ulb = mpc_args["ulb"]

        # Set problem dimensions
        self.Nx = self.model.f_n * 2 + 3  # Tracked features number + realtive position
        self.Ne = self.model.f_n + 1
        self.Nt = self.N
        self.Nr = 1 + 3 * 3 * self.model.f_n    # range + f_n reference curves in matrix form
        self.NRr = 3 * 3                        # rotation matrix from leader
        self.Nu = self.model.m
        self.dt = self.model.dt

        # Initialize trajectory variables
        self.x_pred = None
        self.u_pred = None
        self.cost = np.inf

        # Create options and cost functions, followed by a test
        self.set_options_dicts()
        self.set_cost_functions()

        # Create solver object
        self.create_solver()

        pass

    def check_args_validity(self, mpc_args):
        """
        Helper function to handle MPC parameters validity checks.

        :param mpc_args: dictionary with the MPC parameters
        :type mpc_args: dict
        """
        if "model" not in mpc_args:
            raise ValueError("Missing 'model' in mpc_args.")

        if "dynamics" not in mpc_args:
            raise ValueError("Missing MPC 'dynamics' function in mpc_args.")

        if "N" not in mpc_args:
            raise ValueError("Missing MPC 'N' in mpc_args.")

        if "Q" not in mpc_args:
            raise ValueError("Missing MPC weight matrix 'Q' in mpc_args.")

        if "R" not in mpc_args:
            raise ValueError("Missing MPC weight matrix 'R' in mpc_args.")

        if "P" not in mpc_args:
            raise ValueError("Missing MPC weight matrix 'P' in mpc_args.")

        if "xub" not in mpc_args or "xlb" not in mpc_args:
            raise ValueError("Missing MPC state bounds 'xub'/'xlb' in mpc_args.")

        if "uub" not in mpc_args or "ulb" not in mpc_args:
            raise ValueError("Missing MPC control bounds 'uub'/'ulb' in mpc_args.")

        if "xf" not in mpc_args:
            raise ValueError("Missing MPC terminal state constraint 'xf' in mpc_args.")

    def create_solver(self):
        """
        Instantiate the MPIBVS controller.
        """
        build_solver_time = -time.time()
        self.x_sp = None

        # Starting state parameters
        x0 = ca.MX.sym('x0', self.Nx)
        x_ref = ca.MX.sym('x_ref', self.Nr * self.Nt)
        u_ref = ca.MX.sym('u_ref', self.Nu * (self.Nt - 1))
        z0 = ca.MX.sym('z0', self.model.f_n)
        Rlf = ca.MX.sym('RLF', self.NRr)
        param_s = ca.vertcat(x0, x_ref, Rlf, u_ref, z0)
        Rlf = Rlf.reshape((3, 3))

        # Create optimization variables
        opt_var = ctools.struct_symMX([(ctools.entry('u', shape=(self.Nu,), repeat=self.Nt - 1),
                                        ctools.entry('x', shape=(self.Nx,), repeat=self.Nt),
                                        )])
        self.opt_var = opt_var
        self.num_var = opt_var.size

        # Decision variable boundries
        self.optvar_lb = opt_var(-np.inf)
        self.optvar_ub = opt_var(np.inf)

        # Set initial values
        obj = ca.MX(0)
        con_eq = []
        con_ineq = []
        con_ineq_lb = []
        con_ineq_ub = []
        con_eq.append(opt_var['x', 0] - x0)

        # Generate MPC Problem
        for t in range(self.Nt - 1):
            # Get variables
            x_t = opt_var['x', t]
            u_t = opt_var['u', t]
            xi_ref = x_ref[t * self.Nr:(t + 1) * self.Nr]
            ui_ref = u_ref[t * self.Nu:(t + 1) * self.Nu]

            # Dynamics constraint
            x_t_next = self.dynamics(x_t, u_t, z0, ui_ref, Rlf)
            con_eq.append(x_t_next - opt_var['x', t + 1])

            # Input constraints
            con_ineq.append(u_t)
            con_ineq_ub.append(self.uub)
            con_ineq_lb.append(np.full((self.Nu,), -ca.inf))
            con_ineq.append(u_t)
            con_ineq_ub.append(np.full((self.Nu,), ca.inf))
            con_ineq_lb.append(self.ulb)

            # State constraints
            con_ineq.append(x_t)
            con_ineq_ub.append(self.xub)
            con_ineq_lb.append(np.full((self.Nx,), -ca.inf))

            con_ineq.append(x_t)
            con_ineq_ub.append(np.full((self.Nx,), ca.inf))
            con_ineq_lb.append(self.xlb)

            # Objective Function / Cost Function
            obj += self.running_cost(x_t, xi_ref, self.Q, u_t - ui_ref, self.R)

        # Terminal Cost
        obj += self.terminal_cost(opt_var['x', -1], x_ref[-self.Nr:], self.P)

        # Equality constraints bounds are 0 (they are equality constraints),
        # -> Refer to CasADi documentation
        num_eq_con = ca.vertcat(*con_eq).size1()
        num_ineq_con = ca.vertcat(*con_ineq).size1()
        con_eq_lb = np.zeros((num_eq_con, 1))
        con_eq_ub = np.zeros((num_eq_con, 1))

        # Stack constraints
        con = ca.vertcat(*(con_eq + con_ineq))
        self.con_lb = ca.vertcat(con_eq_lb, *con_ineq_lb)
        self.con_ub = ca.vertcat(con_eq_ub, *con_ineq_ub)

        # Create solver:
        nlp = dict(x=opt_var, f=obj, g=con, p=param_s)
        self.solver = ca.nlpsol('mpc_solver', 'ipopt', nlp,
                                self.sol_options_ipopt)

        build_solver_time += time.time()
        print('\n________________________________________')
        print('# Receding horizon length: %d ' % self.Nt)
        print('# Receding horizon time-span: {} seconds'.format(self.Nt * self.dt))
        print('# Time to build mpc solver: %f sec' % build_solver_time)
        print('# Number of variables: %d' % self.num_var)
        print('# Number of equality constraints: %d' % num_eq_con)
        print('# Number of inequality constraints: %d' % num_ineq_con)
        print('----------------------------------------')
        pass

    def set_options_dicts(self):
        """
        Helper function to set the dictionaries for solver and function options
        """
        self.set_jit = False

        self.fun_options = {}

        self.sol_options_sqp = {}

        # Options for IPOPT Solver
        # -> IPOPT
        self.sol_options_ipopt = {
            'ipopt.max_iter': 20,
            'ipopt.max_resto_iter': 30,
            'ipopt.print_level': 0,
            'ipopt.print_header': False,
            'print_time': False,
            'verbose': False,
            'expand': True,
            "jit": self.set_jit,
            "jit_options": {'compiler': 'ccache gcc',
                            'flags': ["-O2", "-pipe"]},
            'compiler': 'shell',
        }
        self.sol_options_ipopt = {
            'ipopt.print_level': 0,
            'print_time': False,
            'verbose': False,
            'expand': True,
        }
        return True

    def set_cost_functions(self):
        """
        Helper function to setup the cost functions.
        """

        # Create functions and function variables for calculating the cost
        Q = ca.MX.sym('Q', self.Ne, self.Ne)
        P = ca.MX.sym('P', self.Ne, self.Ne)
        R = ca.MX.sym('R', self.Nu, self.Nu)

        x = ca.MX.sym('x', self.Nx)
        xr = ca.MX.sym('xr', self.Nr)
        u = ca.MX.sym('u', self.Nu)

        # Prepare variables
        e_vec = (x[0]**2 + x[1]**2 + x[2]**2) - xr[0]**2  # It is squared, so no need for abs

        # Range error: || ||td|| - rd  ||**2
        # e_vec = ca.norm_2((x[0]**2 + x[1]**2 + x[2]**2) - xr[0]**2)

        # Get matrices
        for i in range(0, self.model.f_n):
            f = ca.vertcat(x[3 + i * 2:3 + (i * 2 + 2)], 1)
            M = ca.reshape(xr[1 + i * 9:1 + (i * 9 + 9)], 3, 3)
            e_vec = ca.vertcat(e_vec, f.T @ M @ f)

        # Calculate running cost
        ln = ca.mtimes(ca.mtimes(e_vec.T, Q), e_vec) \
            + ca.mtimes(ca.mtimes(u.T, R), u)

        self.running_cost = ca.Function('ln', [x, xr, Q, u, R], [ln],
                                        self.fun_options)

        # Calculate terminal cost
        V = ca.mtimes(ca.mtimes(e_vec.T, P), e_vec)
        self.terminal_cost = ca.Function('V', [x, xr, P], [V],
                                         self.fun_options)

        return

    def get_error_vector(self, x, xr):
        e_vec = (x[0]**2 + x[1]**2 + x[2]**2) - xr[0]**2
        e_vec = e_vec.T**2 * self.Q[0, 0]

        # Get matrices
        for i in range(0, self.model.f_n):
            f = ca.vertcat(x[3 + i * 2:3 + (i * 2 + 2)], 1)
            M = ca.reshape(xr[1 + i * 9:1 + (i * 9 + 9)], 3, 3)
            e_f = f.T @ M @ f
            e_f = e_f.T**2 * self.Q[i + 1, i + 1]
            e_vec = ca.vertcat(e_vec, e_f)
        return e_vec

    def get_veronese_error_vector(self, x, xr, lines):
        e_vec = (x[0]**2 + x[1]**2 + x[2]**2) - xr[0]**2

        # Get matrices
        for i in range(0, self.model.f_n):
            f = ca.vertcat(x[3 + i * 2:3 + (i * 2 + 2)], 1)
            vf = lift_image_points(f)
            line = lines[:, i]
            e_vec = ca.vertcat(e_vec, vf.T @ line)
        return e_vec

    def compare_errors(self, x, xr, lines):

        e_vec = (x[0]**2 + x[1]**2 + x[2]**2) - xr[0]**2
        ev_vec = e_vec

        # Get matrices
        for i in range(0, self.model.f_n):
            # opt 1
            f = ca.vertcat(x[3 + i * 2:3 + (i * 2 + 2)], 1)
            vf = lift_image_points(f)
            line = lines[:, i]
            ev = vf.T @ line

            # opt 2
            f = ca.vertcat(x[3 + i * 2:3 + (i * 2 + 2)], 1)
            M = ca.reshape(xr[1 + i * 9:1 + (i * 9 + 9)], 3, 3)
            e = f.T @ M @ f

            # Stack vectors
            ev_vec = ca.vertcat(ev_vec, ev)
            e_vec = ca.vertcat(e_vec, e)
        return e_vec

    def solve_mpc(self, x0, p=None):
        """
        Solve the MPC problem

        :param x0: state
        :type x0: ca.DM
        :param u0: initia guess for the control input, defaults to None
        :type u0: ca.DM, optional
        :return: predicted states and control inputs
        :rtype: ca.DM ca.DM vectors
        """

        # Initial state
        if self.x_sp is None:
            raise ValueError("State setpoint not initialized. Please set reference.")

        if self.u_sp is None:
            raise ValueError("Control setpoint not initialized. Please set reference.")

        # Initialize variables
        self.optvar_x0 = np.full((1, self.Nx), x0.T)

        # Initial guess of the warm start variables
        self.optvar_init = self.opt_var(0)
        self.optvar_init['x', 0] = self.optvar_x0[0]

        # param_s = ca.vertcat(x0, x_ref, u0, z0)
        param = ca.vertcat(x0, self.x_sp, self.u_sp, p)
        args = dict(x0=self.optvar_init,
                    lbx=self.optvar_lb,
                    ubx=self.optvar_ub,
                    lbg=self.con_lb,
                    ubg=self.con_ub,
                    p=param)

        # Solve NLP
        self.solve_time = -time.time()
        sol = self.solver(**args)
        self.solve_time += time.time()
        status = self.solver.stats()['success']
        optvar = self.opt_var(sol['x'])
        self.cost = float(sol['f'])

        print(" Solve time: {:.4f} - J*: {:.4f}".format(self.solve_time, float(sol['f'])), "|", end="")
        if status is False:
            print(' Stop reason: ', self.solver.stats()['return_status'], "| ", end="")

        # Store predicted trajectories
        self.x_pred = optvar['x']
        self.u_pred = optvar['u']

        return self.x_pred, self.u_pred

    def control(self, x0, z0):
        """
        MPC interface.

        :param x0: initial state
        :type x0: ca.DM
        :param u0: initial guess for control input, defaults to None
        :type u0: ca.DM, optional
        :return: first control input
        :rtype: ca.DM
        """

        x_pred, u_pred = self.solve_mpc(x0, p=z0)

        # Save predictions as numpy arrays
        self.x_pred_np = np.array(x_pred).T.reshape((-1, self.Nt))
        self.u_pred_np = np.array(u_pred).T.reshape((-1, self.Nt - 1))

        # Return first input
        return u_pred[0]

    def get_predicted_features(self):
        """
        Return predicted features from last call to 'solve_mpc' in a numpy array format.

        :return: predicted features
        :rtype: ca.DM
        """
        # Features are:
        #   0,0 ---- N
        #    |
        #   Fn
        prediction_np = np.asarray(self.x_pred).reshape((self.Nt, self.Nx)).T
        return prediction_np[3:, :].reshape(-1, self.Nt)

    def get_pred_trajectories(self):
        """
        Return predicted trajectories  from last call to 'solve_mpc'.

        :return: predicted control inptuts, predicted states
        :rtype: ca.DM, ca.DM
        """

        return self.u_pred, self.x_pred

    def get_predicted_costs(self):
        """
        Return predicted costs from last call to 'solve_mpc'.

        :return: predicted costs for state and control
        :rtype: dict
        """
        # Cost matrices
        Q = self.Q
        R = self.R
        P = self.P

        # Calculate costs
        state_cost = np.empty((1, 0))
        control_cost = np.empty((1, 0))
        for t in range(self.Nt - 1):
            stage_cost = self.running_cost(self.x_pred[t], self.x_sp[t * self.Nr:(t + 1) * self.Nr], Q, self.u_pred[t] - self.u_sp[t * self.Nu:(t + 1) * self.Nu], R)
            u_err = self.u_pred[t] - self.u_sp[t * self.Nu:(t + 1) * self.Nu]
            control_cost = np.append(control_cost, u_err.T @ R @ u_err, axis=1)
            state_cost = np.append(state_cost, stage_cost - u_err.T @ R @ u_err, axis=1)

        t += 1
        terminal_cost = self.terminal_cost(self.x_pred[-1], self.x_sp[t * self.Nr:(t + 1) * self.Nr], P)
        state_cost = np.append(state_cost, terminal_cost, axis=1)
        predicted_costs = {"x": state_cost, "u": control_cost}
        return predicted_costs

    def get_cost(self):
        return self.cost

    def set_reference(self, x_sp, u_sp=None, test_RLF=None):
        """
        Set MPC reference.

        :param x_sp: reference for the state
        :type x_sp: ca.DM
        """
        self.x_sp = x_sp.reshape(self.Nr * self.Nt, 1, order="F")
        self.x_sp = np.concatenate((self.x_sp, test_RLF.reshape((-1, 1))), axis=0)
        if u_sp.shape[0] != self.Nu * (self.Nt - 1) or u_sp.shape[1] != 1:
            if u_sp.shape[0] == self.Nu and u_sp.shape[1] == 1:
                u_sp = np.tile(u_sp, (self.Nt - 1, 1))
            elif u_sp.shape[0] == self.Nu and u_sp.shape[1] == (self.Nt - 1):
                u_sp = u_sp.reshape(-1, 1, order="F")
                # Wrong here?
            else:
                print("Error: Wrong size of control setpoint. Expected: ", self.Nu * (self.Nt - 1), "x 1")
                print("\t Got: ", u_sp.shape[0], "x", u_sp.shape[1])
                exit()
        self.u_sp = u_sp

    def get_last_solve_time(self):
        """
        Helper function that returns the last solver time.
        """
        return self.solve_time
