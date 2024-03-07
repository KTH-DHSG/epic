class ConfigMPIBVS(object):
    def __init__(self, jit):
        self.jit = jit

    def get_functions_config(self, basic=True):
        """
        Get functions options.

        :param basic: check if empty options are used
        :type basic: bool
        """
        if not basic:
            fun_options = {
                "jit": self.jit,
                "jit_options": {'compiler': 'ccache gcc',
                                'flags': ["-O2", "-pipe"]},
                'compiler': 'shell',
                'jit_temp_suffix': False
            }
        else:
            fun_options = {}
        return fun_options

    def get_solvers_config(self, basic=False):
        """
        Get solver options.

        :param basic: check if empty options are used
        :type basic: bool
        """
        # Options for NLP Solvers
        if not basic:
            # SQP Method
            qp_opts = {
                'max_iter': 1000,
                'error_on_fail': False,
                'print_header': False,
                'print_iter': False
            }
            sol_options_sqp = {
                'qpsol': 'qrqp',
                "jit": self.jit,
                "jit_options": {'compiler': 'ccache gcc',
                                'flags': ["-O2", "-pipe"]},
                'compiler': 'shell',
                'jit_temp_suffix': False,
                'print_header': False,
                'print_time': False,
                'print_iteration': False,
                'qpsol_options': qp_opts
            }

            # -> IPOPT
            sol_options_ipopt = {
                # 'ipopt.max_iter': 20,
                # 'ipopt.max_resto_iter': 30,
                "ipopt.print_level": 0,
                # 'ipopt.mu_init': 0.01,
                "ipopt.tol": 1e-19,
                "ipopt.acceptable_obj_change_tol": 1e-10,
                "ipopt.tiny_step_y_tol": 1e-10,
                # 'ipopt.warm_start_init_point': 'yes',
                # 'ipopt.warm_start_bound_push': 1e-4,
                # 'ipopt.warm_start_bound_frac': 1e-4,
                # 'ipopt.warm_start_slack_bound_frac': 1e-4,
                # 'ipopt.warm_start_slack_bound_push': 1e-4,
                # 'ipopt.warm_start_mult_bound_push': 1e-4,
                "print_time": False,
                "verbose": False,
                "expand": True,
                "jit": self.jit,
                "jit_options": {"compiler": "ccache gcc", "flags": ["-O2", "-pipe"]},
                "compiler": "shell",
            }
        else:
            sol_options_sqp = {}
            sol_options_ipopt = {
                'ipopt.print_level': 0,
                'print_time': False,
                'verbose': False,
                'expand': True,
            }

        return sol_options_sqp, sol_options_ipopt
