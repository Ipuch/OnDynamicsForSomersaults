from typing import Any, Callable, Union
from itertools import product
from dataclasses import dataclass

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from bioptim import OdeSolver, SolutionIntegrator, Solver, MultiBodyDynamics

from viz import add_custom_plots
from Comparison import integrate_sol, compute_error_single_shooting


def filename(params: dict):
    file_str = ""
    for ii in params.values():
        file_str += f"{ii}_"
    file_str = file_str.replace("\n", "_")
    file_str = file_str.replace(".", "_")
    file_str = file_str.replace(" ", "_")
    file_str.join(".bo")
    return file_str[:-1]


class ComparisonParameters:
    def __init__(
            self,
            biorbd_model_path: Union[str, list] = None,
            ode_solver: Union[OdeSolver, list] = None,
            tolerance: Union[float, list] = None,
            n_shooting: Union[int, list] = None,
            multibody_dynamics: Union[MultiBodyDynamics, list] = MultiBodyDynamics.IMPLICIT,
    ):

        self.biorbd_model_path = self._is_a_list(biorbd_model_path)
        self.ode_solver = self._is_a_list(ode_solver)
        self.tolerance = self._is_a_list(tolerance)
        self.n_shooting = self._is_a_list(n_shooting)
        self.multibody_dynamics = self._is_a_list(multibody_dynamics)

        self.parameters_compared = dict()
        self.parameters_not_compared = dict()
        self.product_list = []
        self._set_varying_parameters()
        self._set_product_list()

    @staticmethod
    def _is_a_list(param):
        return param[0] if isinstance(param, list) and len(param) == 1 else param

    def _set_varying_parameters(self):
        self.parameters_compared = dict()
        if isinstance(self.biorbd_model_path, list):
            self.parameters_compared["biorbd_model_path"] = self.biorbd_model_path
        else:
            self.parameters_not_compared["biorbd_model_path"] = self.biorbd_model_path
        if isinstance(self.ode_solver, list):
            self.parameters_compared["ode_solver"] = self.ode_solver
        else:
            self.parameters_not_compared["ode_solver"] = self.ode_solver
        if isinstance(self.tolerance, list):
            self.parameters_compared["tolerance"] = self.tolerance
        else:
            self.parameters_not_compared["tolerance"] = self.tolerance
        if isinstance(self.n_shooting, list):
            self.parameters_compared["n_shooting"] = self.n_shooting
        else:
            self.parameters_not_compared["n_shooting"] = self.n_shooting
        if isinstance(self.multibody_dynamics, list):
            self.parameters_compared["multibody_dynamics"] = self.multibody_dynamics
        else:
            self.parameters_not_compared["multibody_dynamics"] = self.multibody_dynamics

    def product_generator(self):
        keys = self.parameters_compared.keys()
        vals = self.parameters_compared.values()
        for instance in product(*vals):
            yield dict(zip(keys, instance))

    def _set_product_list(self):
        list_combinations = []
        vals = self.parameters_compared.values()
        keys = self.parameters_compared.keys()
        for instance in product(*vals):
            list_combinations.append(dict(zip(keys, instance)))
        self.product_list = list_combinations
        print(self.product_list)

    def get_parameter(self, parameter_name):
        if parameter_name in self.parameters_not_compared:
            return self.parameters_not_compared[parameter_name]
        elif parameter_name in self.parameters_compared:
            return self.parameters_compared[parameter_name]
        else:
            raise ValueError(f"This parameter {parameter_name} is not in this ComparisonParameters object.")

    def size(self, parameter_name):
        return 1 if not isinstance(self.get_parameter(parameter_name), list) else len(
            self.get_parameter(parameter_name))


class ComparisonAnalysis:
    def __init__(self, Ocp: Any, Parameters: ComparisonParameters, solver_options: Solver):

        self.Ocp = Ocp
        self.ocp_arg = list(Ocp().__dict__.keys())
        self.solver_arg = ["tol", "cons"]
        self.Parameters = Parameters
        self.solver_options = solver_options
        self.df = pd.DataFrame()

    def loop(self, res_path: str = None):

        for param in self.Parameters.product_list:
            print("####################")
            print("Solve with")
            print(param)
            print("####################")

            biorbd_model_path = (
                param["biorbd_model_path"]
                if "biorbd_model_path" in param
                else self.Parameters.parameters_not_compared["biorbd_model_path"]
            )
            ode_solver = (
                param["ode_solver"] if "ode_solver" in param else self.Parameters.parameters_not_compared["ode_solver"]
            )
            n_shooting = (
                param["n_shooting"] if "n_shooting" in param else self.Parameters.parameters_not_compared["n_shooting"]
            )
            multibody_dynamics = (
                param["multibody_dynamics"]
                if "multibody_dynamics" in param
                else self.Parameters.parameters_not_compared["multibody_dynamics"]
            )
            tol = param["tolerance"] if "tolerance" in param else self.Parameters.parameters_not_compared["tolerance"]

            bo_file = filename(param)
            print(bo_file)

            CurOCP = self.Ocp(
                biorbd_model_path=biorbd_model_path,
                ode_solver=ode_solver,
                n_shooting=n_shooting,
                multibody_dynamics=multibody_dynamics,
            )
            cur_ocp = CurOCP.ocp
            add_custom_plots(cur_ocp)

            self.solver_options.set_convergence_tolerance(tol)
            self.solver_options.set_constraint_tolerance(tol)
            sol = cur_ocp.solve(self.solver_options)
            # filling dataframe
            consistency = compute_error_single_shooting(
                sol, cur_ocp.nlp[0].tf, integrator=SolutionIntegrator.SCIPY_DOP853
            )
            # continuity_consistency = compute_error_single_shooting(
            #     sol, cur_ocp.nlp[0].tf, integrator=None
            # )
            values_to_add = {
                "biorbd_model_path": biorbd_model_path,
                "ode_solver": ode_solver,
                "n_shooting": n_shooting,
                "tolerance": tol,
                "multibody_dynamics": multibody_dynamics,
                "iter": sol.iterations,
                "time": sol.real_time_to_optimize,
                "convergence": sol.status,
                "cost": np.squeeze(sol.cost.toarray()),
                "constraints": np.mean(abs(sol.constraints.toarray())),
                "constraints_RMSE": np.sqrt(np.mean(sol.constraints.toarray() ** 2)),
                "translation consistency": consistency[0],
                "angular consistency": consistency[1],
                "states_ss": integrate_sol(cur_ocp, sol),
                "controls": sol.controls["all"],
                "q": sol.states["q"],
                "qdot": sol.states["qdot"],
                "filename": bo_file,
            }

            # filling states
            if cur_ocp.nlp[0].ode_solver.is_direct_collocation:
                n = cur_ocp.nlp[0].ode_solver.polynomial_degree + 1
                values_to_add["states"] = sol.states["all"][:, ::n]
            else:
                values_to_add["states"] = sol.states["all"]

            self.df = self.df.append(values_to_add, ignore_index=True)

            if res_path is not None:
                cur_ocp.save(sol, f"{res_path}/{bo_file}", stand_alone=False)

    # def initial_guess(self):

    def graphs(
            self,
            first_parameter: str = "ode_solver",
            second_parameter: str = "n_shooting",
            third_parameter: str = "tolerance",
            fixed_parameters: dict = {},
            res_path: str = None,
            show: bool = True,
            figsize: tuple = (12, 12),
            tick_width: float = 0.2,
            dot_width: float = 0.03,
            size: int = 10,
            marker: str = "o",
            alpha: float = 1,
            markeredgewidth=0.1,
            markeredgecolor="black",
    ):
        def abscissa_computations(nb_first: int, nb_second: int, width: float = tick_width):
            x = np.arange(nb_first)
            ticks = np.zeros((nb_first, nb_second))
            for i in range(nb_second):
                ticks[:, i] = x + (-nb_second / 2 + 1 / 2 + i) * width
            return ticks

        def abscissa_offsets(ticks: np.array, nb_elements: int, offset_num: int, width: float = dot_width):
            return ticks + (-nb_elements / 2 + 1 / 2 + offset_num) * width

        all_param = [first_parameter, second_parameter, third_parameter]
        param_to_be_fixed = set(list(self.Parameters.parameters_compared.keys())) - set(all_param)
        param_left = param_to_be_fixed - set(list(fixed_parameters.keys()))

        if bool(param_left):
            raise ValueError(f" The parameters need to be set for {param_left}")

        # set values for varying parameters not considered in these graph
        df = self.df
        for param in fixed_parameters:
            df = self.df[df[param] == fixed_parameters[param]]

        n1 = self.Parameters.size(first_parameter)
        n2 = self.Parameters.size(second_parameter)
        n3 = self.Parameters.size(third_parameter)

        pal = sns.color_palette(palette="coolwarm", n_colors=n3)
        pal.reverse()

        first_parameter_labels = [i.__str__() for i in
                                  self.Parameters.get_parameter(first_parameter)] if n1 > 1 else [str(
            self.Parameters.get_parameter(first_parameter))]
        second_parameter_labels = [str(j) for j in
                                   self.Parameters.get_parameter(second_parameter)] * n1 if n2 > 1 else [str(
            self.Parameters.get_parameter(second_parameter))] * n1
        third_parameter_labels = list(map(str, self.Parameters.get_parameter(third_parameter))) if n3 > 1 else [str(
            self.Parameters.get_parameter(third_parameter))]

        args = ["time", "iter", "cost", "constraints", "translation consistency", "angular consistency"]
        args_y_label = [
            "time (s)",
            "iterations",
            "cost function value",
            "constraints",
            "Translation consistency (mm)",
            "Angular consistency (deg)",
        ]
        if n2 * tick_width >= 1:
            tick_width = 1 / n2 - 0.1
        x_ticks = abscissa_computations(n1, n2, width=tick_width)

        for i in range(len(args)):
            fig, ax = plt.subplots(1, 1, figsize=figsize)

            # Plot dots
            T = [self.Parameters.get_parameter(third_parameter)] if not isinstance(
                self.Parameters.get_parameter(third_parameter), list) else self.Parameters.get_parameter(
                third_parameter)
            for ii, i_3rd in enumerate(T):
                # Get elements in dataframe TODO: Exclude other varying conditions if any
                ddf = df[df[third_parameter] == i_3rd]
                D = ddf.pivot(index=first_parameter, columns=second_parameter, values=args[i])

                plt.plot(
                    abscissa_offsets(x_ticks, n3, ii, width=dot_width).T,
                    D.values.T,
                    marker,
                    markerfacecolor=pal[ii],
                    ms=size,
                    markeredgecolor=markeredgecolor,
                    markeredgewidth=markeredgewidth,
                    alpha=alpha,
                )

            AX = ax
            y_max = AX.get_ylim()

            # Abscissa labels
            AX.set_xticks(x_ticks.ravel())
            AX.set_xticklabels(second_parameter_labels)
            plt.setp(AX.get_xticklabels(), rotation=20)

            # Plot axis legends
            AX.set_ylabel(args_y_label[i])
            AX.set_xlabel(second_parameter)
            AX.set_yscale("log")

            # Plot titles of first parameters
            for ii in range(n1):
                t = AX.text(ii, y_max[1], first_parameter_labels[ii], ha="center", va="bottom", rotation=0, size=9)
            AX.spines["top"].set_visible(False)
            AX.spines["right"].set_visible(False)

            if np.min(y_max) > 0:
                AX.set_ylim(y_max)

            # Set vertical lines
            AX.vlines(np.arange(0, n1 - 1) + 0.5, ymin=0, ymax=y_max[1], color="black", ls="--")
            temp = np.arange(0, n1 + 1) - 0.5
            x_lim = (np.min(temp), np.max(temp))
            AX.set_xlim(x_lim)

            # Build the legend for third parameter out of the axes
            h = []
            for ii in range(n3):
                h_tol = AX.scatter(
                    [],
                    [],
                    c=np.array([pal[ii]]),
                    marker=marker,
                    s=size,
                )
                h.append(h_tol)
            Title = third_parameter
            Title = Title[0].upper() + Title[1:]
            plt.legend(
                handles=h,
                labels=third_parameter_labels,
                title=Title,
                loc=(1.04, 0),
            )

            plt.tight_layout()
            if res_path is not None:
                plt.savefig(f"{res_path}/{args[i]}_{first_parameter}_{second_parameter}_{third_parameter}.jpg")
        if show:
            plt.show()

    def graphs_time_series(
            self,
            first_parameter: str = "q",
            # second_parameter: str = "n_shooting",
            # third_parameter: str = "tolerance",
            # res_path: str = None,
            # show: bool = True,
            figsize: tuple = (12, 12),
            # tick_width: float = 0.2,
            # dot_width: float = 0.03,
            # size: int = 10,
            # marker: str = "o",
            # alpha: float = 1,
            # markeredgewidth=0.1,
            # markeredgecolor="black",
    ):

        n1 = len(self.Parameters.parameters_compared[first_parameter])
        n2 = len(self.Parameters.parameters_compared[second_parameter])
        n3 = len(self.Parameters.parameters_compared[third_parameter])

        pal = sns.color_palette(palette="coolwarm", n_colors=n3)
        pal.reverse()

        first_parameter_labels = [i.__str__() for i in self.Parameters.parameters_compared[first_parameter]]
        second_parameter_labels = [str(j) for j in self.Parameters.parameters_compared[second_parameter]] * n1
        third_parameter_labels = list(map(str, self.Parameters.parameters_compared[third_parameter]))

        args = ["time", "iter", "cost", "constraints", "translation consistency", "angular consistency"]
        args_y_label = [
            "time (s)",
            "iterations",
            "cost function value",
            "constraints",
            "Translation consistency (mm)",
            "Angular consistency (deg)",
        ]
        if n2 * tick_width >= 1:
            tick_width = 1 / n2 - 0.1
        x_ticks = abscissa_computations(n1, n2, width=tick_width)

        for i in range(len(args)):
            fig, ax = plt.subplots(1, 1, figsize=figsize)

            # Plot dots
            for ii, tol in enumerate(self.Parameters.parameters_compared[third_parameter]):
                # Get elements in dataframe TODO: Exclude other varying conditions if any
                ddf = self.df[self.df[third_parameter] == tol]
                D = ddf.pivot(index=first_parameter, columns=second_parameter, values=args[i])

                plt.plot(
                    abscissa_offsets(x_ticks, n3, ii, width=dot_width).T,
                    D.values.T,
                    marker,
                    markerfacecolor=pal[ii],
                    ms=size,
                    markeredgecolor=markeredgecolor,
                    markeredgewidth=markeredgewidth,
                    alpha=alpha,
                )

            AX = ax
            y_max = AX.get_ylim()

            # Abscissa labels
            AX.set_xticks(x_ticks.ravel())
            AX.set_xticklabels(second_parameter_labels)
            plt.setp(AX.get_xticklabels(), rotation=20)

            # Plot axis legends
            AX.set_ylabel(args_y_label[i])
            AX.set_xlabel(second_parameter)
            AX.set_yscale("log")

            # Plot titles of first parameters
            for ii in range(n1):
                t = AX.text(ii, y_max[1], first_parameter_labels[ii], ha="center", va="bottom", rotation=0, size=9)
            AX.spines["top"].set_visible(False)
            AX.spines["right"].set_visible(False)

            if np.min(y_max) > 0:
                AX.set_ylim(y_max)

            # Set vertical lines
            AX.vlines(np.arange(0, n1 - 1) + 0.5, ymin=0, ymax=y_max[1], color="black", ls="--")
            temp = np.arange(0, n1 + 1) - 0.5
            x_lim = (np.min(temp), np.max(temp))
            AX.set_xlim(x_lim)

            # Build the legend for third parameter out of the axes
            h = []
            for ii in range(n3):
                h_tol = AX.scatter(
                    [],
                    [],
                    c=np.array([pal[ii]]),
                    marker=marker,
                    s=size,
                )
                h.append(h_tol)
            Title = third_parameter
            Title = Title[0].upper() + Title[1:]
            plt.legend(
                handles=h,
                labels=third_parameter_labels,
                title=Title,
                loc=(1.04, 0),
            )

            plt.tight_layout()
            if res_path is not None:
                plt.savefig(f"{res_path}/{args[i]}_{first_parameter}_{second_parameter}_{third_parameter}.jpg")
        if show:
            plt.show()
