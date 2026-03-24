# -*- coding: utf-8 -*-
# @Time     : 2026/03/22
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


from typing import Dict, List, Optional, Sequence, Tuple

import pyomo.environ as pyo
from pyomo.repn.standard_repn import generate_standard_repn

from algo.algo_simple_tools import (
    build_cglp_lower_bound_row_from_alpha_eta,
    flip_prefix_tail,
    get_prefix_tuple,
    normalize_cglp_lower_bound_row,
    prefix_to_w_vector,
)
from model.model_base import ModelBase
from util.names import VarName


class ModelCGLP(ModelBase):
    """CGLP feasibility model used to generate optimality cuts from exact points."""

    def __init__(
        self,
        main_model,
        x_var_keys,
        z_var_keys,
        structural_constr_names,
        theta_keys,
        lower_bound_rows,
        model_name='cglp',
        exact_point_data=None,
    ):
        super().__init__(model_name=model_name)

        self.main_model = main_model
        self.x_var_keys = list(x_var_keys)
        self.z_var_keys = list(z_var_keys)
        self.structural_constr_names = list(structural_constr_names)
        self.theta_keys = list(theta_keys)

        self.n = len(self.x_var_keys)

        self.exact_point_data = [] if exact_point_data is None else list(exact_point_data)

        self.main_row_data = []

        self.lower_bound_row_dict = {}
        self.lower_bound_row_name_list = []
        self.psi_key_by_row_name = {}

        self.exact_point_order = []
        self.exact_point_value_dict = {}

        self.v_prefix_sets = {i: set() for i in range(1, self.n + 1)}
        self.hat_v_prefix_sets = {i: set() for i in range(1, self.n + 1)}
        self.active_w_row_name_dict = {i: {} for i in range(1, self.n + 1)}

        self.old_exact_point_row_name_list = []
        self.current_tight_row_name = None
        self.current_tight_point = None

        self.latest_alpha = None
        self.latest_eta = None

        self._alpha_var_name = 'alpha'
        self._eta_var_name = 'eta'
        self._sigma_var_name = 'sigma'
        self._rho_var_name = 'rho'
        self._varphi_var_name = 'varphi'
        self._nu_var_name = 'nu'
        self._psi_var_name = 'psi'

        self.sync_lower_bound_rows(lower_bound_rows)

    def build_model(self):
        """Build the static CGLP structure and preload any exact points passed at construction."""
        self._derive_main_row_data()
        self._build_static_vars()
        self._build_static_rows()

        if self.exact_point_data:
            for point_item in self.exact_point_data:
                if isinstance(point_item, dict):
                    x_point = point_item['x_point']
                    q_value = point_item['q_value']
                else:
                    x_point, q_value = point_item
                self.update_with_exact_point(x_point=x_point, q_value=q_value)

    def sync_lower_bound_rows(self, cut_rows):
        """Refresh the lower-bound cut block and rebuild the affected balance rows if needed."""
        self.lower_bound_row_dict = {}
        self.lower_bound_row_name_list = []

        for row in cut_rows:
            normalized_row = normalize_cglp_lower_bound_row(
                row=row,
                x_var_keys=self.x_var_keys,
            )
            row_name = normalized_row['name']
            self.lower_bound_row_dict[row_name] = normalized_row
            self.lower_bound_row_name_list.append(row_name)

            if self._psi_var_name in self.var:
                self._ensure_psi_var(row_name=row_name)

        if self.main_row_data:
            self._refresh_lower_bound_block_rows()

    def update_with_exact_point(self, x_point, q_value):
        """Add one exact epigraph point and update the incremental W-row state from Algorithm 5."""
        considered_point = tuple(int(v) for v in x_point)

        if considered_point in self.exact_point_value_dict:
            return

        for i in range(1, self.n + 1):
            current_v_i = self.v_prefix_sets[i]
            current_hat_v_i = self.hat_v_prefix_sets[i]

            bar_prefix = get_prefix_tuple(binary_point=considered_point, prefix_len=i)
            hat_prefix = flip_prefix_tail(prefix_tuple=bar_prefix)

            if hat_prefix not in current_hat_v_i and hat_prefix not in current_v_i:
                current_hat_v_i.add(hat_prefix)
                self._set_w_row(prefix_len=i, prefix_tuple=hat_prefix, is_active=True)

            if bar_prefix in current_hat_v_i:
                current_hat_v_i.remove(bar_prefix)
                self._set_w_row(prefix_len=i, prefix_tuple=bar_prefix, is_active=False)

            current_v_i.add(bar_prefix)

        if self.current_tight_row_name is not None:
            prev_q_value = self.exact_point_value_dict[self.current_tight_point]
            self.update_constr_by_name(
                self.current_tight_row_name,
                self._build_exact_point_expr(
                    x_point=self.current_tight_point,
                    q_value=prev_q_value,
                    use_equality=False,
                )
            )
            self.old_exact_point_row_name_list.append(self.current_tight_row_name)

        self.exact_point_order.append(considered_point)
        self.exact_point_value_dict[considered_point] = float(q_value)

        row_name = f'exact_point_{len(self.exact_point_order)}'
        self.add_constr(
            self._build_exact_point_expr(
                x_point=considered_point,
                q_value=q_value,
                use_equality=True,
            ),
            name=row_name
        )
        self.current_tight_row_name = row_name
        self.current_tight_point = considered_point

    def solve_cglp(self):
        """Solve the feasibility LP and return the cut coefficients extracted from the solution."""
        self.solve(param_dict={})
        return self.get_cut_coefficients()

    def get_cut_coefficients(self):
        """Read the current alpha and eta values from the solved CGLP model."""
        self.latest_alpha = {
            x_key: pyo.value(self.var[self._alpha_var_name][x_key])
            for x_key in self.x_var_keys
        }
        self.latest_eta = pyo.value(self.var[self._eta_var_name][0])
        return self.latest_alpha.copy(), self.latest_eta

    def build_group_cut_expr(self, model_main, alpha=None, eta=None):
        """Translate the solved CGLP coefficients into the master-model group cut expression."""
        if alpha is None or eta is None:
            alpha, eta = self.get_cut_coefficients()

        lhs = pyo.quicksum(
            model_main.var[VarName.SUB_OBJ_EST][theta_key]
            for theta_key in self.theta_keys
        )
        rhs = -eta - pyo.quicksum(
            alpha[var_key] * model_main.var[var_key[0]][var_key[1]]
            for var_key in self.x_var_keys
        )

        return lhs, rhs

    def solve_and_build_group_cut(self, model_main=None):
        """Solve the CGLP and return both the coefficients and the corresponding master cut."""
        self.solve_cglp()
        alpha, eta = self.get_cut_coefficients()

        if model_main is None:
            model_main = self.main_model

        lhs, rhs = self.build_group_cut_expr(
            model_main=model_main,
            alpha=alpha,
            eta=eta,
        )
        return alpha, eta, lhs, rhs

    def build_lower_bound_row(self, cut_name: str, alpha=None, eta=None) -> dict:
        """Translate cut coefficients into the normalized lower-bound row format used by persistent CGLP state."""
        if alpha is None or eta is None:
            alpha, eta = self.get_cut_coefficients()

        return build_cglp_lower_bound_row_from_alpha_eta(
            cut_name=cut_name,
            alpha=alpha,
            eta=eta,
        )

    def _build_static_vars(self):
        """Create the CGLP variables for cut coefficients, Farkas multipliers, and chain terms."""
        self.var[self._alpha_var_name] = {
            x_key: self.add_var(domain=pyo.Reals, name=f'{self._alpha_var_name}_{x_key}')
            for x_key in self.x_var_keys
        }

        self.var[self._eta_var_name] = {
            0: self.add_var(domain=pyo.Reals, name=self._eta_var_name)
        }

        self.var[self._sigma_var_name] = {
            x_key: self.add_var(domain=pyo.Reals, name=f'{self._sigma_var_name}_{x_key}')
            for x_key in self.x_var_keys
        }

        self.var[self._rho_var_name] = {
            i: self.add_var(domain=pyo.Reals, name=f'{self._rho_var_name}_{i}')
            for i in range(1, self.n + 1)
        }

        self.var[self._varphi_var_name] = {
            i: self.add_var(domain=pyo.NonNegativeReals, name=f'{self._varphi_var_name}_{i}')
            for i in range(2, self.n + 1)
        }

        self.var[self._nu_var_name] = {
            row_idx: self.add_var(domain=pyo.NonNegativeReals, name=f'{self._nu_var_name}_{row_idx}')
            for row_idx in range(len(self.main_row_data))
        }

        self.var[self._psi_var_name] = {}
        for row_name in self.lower_bound_row_name_list:
            self._ensure_psi_var(row_name=row_name)

    def _build_static_rows(self):
        """Add the static CGLP feasibility rows that do not depend on new exact points."""
        for x_key in self.x_var_keys:
            self.add_constr(
                self._build_alpha_balance_expr(x_key=x_key),
                name=self._get_alpha_balance_row_name(x_key=x_key)
            )

        self.add_constr(
            self._build_psi_normalization_expr(),
            name='psi_normalization'
        )

        self.add_constr(
            self._build_eta_balance_expr(),
            name='eta_balance'
        )

        for z_key in self.z_var_keys:
            self.add_constr(
                pyo.quicksum(
                    self.main_row_data[row_idx]['z_coef'][z_key] * self.var[self._nu_var_name][row_idx]
                    for row_idx in range(len(self.main_row_data))
                ) >= 0,
                name=f'z_balance_{z_key}'
            )

        for i in range(2, self.n + 1):
            x_key = self.x_var_keys[i - 1]
            self.add_constr(
                self.var[self._sigma_var_name][x_key] + self.var[self._varphi_var_name][i] >= 0,
                name=f'sigma_chain_{i}'
            )

        for i in range(1, self.n):
            self.add_constr(
                -self.var[self._rho_var_name][i]
                + self.var[self._rho_var_name][i + 1]
                - self.var[self._varphi_var_name][i + 1] >= 0,
                name=f'rho_chain_{i}'
            )

        self.set_objective(0.0, sense=pyo.minimize)

    def _derive_main_row_data(self):
        """Extract master structural rows into dense x/z coefficient dictionaries for the CGLP."""
        self.main_row_data = []

        x_var_id_map = {
            id(self.main_model.var[var_name][var_key]): (var_name, var_key)
            for var_name, var_key in self.x_var_keys
        }
        z_var_id_map = {
            id(self.main_model.var[var_name][var_key]): (var_name, var_key)
            for var_name, var_key in self.z_var_keys
        }

        for constr_name in self.structural_constr_names:
            constr_item = self.main_model.get_constr_item_by_name(constr_name)

            if constr_item.upper is not None:
                repn = generate_standard_repn(
                    constr_item.body - constr_item.upper,
                    compute_values=True,
                )
                x_coef = {x_key: 0.0 for x_key in self.x_var_keys}
                z_coef = {z_key: 0.0 for z_key in self.z_var_keys}
                if repn.linear_vars is not None:
                    for linear_var, linear_coef in zip(repn.linear_vars, repn.linear_coefs):
                        var_id = id(linear_var)
                        coef_value = float(pyo.value(linear_coef))
                        if var_id in x_var_id_map:
                            x_coef[x_var_id_map[var_id]] += coef_value
                        elif var_id in z_var_id_map:
                            z_coef[z_var_id_map[var_id]] += coef_value
                self.main_row_data.append(
                    {
                        'x_coef': x_coef,
                        'z_coef': z_coef,
                        'rhs': -(0.0 if repn.constant is None else float(pyo.value(repn.constant))),
                    }
                )

            if constr_item.lower is not None:
                repn = generate_standard_repn(
                    -constr_item.body + constr_item.lower,
                    compute_values=True,
                )
                x_coef = {x_key: 0.0 for x_key in self.x_var_keys}
                z_coef = {z_key: 0.0 for z_key in self.z_var_keys}
                if repn.linear_vars is not None:
                    for linear_var, linear_coef in zip(repn.linear_vars, repn.linear_coefs):
                        var_id = id(linear_var)
                        coef_value = float(pyo.value(linear_coef))
                        if var_id in x_var_id_map:
                            x_coef[x_var_id_map[var_id]] += coef_value
                        elif var_id in z_var_id_map:
                            z_coef[z_var_id_map[var_id]] += coef_value
                self.main_row_data.append(
                    {
                        'x_coef': x_coef,
                        'z_coef': z_coef,
                        'rhs': -(0.0 if repn.constant is None else float(pyo.value(repn.constant))),
                    }
                )

    def _ensure_psi_var(self, row_name):
        """Create one nonnegative psi multiplier for a lower-bound row the first time it appears."""
        if row_name in self.psi_key_by_row_name:
            return self.var[self._psi_var_name][self.psi_key_by_row_name[row_name]]

        psi_key = len(self.psi_key_by_row_name)
        self.psi_key_by_row_name[row_name] = psi_key
        self.var[self._psi_var_name][psi_key] = self.add_var(
            domain=pyo.NonNegativeReals,
            name=f'{self._psi_var_name}_{psi_key}'
        )
        return self.var[self._psi_var_name][psi_key]

    def _refresh_lower_bound_block_rows(self):
        """Refresh the rows whose coefficients depend on the current lower-bound cut collection."""
        for x_key in self.x_var_keys:
            self.update_constr_by_name(
                self._get_alpha_balance_row_name(x_key=x_key),
                self._build_alpha_balance_expr(x_key=x_key)
            )

        self.update_constr_by_name(
            'psi_normalization',
            self._build_psi_normalization_expr()
        )

        self.update_constr_by_name(
            'eta_balance',
            self._build_eta_balance_expr()
        )

    def _build_alpha_balance_expr(self, x_key):
        """Build the alpha row that couples cut coefficients with master and lower-bound multipliers."""
        return (
            self.var[self._alpha_var_name][x_key]
            - self.var[self._sigma_var_name][x_key]
            + pyo.quicksum(
                self.main_row_data[row_idx]['x_coef'][x_key] * self.var[self._nu_var_name][row_idx]
                for row_idx in range(len(self.main_row_data))
            )
            + pyo.quicksum(
                self.lower_bound_row_dict[row_name]['x_coef'][x_key]
                * self.var[self._psi_var_name][self.psi_key_by_row_name[row_name]]
                for row_name in self.lower_bound_row_name_list
            )
        ) == 0

    def _build_psi_normalization_expr(self):
        """Build the normalization row that keeps the psi multipliers on the unit simplex."""
        return pyo.quicksum(
            self.var[self._psi_var_name][self.psi_key_by_row_name[row_name]]
            for row_name in self.lower_bound_row_name_list
        ) == 1

    def _build_eta_balance_expr(self):
        """Build the eta row that aggregates RHS terms from master rows and lower-bound cuts."""
        return (
            -self.var[self._rho_var_name][self.n]
            + self.var[self._eta_var_name][0]
            - pyo.quicksum(
                self.main_row_data[row_idx]['rhs'] * self.var[self._nu_var_name][row_idx]
                for row_idx in range(len(self.main_row_data))
            )
            - pyo.quicksum(
                self.lower_bound_row_dict[row_name]['rhs']
                * self.var[self._psi_var_name][self.psi_key_by_row_name[row_name]]
                for row_name in self.lower_bound_row_name_list
            )
        ) >= 0

    def _build_exact_point_expr(self, x_point, q_value, use_equality):
        """Build the exact-point row, using equality for the newest point and inequality for older ones."""
        expr_lhs = pyo.quicksum(
            x_point[idx] * self.var[self._alpha_var_name][self.x_var_keys[idx]]
            for idx in range(self.n)
        ) + self.var[self._eta_var_name][0]

        if use_equality:
            return expr_lhs == -float(q_value)
        return expr_lhs >= -float(q_value)

    def _set_w_row(self, prefix_len, prefix_tuple, is_active):
        """Add, refresh, or remove one active W-row induced by the current exact-point prefixes."""
        row_name = f'w_row_{prefix_len}_{prefix_tuple}'
        if not is_active:
            if row_name in self._constr_name_map:
                self.remove_constr_by_name(row_name)
            self.active_w_row_name_dict[prefix_len].pop(prefix_tuple, None)
            return

        w_vector = prefix_to_w_vector(prefix_tuple=prefix_tuple, total_dim=self.n)
        expr = pyo.quicksum(
            w_vector[idx] * self.var[self._sigma_var_name][self.x_var_keys[idx]]
            for idx in range(self.n)
        ) + self.var[self._rho_var_name][prefix_len] >= 0

        if row_name in self._constr_name_map:
            self.update_constr_by_name(row_name, expr)
        else:
            self.add_constr(expr, name=row_name)

        self.active_w_row_name_dict[prefix_len][prefix_tuple] = row_name

    @staticmethod
    def _get_alpha_balance_row_name(x_key):
        """Return the stable row name for one alpha balance equation."""
        return f'alpha_balance_{x_key}'
