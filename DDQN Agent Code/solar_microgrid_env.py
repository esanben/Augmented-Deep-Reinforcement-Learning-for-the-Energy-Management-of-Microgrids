"""
solar_microgrid_env.py
----------------------
Standalone environment module for the Solar Microgrid simulation.
"""

from collections import defaultdict


class SolarMicrogridEnv:
    def __init__(self, load_data, solar_data, price_data, DG1_space, DG2_space,
                 DG3_space, PGrid_space, DG1_status, DG2_status, DG3_status, Batt_Power_space):
        # State space definitions
        self.state_size = 4

        # Action space definitions
        self.action_space = [
            (a, b, c, d, e, f, g, h)
            for a in DG1_space
            for b in DG1_status
            for c in DG2_space
            for d in DG2_status
            for e in DG3_space
            for f in DG3_status
            for g in PGrid_space
            for h in Batt_Power_space
        ]
        self.action_size = len(self.action_space)

        # Battery parameters
        self.bess_eff    = 0.9   # Battery efficiency 90%
        self.bess_max    = 1.5   # Upper bound for battery (1.5 MWh)
        self.bess_min    = 0.3   # Lower bound for battery (0.3 MWh)
        self.batt_init   = 0.3   # Initial battery state (MWh)
        self.ch_dis_max  = 0.5   # Max charge/discharge rate (MW)

        # Diesel generator cost coefficients
        self.coef_a1 = 0.01; self.coef_a2 = 0.02; self.coef_a3 = 0.03
        self.coef_b1 = 0.55; self.coef_b2 = 0.75; self.coef_b3 = 0.85
        self.coef_c1 = 1;    self.coef_c2 = 2.5;  self.coef_c3 = 3.3

        # Ramp limits for the 3 generators
        self.ramp_limit_dg1 = 1.0
        self.ramp_limit_dg2 = 1.2
        self.ramp_limit_dg3 = 1.5

        # Time-series data
        self.load_data  = load_data
        self.solar_data = solar_data
        self.price_data = price_data

        # Initial state: [solar, load, price, battery_SoC]
        self.state = [self.solar_data[0], self.load_data[0], self.price_data[0], self.batt_init]
        self.current_time_step = 0

        # Battery degradation cost parameters
        self.IC_bess    = 80 * self.bess_max
        self.LCN        = 3000
        self.bess_coef  = self.IC_bess / (self.bess_max * self.LCN)

        # Solar PV O&M cost
        self.pv_omcosts = 0.7  # $/kWh

        # Generator state tracking (min up/down time)
        self.generators = {
            1: {'last_on': None, 'last_off': None, 'p_on': 0, 'p_off': 0,'prev_status': 1, 'prev_statuses': [], 'prev_last_on': None, 'prev_last_off': None},
            2: {'last_on': None, 'last_off': None, 'p_on': 0, 'p_off': 0,'prev_status': 1, 'prev_statuses': [], 'prev_last_on': None, 'prev_last_off': None},
            3: {'last_on': None, 'last_off': None, 'p_on': 0, 'p_off': 0,'prev_status': 1, 'prev_statuses': [], 'prev_last_on': None, 'prev_last_off': None},
        }

        # Startup / shutdown cost parameters
        self.startup_shutdown    = defaultdict(int)
        self.last_status         = []
        self.start_cost_g1_hot   = 0.2; self.start_cost_g1_cold = 0.5
        self.start_cost_g2_hot   = 0.3; self.start_cost_g2_cold = 0.6
        self.start_cost_g3_hot   = 0.4; self.start_cost_g3_cold = 0.7


    def step(self, t, init_prod_dg1, init_prod_dg2, init_prod_dg3,
             dg1_action, dg2_action, dg3_action, pgrid_action,
             dg1_status_action, dg2_status_action, dg3_status_action,
             decision_pbatt, soc_prev):

        solar = self.solar_data[self.current_time_step]
        load  = self.load_data[self.current_time_step]
        price = self.price_data[self.current_time_step]

        # Initialise all penalty buckets
        penalty_1 = penalty_2 = penalty_3 = penalty_4  = penalty_5  = penalty_6  = 0
        penalty_7 = penalty_8 = penalty_9 = penalty_10 = penalty_11 = penalty_12 = 0
        p1_on = p2_on = p3_on = p1_off = p2_off = p3_off = 0

        # ---- MIN UP / DOWN CONSTRAINTS ----
        for gen in [1, 2, 3]:
            self.generators[gen]['p_on']  = 0
            self.generators[gen]['p_off'] = 0

        for gen, status in [(1, dg1_status_action), (2, dg2_status_action), (3, dg3_status_action)]:
            prev_statuses  = self.generators[gen]['prev_statuses']
            prev_last_on   = self.generators[gen]['prev_last_on']
            prev_last_off  = self.generators[gen]['prev_last_off']
            prev_status    = self.generators[gen]['prev_status']

            prev_statuses.append(status)

            if status == 1:
                self.generators[gen]['last_on'] = t
                if self.generators[gen]['last_off'] is None or t - self.generators[gen]['last_off'] == 2:
                    self.generators[gen]['p_on'] = 0
                elif (prev_status == 0) and (prev_statuses[-4:-1] != [0, 0, 0]):
                    self.generators[gen]['p_on'] = 500
                elif (len(prev_statuses) >= 3) and (prev_statuses[-4:-1] == [0, 0, 0]) and (prev_last_on is None):
                    self.generators[gen]['p_on'] = 0
                elif (len(prev_statuses) >= 3) and (prev_statuses[-4:-1] == [0, 0, 0]) and (prev_last_off - prev_last_on >= 3):
                    self.generators[gen]['p_on'] = 0
                elif (prev_status == 1) and (prev_statuses[-3:] == [1, 1, 1]):
                    self.generators[gen]['p_on'] = 0
                else:
                    self.generators[gen]['p_on'] = 500

            if status == 0:
                self.generators[gen]['last_off'] = t
                if self.generators[gen]['last_on'] is None or t - self.generators[gen]['last_on'] == 2:
                    self.generators[gen]['p_off'] = 0
                elif (prev_status == 1) and (prev_statuses[-4:-1] != [1, 1, 1]):
                    self.generators[gen]['p_off'] = 500
                elif len(prev_statuses) >= 3 and prev_statuses[-4:-1] == [1, 1, 1] and (prev_last_off is None):
                    self.generators[gen]['p_off'] = 0
                elif len(prev_statuses) >= 3 and prev_statuses[-4:-1] == [1, 1, 1] and (prev_last_on - prev_last_off >= 3):
                    self.generators[gen]['p_off'] = 0
                elif (prev_status == 0) and (prev_statuses[-3:] == [0, 0, 0]):
                    self.generators[gen]['p_off'] = 0
                else:
                    self.generators[gen]['p_off'] = 500

            self.generators[gen]['prev_last_on']  = self.generators[gen]['last_on']
            self.generators[gen]['prev_last_off'] = self.generators[gen]['last_off']
            self.generators[gen]['prev_status']   = status

        p1_on  = self.generators[1]['p_on'];  p1_off = self.generators[1]['p_off']
        p2_on  = self.generators[2]['p_on'];  p2_off = self.generators[2]['p_off']
        p3_on  = self.generators[3]['p_on'];  p3_off = self.generators[3]['p_off']
        G1_l_on  = self.generators[1]['last_on'];  G1_l_off = self.generators[1]['last_off']
        G2_l_on  = self.generators[2]['last_on'];  G2_l_off = self.generators[2]['last_off']
        G3_l_on  = self.generators[3]['last_on'];  G3_l_off = self.generators[3]['last_off']

        # ---- STARTUP & SHUTDOWN COSTS ----
        generators = [1, 2, 3]
        min_down_time_threshold = 3

        if not hasattr(self, 'down_time'):
            self.down_time = {gen: 0 for gen in generators}

        dg_status_actions = {1: dg1_status_action, 2: dg2_status_action, 3: dg3_status_action}
        init_prods        = {1: init_prod_dg1,       2: init_prod_dg2,       3: init_prod_dg3}

        if t == 0:
            for gen in generators:
                if init_prods[gen] > 0:
                    if dg_status_actions[gen] == 1:
                        self.startup_shutdown[f'startgen_hot{gen}']  = 0
                        self.startup_shutdown[f'startgen_cold{gen}'] = 0
                        self.startup_shutdown[f'shut_gen{gen}']      = 0
                        self.down_time[gen] = 0
                    else:
                        self.startup_shutdown[f'startgen_hot{gen}']  = 0
                        self.startup_shutdown[f'startgen_cold{gen}'] = 0
                        self.startup_shutdown[f'shut_gen{gen}']      = 1
                        self.down_time[gen] += 1
                else:
                    if dg_status_actions[gen] == 1:
                        self.startup_shutdown[f'startgen_cold{gen}'] = 1
                        self.startup_shutdown[f'startgen_hot{gen}']  = 0
                        self.startup_shutdown[f'shut_gen{gen}']      = 0
                        self.down_time[gen] = 0
                    else:
                        self.startup_shutdown[f'startgen_cold{gen}'] = 0
                        self.startup_shutdown[f'startgen_hot{gen}']  = 0
                        self.startup_shutdown[f'shut_gen{gen}']      = 0
                        self.down_time[gen] += 1

            self.last_status = [dg1_status_action, dg2_status_action, dg3_status_action].copy()
        else:
            for gen in generators:
                current_status = dg_status_actions[gen]
                if current_status == 1:
                    if self.last_status[gen - 1] == 0:
                        if self.down_time[gen] > min_down_time_threshold:
                            self.startup_shutdown[f'startgen_cold{gen}'] = 1
                            self.startup_shutdown[f'startgen_hot{gen}']  = 0
                        else:
                            self.startup_shutdown[f'startgen_hot{gen}']  = 1
                            self.startup_shutdown[f'startgen_cold{gen}'] = 0
                        self.down_time[gen] = 0
                    else:
                        self.startup_shutdown[f'startgen_hot{gen}']  = 0
                        self.startup_shutdown[f'startgen_cold{gen}'] = 0
                    self.startup_shutdown[f'shut_gen{gen}'] = 0
                elif current_status == 0:
                    if self.last_status[gen - 1] == 1:
                        self.startup_shutdown[f'shut_gen{gen}'] = 1
                    else:
                        self.startup_shutdown[f'shut_gen{gen}'] = 0
                    self.down_time[gen] += 1

            self.last_status = [dg1_status_action, dg2_status_action, dg3_status_action].copy()

        start_gen_1_hot  = self.startup_shutdown['startgen_hot1']
        start_gen_1_cold = self.startup_shutdown['startgen_cold1']
        shut_gen_1       = self.startup_shutdown['shut_gen1']
        start_gen_2_hot  = self.startup_shutdown['startgen_hot2']
        start_gen_2_cold = self.startup_shutdown['startgen_cold2']
        shut_gen_2       = self.startup_shutdown['shut_gen2']
        start_gen_3_hot  = self.startup_shutdown['startgen_hot3']
        start_gen_3_cold = self.startup_shutdown['startgen_cold3']
        shut_gen_3       = self.startup_shutdown['shut_gen3']

        # ---- GENERATOR PENALTIES ----
        penalty_1 = 500 if (dg1_status_action == 0 and dg1_action > 0) or (dg1_status_action == 1 and dg1_action == 0) else 0
        penalty_2 = 500 if (dg2_status_action == 0 and dg2_action > 0) or (dg2_status_action == 1 and dg2_action == 0) else 0
        penalty_3 = 500 if (dg3_status_action == 0 and dg3_action > 0) or (dg3_status_action == 1 and dg3_action == 0) else 0

        # Ramp limit penalties
        penalty_4 = 500 if abs(dg1_action - init_prod_dg1) > self.ramp_limit_dg1 else 0
        penalty_5 = 500 if abs(dg2_action - init_prod_dg2) > self.ramp_limit_dg2 else 0
        penalty_6 = 500 if abs(dg3_action - init_prod_dg3) > self.ramp_limit_dg3 else 0

        # ---- BATTERY ----
        if decision_pbatt > 0:
            u_batt = 1
            next_batt_state = soc_prev + (self.bess_eff * u_batt * decision_pbatt)
        elif decision_pbatt < 0:
            u_batt = 0
            next_batt_state = soc_prev + (((1 - u_batt) * decision_pbatt) / self.bess_eff)
        else:
            u_batt = 1
            next_batt_state = soc_prev

        penalty_7 = 500 if (next_batt_state < self.bess_min) or (next_batt_state > self.bess_max) else 0

        # ---- POWER BALANCE ----
        power_source_wt_pgrid_pbat = (
            solar
            + (dg1_status_action * dg1_action)
            + (dg2_status_action * dg2_action)
            + (dg3_status_action * dg3_action)
            + pgrid_action
            - decision_pbatt
        )
        penalty_8 = 0    if power_source_wt_pgrid_pbat >= load else 1000
        penalty_9 = 1000 if (power_source_wt_pgrid_pbat < load) and (decision_pbatt == 0) else 0

        if t == 0:
            penalty_10 = 500 if abs(dg1_action - init_prod_dg1) > self.ramp_limit_dg1 else 0
            penalty_11 = 500 if abs(dg2_action - init_prod_dg2) > self.ramp_limit_dg2 else 0
            penalty_12 = 500 if abs(dg3_action - init_prod_dg3) > self.ramp_limit_dg3 else 0

        penalty_total = [
            penalty_1, penalty_2, penalty_3,
            penalty_4, penalty_5, penalty_6,
            penalty_7, penalty_8, penalty_9,
            penalty_10, penalty_11, penalty_12,
            p1_on, p2_on, p3_on, p1_off, p2_off, p3_off,
        ]

        reward, obj, all_diff_costs = self.calculate_reward(
            dg1_action, dg2_action, dg3_action, pgrid_action,
            dg1_status_action, dg2_status_action, dg3_status_action,
            price, penalty_total, decision_pbatt, next_batt_state, solar,
            start_gen_1_hot, start_gen_1_cold, shut_gen_1,
            start_gen_2_hot, start_gen_2_cold, shut_gen_2,
            start_gen_3_hot, start_gen_3_cold, shut_gen_3,
        )

        done = False
        if self.current_time_step < len(self.solar_data):
            next_solar = self.solar_data[self.current_time_step]
            next_load  = self.load_data[self.current_time_step]
            next_price = self.price_data[self.current_time_step]
            next_state = [next_solar, next_load, next_price, next_batt_state]
        else:
            next_solar = self.solar_data[0]
            next_load  = self.load_data[0]
            next_price = self.price_data[0]
            next_state = [next_solar, next_load, next_price, self.batt_init]
            done = True

        self.current_time_step += 1
        return next_state, reward, obj, all_diff_costs, penalty_total, done

    # ------------------------------------------------------------------
    # calculate_reward
    # ------------------------------------------------------------------
    def calculate_reward(self, dg1_action, dg2_action, dg3_action, pgrid_action,
                         dg1_status_action, dg2_status_action, dg3_status_action,
                         price, penalty_total, decision_pbatt, next_batt_state, solar,
                         start_gen_1_hot, start_gen_1_cold, shut_gen_1,
                         start_gen_2_hot, start_gen_2_cold, shut_gen_2,
                         start_gen_3_hot, start_gen_3_cold, shut_gen_3):
        """Compute operating costs, penalties, and reward."""

        k1 = 0.1; k2 = 0.1; k3 = 0.1; k4 = 0.1; k5 = 0.6

        cost_dg1 = (
            dg1_status_action * ((self.coef_a1 * dg1_action ** 2) + (self.coef_b1 * dg1_action) + self.coef_c1)
            + (start_gen_1_hot  * self.start_cost_g1_hot)
            + (start_gen_1_cold * self.start_cost_g1_cold)
        )
        cost_dg2 = (
            dg2_status_action * ((self.coef_a2 * dg2_action ** 2) + (self.coef_b2 * dg2_action) + self.coef_c2)
            + (start_gen_2_hot  * self.start_cost_g2_hot)
            + (start_gen_2_cold * self.start_cost_g2_cold)
        )
        cost_dg3 = (
            dg3_status_action * ((self.coef_a3 * dg3_action ** 2) + (self.coef_b3 * dg3_action) + self.coef_c3)
            + (start_gen_3_hot  * self.start_cost_g3_hot)
            + (start_gen_3_cold * self.start_cost_g3_cold)
        )
        cost_dg       = cost_dg1 + cost_dg2 + cost_dg3
        cost_pgrid    = price * pgrid_action
        bess_degrad   = (self.bess_coef * abs(decision_pbatt)) + (self.bess_coef * next_batt_state)
        pv_om_costs   = self.pv_omcosts * solar
        penalties     = sum(penalty_total)

        all_diff_costs = (cost_dg, cost_pgrid, bess_degrad, pv_om_costs)
        obj    = cost_dg + cost_pgrid + bess_degrad + pv_om_costs
        r      = (k1 * cost_dg) + (k2 * cost_pgrid) + (k3 * bess_degrad) + (k4 * pv_om_costs) + (k5 * penalties)
        reward = -1 * r

        return reward, obj, all_diff_costs

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self):
        """Reset environment to the initial state."""
        self.state = [self.solar_data[0], self.load_data[0], self.price_data[0], self.batt_init]
        self.current_time_step = 0
        return self.state
