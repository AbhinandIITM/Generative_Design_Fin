import os
import numpy as np
import matplotlib.pyplot as plt

class AUVSimulator:
    """
    Research-grade 3-DOF Maneuvering Simulator based on Fossen's equations.
    Models added mass and cross-flow drag via strip theory along the hull profile.
    """
    def __init__(self, D=0.166, L_nose=0.12, L_mid=0.3008, L_tail=0.1272, rho=1000.0):
        # Geometry
        self.D = D
        self.R = D / 2.0
        self.L_nose = L_nose
        self.L_mid = L_mid
        self.L_tail = L_tail
        self.L_total = L_nose + L_mid + L_tail
        self.rho = rho
        
        # Actuator parameters
        self.A_rudder = 0.005  # m^2, movable rudder area
        self.x_rudder = -self.L_total / 2.0 + 0.05 # location of rudder from CG
        self.CL_alpha = 3.0 # Lift curve slope
        
        # Discretize hull for integration (strip theory)
        self.dx = 0.005
        self.x_stations = np.arange(-self.L_total/2, self.L_total/2, self.dx)
        self.r_stations = np.array([self.radius_at_x(x) for x in self.x_stations])
        
        # Calculate mass and inertia based on exact varying profile
        self.volume = np.sum(np.pi * self.r_stations**2 * self.dx)
        self.m = self.rho * self.volume
        self.I_z = np.sum(self.rho * np.pi * self.r_stations**2 * self.x_stations**2 * self.dx)
        self.x_G = 0.0 
        
        self.prop_thrust = 0.0
        self.compute_derivatives()
        
    def radius_at_x(self, x):
        """ Get the radius of the AUV at a given axial coordinate """
        xt = x + self.L_total / 2.0
        if xt < self.L_tail:
            return self.R * (xt / self.L_tail)
        elif xt < self.L_tail + self.L_mid:
            return self.R
        else:
            xn = self.L_total - xt
            return self.R * np.sqrt(1 - (1 - xn/self.L_nose)**2) if xn < self.L_nose else self.R
            
    def compute_derivatives(self):
        # 1. Added Mass (Strip theory approx integrating along hull)
        self.X_udot = -0.1 * self.m
        self.Y_vdot = -self.rho * np.pi * np.sum(self.r_stations**2 * self.dx)
        self.N_rdot = -self.rho * np.pi * np.sum(self.x_stations**2 * self.r_stations**2 * self.dx)
        self.Y_rdot = -self.rho * np.pi * np.sum(self.x_stations * self.r_stations**2 * self.dx)
        self.N_vdot = self.Y_rdot
        
        # 2. Linear Damping (skin friction & lift)
        self.X_u = -0.5 * self.rho * (np.pi * self.R**2) * 0.1 
        
        # Linear sway and yaw damping (Munk moment lift approximations)
        self.Y_v = -0.5 * self.rho * self.L_total * self.D * 0.1
        self.Y_r =  0.5 * self.rho * self.L_total**2 * self.D * 0.02
        self.N_v =  0.5 * self.rho * self.L_total**2 * self.D * 0.02
        self.N_r = -0.5 * self.rho * self.L_total**3 * self.D * 0.05
        
        # 3. Fin/Rudder Lift (Linearized)
        self.Y_uv_fin = -0.5 * self.rho * self.A_rudder * self.CL_alpha
        self.Y_ur_fin = -0.5 * self.rho * self.A_rudder * self.CL_alpha * self.x_rudder
        self.N_uv_fin = self.Y_uv_fin * self.x_rudder
        self.N_ur_fin = self.Y_ur_fin * self.x_rudder
        
        self.Y_v += self.Y_uv_fin 
        self.Y_r += self.Y_ur_fin
        self.N_v += self.N_uv_fin
        self.N_r += self.N_ur_fin
        
        self.Y_delta = 0.5 * self.rho * self.A_rudder * self.CL_alpha
        self.N_delta = self.Y_delta * self.x_rudder
        
        # Quadratic cross-flow drag coefficient (cylinder)
        self.C_dc = 1.1

    def solve_accelerations(self, u, v, r, delta):
        """ Solves M*v_dot + C*v + D*v = tau for accelerations """
        # Munk moments / Coriolis (Added mass terms)
        C_A_Y = -self.X_udot * u
        C_A_N = (self.X_udot - self.Y_vdot) * u * v
        
        # Quadratic Cross-Flow Drag Integration along hull
        # F_drag = -0.5 * rho * C_d * Area * V_local^2
        U_local = v + self.x_stations * r
        dY_q = -0.5 * self.rho * self.C_dc * (2 * self.r_stations) * np.abs(U_local) * U_local * self.dx
        Y_q = np.sum(dY_q)
        N_q = np.sum(self.x_stations * dY_q)
        X_q = -0.5 * self.rho * (np.pi * self.R**2) * 0.15 * u * abs(u) 
        
        # Linear Damping + Fins
        X_l = self.X_u * u
        Y_l = self.Y_v * u * v + self.Y_r * u * r
        N_l = self.N_v * u * v + self.N_r * u * r
        
        # Rudder Force
        Y_d = self.Y_delta * (u**2) * delta
        N_d = self.N_delta * (u**2) * delta
        
        # Thrust to maintain constant u=1.0 m/s initially
        if self.prop_thrust == 0.0:
            self.prop_thrust = -(self.X_u * 1.0 + -0.5 * self.rho * (np.pi * self.R**2) * 0.15 * 1.0 * abs(1.0))
            
        X_force = X_l + X_q + self.prop_thrust
        Y_force = Y_l + Y_q + Y_d + C_A_Y * r - self.m * u * r
        N_force = N_l + N_q + N_d + C_A_N
        
        # M Matrix components
        M11 = self.m - self.X_udot
        M22 = self.m - self.Y_vdot
        M23 = self.m * self.x_G - self.Y_rdot
        M32 = self.m * self.x_G - self.N_vdot
        M33 = self.I_z - self.N_rdot
        
        # Solve for accelerations
        u_dot = X_force / M11
        det = M22 * M33 - M23 * M32
        v_dot = (M33 * Y_force - M23 * N_force) / det
        r_dot = (-M32 * Y_force + M22 * N_force) / det
        
        return u_dot, v_dot, r_dot


# ============================================================
# RK4 Integrator for robustness with discontinuous inputs
# ============================================================
def rk4_step(auv, state, delta, dt):
    def get_derivs(st):
        u, v, r, x, y, psi = st
        u_dot, v_dot, r_dot = auv.solve_accelerations(u, v, r, delta)
        x_dot = u * np.cos(psi) - v * np.sin(psi)
        y_dot = u * np.sin(psi) + v * np.cos(psi)
        psi_dot = r
        return np.array([u_dot, v_dot, r_dot, x_dot, y_dot, psi_dot])
        
    k1 = get_derivs(state)
    k2 = get_derivs(state + 0.5 * dt * k1)
    k3 = get_derivs(state + 0.5 * dt * k2)
    k4 = get_derivs(state + dt * k3)
    
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ============================================================
# Maneuvering Tests
# ============================================================
def run_turning_circle(auv, t_end=100.0, dt=0.05):
    t_vals = np.arange(0, t_end, dt)
    state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # [u, v, r, x, y, psi]
    
    history = [state]
    for t in t_vals[:-1]:
        delta = 0.0 if t < 5.0 else np.radians(20.0)
        state = rk4_step(auv, state, delta, dt)
        history.append(state)
        
    return t_vals, np.array(history)

def run_zigzag(auv, deg=20.0, t_end=100.0, dt=0.05):
    t_vals = np.arange(0, t_end, dt)
    state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    delta_cmd = deg
    target_psi = deg
    
    history = [state]
    delta_history = [0.0]
    
    for t in t_vals[:-1]:
        psi = np.degrees(state[5])
        
        if t < 5.0:
            delta = 0.0
        else:
            # Switch logic
            if delta_cmd > 0 and psi >= target_psi:
                delta_cmd = -deg
                target_psi = -deg
            elif delta_cmd < 0 and psi <= target_psi:
                delta_cmd = deg
                target_psi = deg
            delta = np.radians(delta_cmd)
            
        state = rk4_step(auv, state, delta, dt)
        history.append(state)
        delta_history.append(delta)
        
    return t_vals, np.array(history), np.array(delta_history)

# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":
    output_folder = "plots"
    os.makedirs(output_folder, exist_ok=True)
    
    diameters = [0.150, 0.160, 0.166, 0.175, 0.182]
    turning_radii = []
    
    plt.figure(figsize=(10, 8))
    
    for D in diameters:
        auv = AUVSimulator(D=D)
        t, hist = run_turning_circle(auv)
        
        x_traj = hist[:, 3]
        y_traj = hist[:, 4]
        
        # Calculate steady metrics
        u_steady = hist[-1, 0]
        v_steady = hist[-1, 1]
        r_steady = hist[-1, 2]
        V_steady = np.sqrt(u_steady**2 + v_steady**2)
        R = V_steady / abs(r_steady) if r_steady != 0 else float('inf')
        turning_radii.append(R)
        
        plt.plot(y_traj, x_traj, label=f"D = {D*1000:.0f} mm (R = {R:.2f} m)")
        
    plt.xlabel("y [m] (Starboard)")
    plt.ylabel("x [m] (Advance)")
    plt.title("Turning Circle Simulation (Fossen 3-DOF Model)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.savefig(os.path.join(output_folder, "turning_circles_fossen.png"), dpi=300, bbox_inches="tight")
    
    # Zig-zag test for the baseline diameter (0.166)
    auv_base = AUVSimulator(D=0.166)
    t_zz, hist_zz, delta_zz = run_zigzag(auv_base, deg=20.0, t_end=60.0)
    
    plt.figure(figsize=(10, 5))
    plt.plot(t_zz, np.degrees(hist_zz[:, 5]), label='Heading (deg)', linewidth=2)
    plt.plot(t_zz, np.degrees(delta_zz), '--', label='Rudder (deg)', linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title("20/20 Zig-Zag Test (D=166mm)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_folder, "zigzag_20_20.png"), dpi=300, bbox_inches="tight")
    
    # Sensitivity Analysis Plot
    turning_radii = np.array(turning_radii)
    diameters_np = np.array(diameters)
    sensitivity = np.gradient(turning_radii, diameters_np)
    
    plt.figure(figsize=(8, 5))
    plt.plot(diameters_np*1000, turning_radii, marker='o', linewidth=2)
    plt.xlabel("Diameter (mm)")
    plt.ylabel("Turning Radius (m)")
    plt.title("Realistic Maneuverability: Turning Radius vs Diameter")
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, "turning_radius_vs_diameter_fossen.png"), dpi=300, bbox_inches="tight")

    print("\n============================================================")
    print("Fossen 3-DOF Simulator: Turning Circle & Sensitivity Analysis")
    print("============================================================")
    print(f"{'Diameter (mm)':<15} | {'Turning Radius (m)':<20} | {'Sensitivity (dR/dD)':<20}")
    print("-" * 60)
    for d, r, s in zip(diameters, turning_radii, sensitivity):
        print(f"{d*1000:<15.0f} | {r:<20.2f} | {s:<20.2f}")
    print("============================================================")