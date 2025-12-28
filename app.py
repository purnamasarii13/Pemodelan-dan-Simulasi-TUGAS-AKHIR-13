import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp, odeint
from scipy.optimize import minimize

st.set_page_config(page_title="Lotka–Volterra (Predator–Prey)", layout="wide")

# -------------------------
# Model & Solvers
# -------------------------
def lv_rhs_odeint(z, t, alpha, beta, delta, gamma):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

def lv_rhs_solveivp(t, z, alpha, beta, delta, gamma):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

def simulate_odeint(params, t_eval, x0, y0):
    alpha, beta, delta, gamma = params
    sol = odeint(lv_rhs_odeint, [x0, y0], t_eval, args=(alpha, beta, delta, gamma))
    return sol[:, 0], sol[:, 1]

def simulate_solveivp(params, t_eval, x0, y0):
    alpha, beta, delta, gamma = params
    sol = solve_ivp(
        fun=lambda t, z: lv_rhs_solveivp(t, z, alpha, beta, delta, gamma),
        t_span=(float(np.min(t_eval)), float(np.max(t_eval))),
        y0=[x0, y0],
        t_eval=t_eval,
        method="RK45",
        rtol=1e-7,
        atol=1e-10,
    )
    if not sol.success:
        return None
    return sol.y[0], sol.y[1]

def simulate_rk4(params, t_eval, x0, y0):
    alpha, beta, delta, gamma = params
    t_eval = np.asarray(t_eval)
    x = np.zeros_like(t_eval, dtype=float)
    y = np.zeros_like(t_eval, dtype=float)
    x[0], y[0] = float(x0), float(y0)

    def f(_t, _x, _y):
        dx = alpha * _x - beta * _x * _y
        dy = delta * _x * _y - gamma * _y
        return dx, dy

    for i in range(1, len(t_eval)):
        h = float(t_eval[i] - t_eval[i - 1])
        ti = float(t_eval[i - 1])
        xi, yi = float(x[i - 1]), float(y[i - 1])

        k1x, k1y = f(ti, xi, yi)
        k2x, k2y = f(ti + h / 2, xi + h * k1x / 2, yi + h * k1y / 2)
        k3x, k3y = f(ti + h / 2, xi + h * k2x / 2, yi + h * k2y / 2)
        k4x, k4y = f(ti + h, xi + h * k3x, yi + h * k3y)

        x[i] = xi + (h / 6) * (k1x + 2 * k2x + 2 * k3x + k4x)
        y[i] = yi + (h / 6) * (k1y + 2 * k2y + 2 * k3y + k4y)

    return x, y

def simulate(params, t_eval, x0, y0, solver: str):
    if solver == "rk4":
        return simulate_rk4(params, t_eval, x0, y0)
    if solver == "odeint":
        return simulate_odeint(params, t_eval, x0, y0)
    if solver == "solve_ivp":
        out = simulate_solveivp(params, t_eval, x0, y0)
        return out
    raise ValueError("solver tidak dikenal")

def mse_loss(params, t_eval, x0, y0, x_data, y_data, solver):
    params = np.array(params, dtype=float)
    if np.any(params <= 0):
        return 1e18
    out = simulate(params, t_eval, x0, y0, solver)
    if out is None:
        return 1e18
    x_sim, y_sim = out
    err_x = np.mean((x_sim - x_data) ** 2)
    err_y = np.mean((y_sim - y_data) ** 2)
    return float(err_x + err_y)

# -------------------------
# UI
# -------------------------
st.title("Analisis Dinamika Populasi Predator–Prey Menggunakan Model Lotka–Volterra dengan Pendekatan Simulasi")

with st.sidebar:
    st.header("1) Input Data")
    uploaded = st.file_uploader("Upload CSV (time-series)", type=["csv"])

    st.caption("Jika tidak upload, app akan mencoba membaca file lokal `hasil_simulasi_utama.csv` (jika ada).")

    st.header("2) Solver")
    solver = st.selectbox(
        "Pilih solver",
        options=["rk4", "solve_ivp", "odeint"],
        index=0,
        help="RK4 buatan sendiri (sesuai tips modul), atau solver SciPy."
    )

    st.header("3) Scaling (Opsional)")
    use_scaling = st.checkbox("Aktifkan normalisasi (bagi skala)", value=False)
    scale = st.number_input("Skala (contoh 1000)", min_value=1.0, value=1000.0, step=100.0)

# Load dataframe
df = None
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    # fallback local file name (untuk deploy: taruh file ini di folder yang sama)
    try:
        df = pd.read_csv("hasil_simulasi_utama.csv")
    except Exception:
        df = None

if df is None:
    st.warning("Upload CSV dulu agar aplikasi bisa berjalan.")
    st.stop()

st.subheader("Preview Data")
st.dataframe(df.head(20), use_container_width=True)

cols = list(df.columns)
st.sidebar.header("4) Pilih Kolom")
t_col = st.sidebar.selectbox("Kolom waktu (t)", cols, index=cols.index("Waktu_hari") if "Waktu_hari" in cols else 0)
x_col = st.sidebar.selectbox("Kolom prey (x)", cols, index=cols.index("Padi_tanpa_kg_ha") if "Padi_tanpa_kg_ha" in cols else 0)
y_col = st.sidebar.selectbox("Kolom predator (y)", cols, index=cols.index("Wereng_tanpa_ind_ha") if "Wereng_tanpa_ind_ha" in cols else 0)

# Extract arrays
t_data = df[t_col].to_numpy(dtype=float)
x_data = df[x_col].to_numpy(dtype=float)
y_data = df[y_col].to_numpy(dtype=float)

# Ensure t is sorted
order = np.argsort(t_data)
t_data = t_data[order]
x_data = x_data[order]
y_data = y_data[order]

# Optional scaling
if use_scaling:
    x_data_s = x_data / float(scale)
    y_data_s = y_data / float(scale)
else:
    x_data_s = x_data.copy()
    y_data_s = y_data.copy()

x0, y0 = float(x_data_s[0]), float(y_data_s[0])

# Parameter controls
with st.sidebar:
    st.header("5) Parameter Lotka–Volterra")
    # Range slider yang aman (Anda bisa ubah)
    alpha = st.slider("alpha (α) — pertumbuhan prey", min_value=0.000001, max_value=5.0, value=0.5, format="%.6f")
    beta  = st.slider("beta (β) — predasi",           min_value=0.000001, max_value=5.0, value=0.001, format="%.6f")
    delta = st.slider("delta (δ) — konversi prey→pred", min_value=0.000001, max_value=5.0, value=0.0005, format="%.6f")
    gamma = st.slider("gamma (γ) — kematian predator", min_value=0.000001, max_value=5.0, value=0.3, format="%.6f")

    st.header("6) Fitting Otomatis (Opsional)")
    do_fit = st.checkbox("Jalankan optimisasi untuk cari parameter terbaik", value=False)
    maxiter = st.number_input("Max iter (Nelder-Mead)", min_value=200, max_value=20000, value=6000, step=200)

params_manual = np.array([alpha, beta, delta, gamma], dtype=float)

# Plot data mentah (time series)
col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Mentah (Time Series)")
    fig = plt.figure(figsize=(7, 4))
    plt.plot(t_data, x_data_s, linestyle="--", label="Prey (Data)")
    plt.plot(t_data, y_data_s, linestyle="--", label="Predator (Data)")
    plt.xlabel("Waktu")
    plt.ylabel("Populasi")
    plt.title("Data Asli (garis putus-putus)")
    plt.grid(True)
    plt.legend()
    st.pyplot(fig, clear_figure=True)

# Fit otomatis (opsional)
best_params = params_manual.copy()
fit_result = None
if do_fit:
    with st.spinner("Menjalankan optimisasi (Nelder–Mead)..."):
        fit_result = minimize(
            fun=lambda p: mse_loss(p, t_data, x0, y0, x_data_s, y_data_s, solver),
            x0=params_manual,
            method="Nelder-Mead",
            options={"maxiter": int(maxiter), "xatol": 1e-10, "fatol": 1e-10}
        )
        if fit_result.success:
            best_params = fit_result.x
        else:
            # tetap tampilkan solusi terbaik yang ditemukan
            best_params = fit_result.x

# Simulate with chosen/best params
out = simulate(best_params, t_data, x0, y0, solver)
if out is None:
    st.error("Simulasi gagal dengan parameter saat ini. Coba ubah parameter atau ganti solver.")
    st.stop()

x_sim, y_sim = out

with col2:
    st.subheader("Ringkasan Parameter")
    alpha_b, beta_b, delta_b, gamma_b = [float(v) for v in best_params]
    info = {
        "solver": solver,
        "use_scaling": use_scaling,
        "scale": float(scale) if use_scaling else 1.0,
        "x0": x0,
        "y0": y0,
        "alpha": alpha_b,
        "beta": beta_b,
        "delta": delta_b,
        "gamma": gamma_b,
    }
    if fit_result is not None:
        info["fit_success"] = bool(fit_result.success)
        info["loss(MSE total)"] = float(fit_result.fun)
    st.json(info)

st.divider()

# Overlay plots: data dashed, sim solid (sesuai instruksi TA)
c1, c2 = st.columns(2)

with c1:
    st.subheader("Overlay Prey — Data vs Simulasi")
    fig = plt.figure(figsize=(7, 4))
    plt.plot(t_data, x_data_s, linestyle="--", label="Prey (Data)")
    plt.plot(t_data, x_sim, linestyle="-", label="Prey (Sim)")
    plt.xlabel("Waktu")
    plt.ylabel("Populasi")
    plt.title("Garis putus-putus = data | Garis solid = simulasi")
    plt.grid(True)
    plt.legend()
    st.pyplot(fig, clear_figure=True)

with c2:
    st.subheader("Overlay Predator — Data vs Simulasi")
    fig = plt.figure(figsize=(7, 4))
    plt.plot(t_data, y_data_s, linestyle="--", label="Predator (Data)")
    plt.plot(t_data, y_sim, linestyle="-", label="Predator (Sim)")
    plt.xlabel("Waktu")
    plt.ylabel("Populasi")
    plt.title("Garis putus-putus = data | Garis solid = simulasi")
    plt.grid(True)
    plt.legend()
    st.pyplot(fig, clear_figure=True)

st.divider()

# Phase portrait
st.subheader("Phase Portrait (x vs y)")
fig = plt.figure(figsize=(6, 6))
plt.plot(x_data_s, y_data_s, linestyle="--", label="Data (x vs y)")
plt.plot(x_sim, y_sim, linestyle="-", label="Sim (x vs y)")
plt.xlabel("Prey (x)")
plt.ylabel("Predator (y)")
plt.title("Phase Portrait: Data vs Simulasi")
plt.grid(True)
plt.legend()
st.pyplot(fig, clear_figure=True)

st.caption("Catatan: Jika orbit membentuk lintasan tertutup → indikasi siklus; jika spiral → menuju keseimbangan atau divergen, tergantung arah spiral.")
