# %%
import numpy as np
import matplotlib.pyplot as plt

# ── Load ────────────────────────────────────────────────────
data = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/eigenfunctions/eigenfunction_clean_m1_-88.0_highlat_anti_hmi_m_720s_dt_1h.npz')

ef_uphi       = data['ef_uphi']
ef_uthe       = data['ef_uthe']
uphi_err_real = data['uphi_err_real']
uphi_err_imag = data['uphi_err_imag']
uthe_err_real = data['uthe_err_real']
uthe_err_imag = data['uthe_err_imag']
lats          = data['lats']

# ── Plot ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
fig.suptitle(r'Eigenfunction $m=1$, $\nu=-88$ nHz', fontsize=13)

components = [
    (ef_uphi.real,  uphi_err_real, r'Re$(u_\phi)$',     'C0', axes[0, 0]),
    (ef_uphi.imag,  uphi_err_imag, r'Im$(u_\phi)$',     'C0', axes[0, 1]),
    (ef_uthe.real,  uthe_err_real, r'Re$(u_\theta)$',   'C1', axes[1, 0]),
    (ef_uthe.imag,  uthe_err_imag, r'Im$(u_\theta)$',   'C1', axes[1, 1]),
]

for vals, errs, label, color, ax in components:
    ax.plot(lats, vals, color=color, lw=1.5, label=label)
    ax.fill_between(lats, vals - errs, vals + errs,
                    alpha=0.3, color=color, label=r'1$\sigma$')
    ax.axhline(0, color='k', lw=0.7, ls='--')
    ax.axvline(0, color='k', lw=0.7, ls=':')
    ax.set_ylabel(label, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

axes[1, 0].set_xlabel('Latitude (°)', fontsize=11)
axes[1, 1].set_xlabel('Latitude (°)', fontsize=11)

plt.tight_layout()
# plt.savefig('eigenfunction_m2.pdf', bbox_inches='tight', dpi=150)
plt.show()
