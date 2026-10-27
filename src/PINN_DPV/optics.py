"""Optical source term of dPV: standard spectrum, the LDS spectral transform, generation.

Eq. (9): in E4 an LDS layer absorbs A_k = 0.95 of the photon flux of every dPV
band between 300 and 400 nm and re-emits eta_geom eta_PL of the absorbed photons
in the single band centred at 606.5 nm; P_eff,k = Phi_eff,k hc / lambda_k. E1-E3
use the standard spectrum unchanged.
Eq. (10): direct-gap absorption alpha = A sqrt(max(0, hc/lambda - Eg)) with
Beer-Lambert attenuation accumulated from the front contact, summed over the
M = 99 bands without quadrature weights, divided by G_0.

`generation_on_grid` reproduces dPV's `optical.compute_G` operation by operation
(left-Riemann accumulation on its grid); `generation_at` evaluates Eq. (10) at any
x_hat with the exact integral of the piecewise-constant alpha. The two agree on
every node of E1 (one material); with heterointerfaces they differ by dPV's own
left-Riemann discretization (measured below 3e-4 of max G away from the interface
node, whose layer dPV assigns by rounding).

The standard spectrum below is dPV's `sun.Lambda_eff` and `sun.P_in_eff`
(deltapv f874e24, MIT License, Copyright (c) the deltapv authors), copied verbatim.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from PINN_DPV import data, scaling

__provenance__ = {
    "revision": "research-concept-modelado-computacional-directo-e-inverso-de-celdas-solares-d-r08.md",
    "sections": ["4"],
    "equations": ["9", "10"],
    "invariants": [
        "generation_matches_dpv",
        "lds_photon_balance",
        "lds_identity_outside_window",
        "continuous_generation_agrees_with_dpv_grid",
        "standard_spectrum_power",
    ],
}

# dPV's standard spectrum (sun.py), verbatim.
LAMBDA_EFF = np.array([
    296.58786520846525, 301.86932586906727, 307.3413862642167,
    314.0101663463744, 322.2157186534005, 331.6588606552103,
    343.16563211861506, 356.4718787589982, 371.43844861016, 388.4562139889681,
    406.85926227327474, 426.0481717998874, 447.62901892787136,
    469.92116394102163, 494.07400235193575, 519.8819737412621, 547.21371249863,
    576.0744631926423, 606.46754270433, 638.2129109456624, 671.2959617860226,
    705.7863072495503, 742.0507337674146, 780.5554341808609, 818.2455840936051,
    858.0879046742818, 897.7411488242675, 940.8662643813793, 987.4902295689035,
    1030.0610336012219, 1074.091374437584, 1117.2269901074096,
    1173.443596282667, 1218.9715095166084, 1264.2332426026787,
    1307.3216392819909, 1342.224915652247, 1447.3678952624173,
    1497.1669408915898, 1542.2086303659403, 1591.7043105789685,
    1642.3988363909662, 1692.8888886873513, 1741.451683376098,
    1783.2390162887173, 1848.737471241858, 1963.9220689859833,
    2001.8975232780376, 2052.9961725177836, 2105.726153145771,
    2157.036887098164, 2209.6681561997557, 2260.783110191019,
    2311.1416799890626, 2359.5013086654512, 2405.924012276593,
    2447.2596671376177, 2479.1748474211004, 2508.9873785326154,
    2803.94419627386, 2862.6037651092292, 2898.589860171952, 2935.189959763852,
    2977.422243585477, 3016.3189651212374, 3065.497960229955,
    3118.9606513702056, 3161.8065262433493, 3196.3617805534977,
    3254.417788924162, 3295.710821301024, 3346.8175838814655,
    3390.141776338096, 3430.816643341351, 3470.4895400476826,
    3508.7437587983923, 3546.544332677495, 3583.80383634895, 3619.463180983516,
    3652.9771254767124, 3687.57586456887, 3717.861131147382,
    3747.7584839248098, 3776.2656006382517, 3803.1349830975896,
    3828.321455229104, 3851.8040704128543, 3873.6506783527016,
    3894.7021530916877, 3912.8548933604898, 3929.98782245093, 3944.9901304159,
    3958.1676145712195, 3969.5282751392497, 3979.1727953185414,
    3986.99564951234, 3992.9791614959527, 3997.1413552005, 3999.457603085752
])

P_IN_EFF = np.array([
    0.00012889204787158213, 0.008777279582043227, 0.10107774097655463,
    0.46722334523923387, 1.2150988129071114, 2.4725889291277845,
    3.5265164985065622, 4.936507253041806, 7.269751112678899,
    8.927357155341136, 16.020906918535356, 19.34033100690697,
    26.057509884831223, 30.829282730150954, 33.39515661592565,
    35.44747476918117, 38.125496404177476, 39.540923343124724,
    40.8404938686826, 41.596166567916164, 41.940750589024525,
    39.789515395817496, 38.39403584669429, 37.691423788156115,
    35.18840644601061, 35.75215291789887, 30.259804321835226,
    18.202504474049892, 27.752491142563077, 28.337421416452848,
    25.1872011944878, 12.673966009585598, 17.23007013474619,
    19.802853014125215, 18.19642379589036, 13.500050082734614,
    3.001238136332884, 2.6573289581422364, 8.217048544364276,
    12.530664298026352, 11.933102731756618, 11.141729752106082,
    9.840969492071611, 7.532064618549481, 3.399772792777609,
    0.045292634528007025, 1.2942960011011828, 2.6679591131597404,
    3.7979450072049112, 4.513945457451602, 4.317987777546656, 3.90667388108776,
    3.43007058789695, 2.8661884962618718, 2.1310126280587927,
    1.529495517033794, 0.9838948727965099, 0.3344187742565935,
    0.042086112090700394, 4.5965977504444245e-6, 0.002519090572158028,
    0.03909265151751956, 0.13365361767129383, 0.17652719137748085,
    0.21141755386263275, 0.14834619599764395, 0.19075355865114366,
    0.35892368435944627, 0.13533761251828436, 0.16353225285698836,
    0.1822229755593091, 0.22581063882710814, 0.3320492714611748,
    0.3977721572943606, 0.4374505055593256, 0.4476869896126664,
    0.377551925311501, 0.35114326073602675, 0.35871733567086955,
    0.3452873527071678, 0.2949742205561386, 0.30177237871160434,
    0.2706572565634048, 0.2483643691929065, 0.23175334821947552,
    0.21865531905695917, 0.19395293482122686, 0.1529684011833061,
    0.14372812778454383, 0.1280118754702922, 0.1138792546671928,
    0.10644202367603008, 0.09530732687067581, 0.08099285747487651,
    0.06504601291568587, 0.05127412754488824, 0.03704777100973708,
    0.023219685557830824, 0.009917010541730737
])

# LDS layer (r08 Sec. 4 and 7; kalytchuk2014)
LDS_A_UV = 0.95
LDS_WINDOW_NM = (300.0, 400.0)
LDS_ETA_GEOM = 0.87
LDS_LAMBDA_EM_NM = 606.5


def standard_spectrum():
    """(lambda_k in nm, P_in,k in W/m^2) of dPV's incident_light("sun")."""
    return LAMBDA_EFF.copy(), P_IN_EFF.copy()


def lds_spectrum(eta_pl: float):
    """Eq. (9), as dpvgen builds the LightSource handed to dPV in E4."""
    lam, P = standard_spectrum()
    A = np.where((lam >= LDS_WINDOW_NM[0]) & (lam <= LDS_WINDOW_NM[1]), LDS_A_UV, 0.0)
    k_em = int(np.argmin(np.abs(lam - LDS_LAMBDA_EM_NM)))
    photons = P * lam  # proportional to Phi_k = P_k lambda_k / (hc)
    absorbed = float(np.sum(A * photons))
    eff = photons * (1.0 - A)
    eff[k_em] += LDS_ETA_GEOM * float(eta_pl) * absorbed
    return lam, eff / lam


def spectrum(family: str, xi: dict):
    """The spectrum that enters the cell: Eq. (9) in E4, the standard one otherwise."""
    return lds_spectrum(xi["eta_PL"]) if family == "E4" else standard_spectrum()


def standard_power() -> float:
    """Denominator of Eq. (14): the standard spectrum's power, also in E4."""
    return float(np.sum(P_IN_EFF))


def _photonflux(lam, P):
    return P / (scaling.HC / (lam * scaling.NM))  # 1 / (m^2 s)


def _alpha(A, Eg_dimless, lam):
    """alpha(lambda) in 1/m for arrays of A (cm^-1 eV^-1/2) and dimensionless Eg."""
    A_si = A / scaling.CM / jnp.sqrt(scaling.EV)
    Eg_si = Eg_dimless * scaling.ENERGY * scaling.EV
    lam_si = lam * scaling.NM
    gap = scaling.HC / lam_si - Eg_si
    return jnp.where(gap > 0, A_si * jnp.sqrt(jnp.abs(gap)), 0.0)


def generation_on_grid(x_debye, nodes, lam, P):
    """dPV's compute_G on its own grid: G in units of G_0, one value per node."""
    x = jnp.asarray(x_debye) * scaling.LENGTH * scaling.CM  # m
    dx = jnp.diff(x)
    phi0 = _photonflux(jnp.asarray(lam), jnp.asarray(P))
    alphas = _alpha(jnp.asarray(nodes["A"])[None, :], jnp.asarray(nodes["Eg"])[None, :],
                    jnp.asarray(lam)[:, None])  # (M, N)
    depth = jnp.cumsum(jnp.concatenate([jnp.zeros((alphas.shape[0], 1)),
                                        alphas[:, :-1] * dx[None, :]], axis=1), axis=1)
    g = phi0[:, None] * jnp.exp(-depth) * alphas
    return jnp.sum(g, axis=0) / 1e6 / scaling.GRATEDENS


def generation_at(x_hat, family: str, xi: dict, lam=None, P=None):
    """Eq. (10) at any x_hat in [0, 1], with the exact optical depth of piecewise alpha.

    A point on a heterointerface takes the material of the layer behind it, as
    dPV's make_design assigns the shared node.
    """
    if lam is None:
        lam, P = spectrum(family, xi)
    ls, mats, _ = data.layers(family, xi)
    l_total = data.l_total_cm(family, xi)
    starts = np.concatenate([[0.0], np.cumsum(ls)[:-1]])
    A = np.array([m["A"] for m in mats])
    Eg = np.array([m["Eg"] / scaling.ENERGY for m in mats])
    alpha_l = _alpha(jnp.asarray(A)[None, :], jnp.asarray(Eg)[None, :],
                     jnp.asarray(lam)[:, None])  # (M, layers), 1/m

    x_cm = jnp.asarray(x_hat) * l_total
    overlap = jnp.clip(x_cm[..., None] - jnp.asarray(starts), 0.0, jnp.asarray(ls))  # (..., layers)
    depth = jnp.einsum("...l,ml->...m", overlap * scaling.CM, alpha_l)  # (..., M)
    # A point within 1e-12 relative of a heterointerface is on it and takes the
    # layer behind; dPV's own choice at that node depends on rounding.
    layer = jnp.sum(x_cm[..., None] >= jnp.asarray(starts) * (1 - 1e-12), axis=-1) - 1
    alpha_here = alpha_l.T[layer]  # (..., M)
    phi0 = _photonflux(jnp.asarray(lam), jnp.asarray(P))
    g = jnp.sum(phi0 * alpha_here * jnp.exp(-depth), axis=-1)
    return g / 1e6 / scaling.GRATEDENS


def _lds_affine():
    """Eq. (9) is affine in eta_PL: P_eff = P0 + eta_PL P1."""
    _, p0 = lds_spectrum(0.0)
    _, p1 = lds_spectrum(1.0)
    return p0, p1 - p0


def generation_jax(fs, x_hat, lay):
    """Eq. (10) for a `data.FamilyStage` at x_hat, with `lay = fs.layout(xi_hat)`.

    Vectorizes over leading axes shared by x_hat and the layout; a point on a
    heterointerface takes the layer behind it, as `generation_at` does.
    """
    lam = jnp.asarray(LAMBDA_EFF)
    if fs.family == "E4":
        p0, p1 = _lds_affine()
        P = jnp.asarray(p0) + jnp.asarray(lay["eta"])[..., None] * jnp.asarray(p1)
    else:
        P = jnp.asarray(P_IN_EFF)
    alpha_l = _alpha(fs.mat["A"][None, :], fs.mat["Eg"][None, :], lam[:, None])  # (M, L)
    thick = lay["thick"]
    starts = jnp.concatenate([jnp.zeros_like(thick[..., :1]), jnp.cumsum(thick, axis=-1)[..., :-1]], axis=-1)
    x_cm = x_hat * lay["l_total"]
    overlap = jnp.clip(x_cm[..., None] - starts, 0.0, thick)
    depth = jnp.einsum("...l,ml->...m", overlap * scaling.CM, alpha_l)
    layer = jnp.sum(x_cm[..., None] >= starts * (1 - 1e-12), axis=-1) - 1
    alpha_here = alpha_l.T[layer]
    phi0 = _photonflux(lam, P)
    return jnp.sum(phi0 * alpha_here * jnp.exp(-depth), axis=-1) / 1e6 / scaling.GRATEDENS
