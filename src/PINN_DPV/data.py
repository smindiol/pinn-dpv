"""The dPV reference dataset: families, device construction, loading and the per-seed split.

Sec. 7 fixes four families (E1 homojunction, E2 CdS/CdTe, E3 perovskite
ETM/absorber/HTM, E4 CdS/CdTe with an LDS layer) and two stages each. This
module carries their nominal materials exactly as the dataset generator
`dpvgen.simulacion` handed them to dPV, rebuilds each device's piecewise
material profile on dPV's own grid the way `make_design` assigns it (every
layer claims start <= x <= start + t, a later layer overwriting the shared
node), reads the HDF5 files, and partitions devices 80/10/10 with the seed of
each run (Sec. 7; never by point or by voltage). Corners of the xi hypercube
always go to test (Sec. 7).

Deviations from r08 that the dataset card fixes and this module follows: the
E3 nominal absorber is 500 nm, the HTM acceptor ceiling is 1.87e18, and E2/E4
use ex2's graded mesh with 4x its nodes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import h5py
import numpy as np

from PINN_DPV import scaling

__provenance__ = {
    "revision": "research-concept-modelado-computacional-directo-e-inverso-de-celdas-solares-d-r08.md",
    "sections": ["7", "8"],
    "equations": [],
    "invariants": [
        "device_materials_match_dataset",
        "split_is_partition_by_device",
        "split_depends_on_seed",
        "corners_go_to_test",
    ],
}

# --- Materials (dPV units), as dpvgen.simulacion builds them ----------------
_MAT_DEFAULTS = {"Et": 0.0, "Br": 0.0, "Cn": 0.0, "Cp": 0.0, "A": 0.0}

E1_MATERIAL = dict(_MAT_DEFAULTS, Chi=3.9, Eg=1.5, eps=9.4, Nc=8e17, Nv=1.8e19,
                   mn=100.0, mp=100.0, tn=1e-8, tp=1e-8, A=2e4)
CDS = dict(_MAT_DEFAULTS, Nc=2.2e18, Nv=1.8e19, Eg=2.4, eps=10.0, mn=100.0, mp=25.0,
           tn=1e-8, tp=1e-13, Chi=4.0, A=1e4)
CDTE = dict(_MAT_DEFAULTS, Nc=8e17, Nv=1.8e19, Eg=1.5, eps=9.4, mn=320.0, mp=40.0,
            tn=5e-9, tp=5e-9, Chi=3.9, A=1e4)

# Initial design x0 of dPV's optimize/psc.py (ETM and HTM are not real materials).
_X0_PSC = [1.661788237392516, 4.698293002285373, 19.6342803183675, 18.83471869026531,
           19.54569869328745, 0.7252792557586427, 1.6231392299175988,
           2.5268524699070234, 2.51936429069554, 6.933634938056497, 19.41835918276137,
           18.271793488422656, 0.46319949214386513, 0.2058139980642224,
           18.63975340175838, 17.643726318153238]
_TAU_PSC, _A_PSC = 1e-6, 2e4
PEROVSKITE = dict(_MAT_DEFAULTS, Eg=1.5, Chi=3.9, eps=10.0, Nc=3.9e18, Nv=2.7e18,
                  mn=2.0, mp=2.0, tn=_TAU_PSC, tp=_TAU_PSC, Br=2.3e-9, A=_A_PSC)
ETM = dict(_MAT_DEFAULTS, Eg=_X0_PSC[0], Chi=_X0_PSC[1], eps=_X0_PSC[2], Nc=10 ** _X0_PSC[3],
           Nv=10 ** _X0_PSC[4], mn=10 ** _X0_PSC[5], mp=10 ** _X0_PSC[6],
           tn=_TAU_PSC, tp=_TAU_PSC, A=_A_PSC)
HTM = dict(_MAT_DEFAULTS, Eg=_X0_PSC[7], Chi=_X0_PSC[8], eps=_X0_PSC[9], Nc=10 ** _X0_PSC[10],
           Nv=10 ** _X0_PSC[11], mn=10 ** _X0_PSC[12], mp=10 ** _X0_PSC[13],
           tn=_TAU_PSC, tp=_TAU_PSC, A=_A_PSC)

#: Contact recombination velocities (cm/s): Snl, Snr, Spl, Spr.
CONTACTS = {
    "E1": (1e7, 0.0, 0.0, 1e7),
    "E2": (1.16e7, 1.16e7, 1.16e7, 1.16e7),
    "E3": (1e7, 1e7, 1e7, 1e7),
    "E4": (1.16e7, 1.16e7, 1.16e7, 1.16e7),
}

#: The 20 canonical files of the dataset card, relative to dpv-dataset/salidas.
#: E2/E4 come from the 4x mesh (m4-*); the older 180-node files are superseded.
CANONICAL_FILES = {
    "E1A": ["s1/dpv_E1A_b000.h5", "s1/dpv_E1A_b001.h5"],
    "E1B": ["s1/dpv_E1B_b000.h5", "s4/dpv_E1B_b001.h5"],
    "E2A": ["m4-s5/dpv_E2A_m4_b000.h5", "m4-s5/dpv_E2A_m4_b001.h5", "m4-s9/dpv_E2A_m4_b002.h5"],
    "E2B": ["m4-s6/dpv_E2B_m4_b000.h5", "m4-s6/dpv_E2B_m4_b001.h5", "m4-s9/dpv_E2B_m4_b002.h5"],
    "E3A": ["s2/dpv_E3A_b000.h5", "s2/dpv_E3A_b001.h5"],
    "E3B": ["s2/dpv_E3B_b000.h5", "s2/dpv_E3B_b001.h5"],
    "E4A": ["m4-s7/dpv_E4A_m4_b000.h5", "m4-s7/dpv_E4A_m4_b001.h5", "m4-s10/dpv_E4A_m4_b002.h5"],
    "E4B": ["m4-s8/dpv_E4B_m4_b000.h5", "m4-s8/dpv_E4B_m4_b001.h5", "m4-s10/dpv_E4B_m4_b002.h5"],
}

MATERIAL_KEYS = ("eps", "Chi", "Eg", "Nc", "Nv", "mn", "mp", "tn", "tp", "Et", "Br", "A")


def layers(family: str, xi: dict):
    """(thicknesses_cm, materials, dopings_cm3) in front-to-back order, as dpvgen builds them."""
    if family == "E1":
        return [xi["L_n"], xi["L_p"]], [E1_MATERIAL, E1_MATERIAL], [xi["Nd_n"], -xi["Na_p"]]
    if family in ("E2", "E4"):
        # dpvgen lengthens the last layer by 1e-9 relative so x = L keeps its material.
        return ([xi["L_CdS"], xi["L_CdTe"] * (1 + 1e-9)], [CDS, CDTE],
                [xi["Nd_CdS"], -xi["Na_CdTe"]])
    if family == "E3":
        return ([xi["L_ETL"], xi["L_abs"], xi["L_HTL"]], [ETM, PEROVSKITE, HTM],
                [xi["Nd_ETM"], 0.0, -xi["Na_HTM"]])
    raise ValueError(f"unknown family: {family}")


def interfaces_cm(family: str, xi: dict):
    """Heterointerface positions in cm (none in E1, one in E2/E4, two in E3)."""
    if family == "E1":
        return []
    ls, _, _ = layers(family, xi)
    return list(np.cumsum(ls[:-1]))


def interfaces_hat(family: str, xi: dict):
    """x_hat_m = sum_{l<=m} L_l / L_total (Sec. 2), with the physical L_total."""
    ls, _, _ = layers(family, xi)
    if family in ("E2", "E4"):
        ls = [xi["L_CdS"], xi["L_CdTe"]]
    total = float(sum(ls))
    return [float(v) / total for v in interfaces_cm(family, xi)]


def l_total_cm(family: str, xi: dict) -> float:
    if family == "E1":
        return xi["L_n"] + xi["L_p"]
    if family in ("E2", "E4"):
        return xi["L_CdS"] + xi["L_CdTe"]
    return xi["L_ETL"] + xi["L_abs"] + xi["L_HTL"]


def node_materials(family: str, xi: dict, x_debye):
    """Dimensionless material arrays on dPV's grid, assigned as `make_design` assigns them.

    Returns a dict of arrays (length N) for MATERIAL_KEYS plus Ndop, each divided
    by its dPV unit, except A, which dPV keeps in cm^-1 eV^-1/2.
    """
    x_cm = np.asarray(x_debye, dtype=np.float64) * scaling.LENGTH
    ls, mats, dops = layers(family, xi)
    out = {k: np.zeros_like(x_cm) for k in MATERIAL_KEYS}
    out["Ndop"] = np.zeros_like(x_cm)
    start = 0
    for t, mat, dop in zip(ls, mats, dops):
        idx = np.logical_and(start <= x_cm, x_cm <= start + t)
        for k in MATERIAL_KEYS:
            unit = 1.0 if k == "A" else scaling.UNITS[k]
            out[k] = np.where(idx, mat[k] / unit, out[k])
        out["Ndop"] = np.where(idx, dop / scaling.DENSITY, out["Ndop"])
        start = start + t
    return out


def contact_velocities(family: str):
    """Dimensionless Snl, Snr, Spl, Spr (scale v_0)."""
    return tuple(s / scaling.VELOCITY for s in CONTACTS[family])


# --- Loading ----------------------------------------------------------------
@dataclass
class Device:
    family: str
    stage: str
    key: str  # "<block>/<group>/<name>", unique within a family-stage
    xi: dict
    u: np.ndarray
    attrs: dict
    x: np.ndarray | None = None  # dPV grid in Debye units (stored as `x_cm`)
    G: np.ndarray | None = None
    phi_eq: np.ndarray | None = None
    V: np.ndarray | None = None
    J: np.ndarray | None = None  # A/cm^2
    pots: np.ndarray | None = None  # (K, 3, N)
    Jn: np.ndarray | None = None  # (K, N-1), Scharfetter-Gummel
    Jp: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    @property
    def pce(self) -> float:
        return float(self.attrs["eff"])


def dataset_root() -> str:
    """Root holding the canonical files: $PINN_DPV_DATA, else a dataset attached to a Kaggle
    kernel under /kaggle/input, else the sibling dpv-dataset checkout."""
    env = os.environ.get("PINN_DPV_DATA")
    if env:
        return env
    import glob

    hits = sorted(glob.glob("/kaggle/input/**/dpv_E1A_b000.h5", recursive=True))
    if hits:
        here = os.path.dirname(hits[0])
        return os.path.dirname(here) if os.path.basename(here) == "s1" else here
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "..", "..", "dpv-dataset", "salidas"))


KAGGLE_DATASET = "sgarizado/dpv-pinn-canonical"


def ensure_dataset(family_stages=None, slug: str = KAGGLE_DATASET) -> str:
    """The data root, downloading the missing canonical files of `family_stages` (all when
    None) from the public Kaggle dataset when running inside a Kaggle kernel.

    Kaggle refuses to attach a dataset to a non-interactive kernel, so kagglehub's in-kernel
    attach path is switched off and each file is fetched over HTTP, one by one: a step pays
    only for the family-stages it reads."""
    wanted = list(CANONICAL_FILES) if family_stages is None else list(family_stages)
    root = dataset_root()
    missing = [rel for fs in wanted for rel, p in zip(CANONICAL_FILES[fs], canonical_paths(fs, root))
               if not os.path.exists(p)]
    if not missing or not os.path.isdir("/kaggle"):
        return root
    os.environ["DISABLE_KAGGLE_CACHE"] = "1"
    import kagglehub

    for rel in missing:
        path = kagglehub.dataset_download(slug, path=os.path.basename(rel))
        os.environ["PINN_DPV_DATA"] = os.path.dirname(path)
        print(f"[data] {os.path.basename(rel)} -> {os.environ['PINN_DPV_DATA']}", flush=True)
    return dataset_root()


def canonical_paths(family_stage: str, root: str | None = None):
    """The family-stage's files under root, laid out as dPV writes them (<run>/<file>) or
    flat (<file>), which is how the Kaggle dataset holds them."""
    root = dataset_root() if root is None else root
    out = []
    for rel in CANONICAL_FILES[family_stage]:
        nested, flat = os.path.join(root, rel), os.path.join(root, os.path.basename(rel))
        out.append(flat if not os.path.exists(nested) and os.path.exists(flat) else nested)
    return out


def _read_group(g, family, stage, key, profiles):
    attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in g.attrs.items()}
    dev = Device(family=family, stage=stage, key=key, xi=json.loads(attrs["xi"]),
                 u=g["u"][()], attrs=attrs, V=g["V"][()], J=g["J"][()])
    if profiles and "x_cm" in g:
        dev.x, dev.G, dev.phi_eq = g["x_cm"][()], g["G"][()], g["phi_eq"][()]
        dev.pots, dev.Jn, dev.Jp = g["pots"][()], g["Jn"][()], g["Jp"][()]
    return dev


def iter_devices(path: str, group: str = "sims", keys=None, profiles: bool = True,
                 only_ok: bool = True):
    """Yield the devices of one HDF5 file's group ("sims", "esquinas" or "refinados")."""
    with h5py.File(path, "r") as f:
        family, stage, block = f.attrs["familia"], f.attrs["etapa"], int(f.attrs["bloque"])
        if group not in f:
            return
        names = sorted(f[group].keys()) if keys is None else keys
        for name in names:
            g = f[group][name]
            if only_ok and g.attrs["estado"] != "ok":
                continue
            yield _read_group(g, family, stage, f"b{block:03d}/{group}/{name}", profiles)


def file_attrs(path: str) -> dict:
    with h5py.File(path, "r") as f:
        return {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}


def xi_specs(path: str):
    return json.loads(file_attrs(path)["parametros_xi"])


def device_keys(family_stage: str, root: str | None = None, group: str = "sims"):
    """Sorted keys of every converged device of one family-stage, across its blocks."""
    out = []
    for path in canonical_paths(family_stage, root):
        with h5py.File(path, "r") as f:
            block = int(f.attrs["bloque"])
            if group not in f:
                continue
            for name in sorted(f[group].keys()):
                if f[group][name].attrs["estado"] == "ok":
                    out.append(f"b{block:03d}/{group}/{name}")
    return out


def load(family_stage: str, keys, root: str | None = None, profiles: bool = True):
    """Load the named devices ("<block>/<group>/<name>") of one family-stage."""
    by_block = {}
    for k in keys:
        b, grp, name = k.split("/")
        by_block.setdefault((b, grp), []).append(name)
    out = {}
    for path in canonical_paths(family_stage, root):
        with h5py.File(path, "r") as f:
            b = f"b{int(f.attrs['bloque']):03d}"
        for (bb, grp), names in by_block.items():
            if bb == b:
                for d in iter_devices(path, grp, names, profiles, only_ok=False):
                    out[d.key] = d
    return [out[k] for k in keys]


# --- Split (Sec. 7 and Algorithm A1 (i)) -------------------------------------
def split(keys, seed: int, corners=(), fractions=(0.8, 0.1, 0.1)):
    """Partition devices by device into train/val/test with the run seed.

    Corners of the xi hypercube are added to test whatever the seed (Sec. 7).
    """
    keys = sorted(keys)
    perm = np.random.default_rng(seed).permutation(len(keys))
    n_train = int(round(fractions[0] * len(keys)))
    n_val = int(round(fractions[1] * len(keys)))
    shuffled = [keys[i] for i in perm]
    return {"train": sorted(shuffled[:n_train]),
            "val": sorted(shuffled[n_train:n_train + n_val]),
            "test": sorted(shuffled[n_train + n_val:]) + sorted(corners)}


# --- A family-stage as JAX functions of xi_hat ---------------------------------
_LAYER_MATERIALS = {"E1": [E1_MATERIAL, E1_MATERIAL], "E2": [CDS, CDTE],
                    "E3": [ETM, PEROVSKITE, HTM], "E4": [CDS, CDTE]}
_LAYER_THICKNESS = {"E1": ["L_n", "L_p"], "E2": ["L_CdS", "L_CdTe"],
                    "E3": ["L_ETL", "L_abs", "L_HTL"], "E4": ["L_CdS", "L_CdTe"]}
#: (parameter name, sign) of each layer's doping; None for an undoped layer.
_LAYER_DOPING = {"E1": [("Nd_n", 1.0), ("Na_p", -1.0)], "E2": [("Nd_CdS", 1.0), ("Na_CdTe", -1.0)],
                 "E3": [("Nd_ETM", 1.0), None, ("Na_HTM", -1.0)],
                 "E4": [("Nd_CdS", 1.0), ("Na_CdTe", -1.0)]}


class FamilyStage:
    """One family-stage: xi_hat in [0, 1]^d to every device quantity the PINN needs, in JAX.

    Built from a dataset file's attributes (`parametros_xi`, `parametros_fijos`),
    so the ranges are those the reference was sampled with (Sec. 7, Eq. 3).
    """

    def __init__(self, family: str, stage: str, specs, fixed):
        import jax.numpy as jnp

        self.family, self.stage = family, stage
        self.specs, self.fixed = list(specs), dict(fixed)
        self.names = [s["nombre"] for s in self.specs]
        self.d = len(self.specs)
        mats = _LAYER_MATERIALS[family]
        self.n_layers = len(mats)
        # Heterointerfaces (Sec. 3): none in E1, whose p-n junction is a doping step in
        # one material; every layer boundary in E2, E3 and E4.
        self.n_int = 0 if family == "E1" else self.n_layers - 1
        #: per-layer dimensionless material (A kept in cm^-1 eV^-1/2), shape (layers,)
        self.mat = {k: jnp.asarray([m[k] / (1.0 if k == "A" else scaling.UNITS[k]) for m in mats])
                    for k in MATERIAL_KEYS}
        self.contacts = contact_velocities(family)

    @classmethod
    def from_file(cls, path: str):
        a = file_attrs(path)
        return cls(a["familia"], a["etapa"], json.loads(a["parametros_xi"]),
                   json.loads(a["parametros_fijos"]))

    @classmethod
    def load(cls, family_stage: str, root: str | None = None):
        return cls.from_file(canonical_paths(family_stage, root)[0])

    def physical(self, xi_hat):
        """Inverse of Eq. (3) for every parameter; fixed ones keep their nominal value."""
        import jax.numpy as jnp

        out = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in self.fixed.items()}
        for j, s in enumerate(self.specs):
            u = xi_hat[..., j]
            if s["escala"] == "log":
                lo, hi = np.log10(s["min"]), np.log10(s["max"])
                out[s["nombre"]] = 10.0 ** (lo + u * (hi - lo))
            else:
                out[s["nombre"]] = s["min"] + u * (s["max"] - s["min"])
        return out

    def layout(self, xi_hat):
        """Thicknesses (cm), signed dopings (dimensionless) and interface x_hat_m of a device."""
        import jax.numpy as jnp

        xi = self.physical(xi_hat)
        thick = jnp.stack([jnp.broadcast_to(xi[n], jnp.shape(xi_hat)[:-1])
                           for n in _LAYER_THICKNESS[self.family]], axis=-1)
        dops = []
        for spec in _LAYER_DOPING[self.family]:
            if spec is None:
                dops.append(jnp.zeros(jnp.shape(xi_hat)[:-1]))
            else:
                dops.append(spec[1] * xi[spec[0]] / scaling.DENSITY)
        ndop = jnp.stack(dops, axis=-1)
        total = jnp.sum(thick, axis=-1)
        x_layers = jnp.cumsum(thick, axis=-1)[..., :-1] / total[..., None]
        x_m = x_layers if self.n_int else x_layers[..., :0]
        eta = xi.get("eta_PL", None)
        return {"thick": thick, "ndop": ndop, "l_total": total, "x_layers": x_layers,
                "x_m": x_m, "delta": scaling.LENGTH / total, "eta": eta}

    def layer_of(self, x_hat, x_layers):
        """Layer index at x_hat: the number of layer boundaries at or before it."""
        import jax.numpy as jnp

        return jnp.sum(x_hat[..., None] >= x_layers, axis=-1).astype(jnp.int32)

    def material(self, layer, ndop):
        """Dimensionless material dict at a layer index, with that layer's doping."""
        import jax.numpy as jnp

        layer = jnp.asarray(layer)
        m = {k: v[layer] for k, v in self.mat.items()}
        m["Ndop"] = jnp.take_along_axis(ndop, layer[..., None], axis=-1)[..., 0]
        return m
