"""Checkpoints, resumption and the results that outlive a worker session.

Algorithm A2, step 4 (Sec. 8) keeps the weights of every voltage so the whole
curve can be evaluated; this repository evaluates each voltage online instead
(AGREED.md) and keeps what a run needs to survive an interrupted session:

- an atomic local checkpoint (parameters, optimizer state, weighting state, PRNG
  key, iteration, voltage index, stage, and the evaluation records so far),
  written by a callback every N iterations and at the end of every voltage;
- resumption from it that reproduces the uninterrupted run, because collocation
  and every random draw are functions of (seed, voltage, iteration), never of a
  stateful generator;
- the light results (run manifest, metrics, per-voltage evaluation) committed to
  the orphan branch `results` of the code repository, and the latest checkpoint
  uploaded as an asset of one release per run of the same repository;
- on Colab, a mirror of the run directory on Google Drive.

The GitHub token is read only from Colab or Kaggle secrets, or from the
PINN_DPV_GH_TOKEN environment variable; it is handed to git through GIT_ASKPASS
from the environment, so it never appears in argv, in a file or in a log.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

import jax
import numpy as np

__provenance__ = {
    "revision": "research-concept-modelado-computacional-directo-e-inverso-de-celdas-solares-d-r05.md",
    "sections": ["8"],
    "equations": [],
    "invariants": [
        "checkpoint_roundtrip_exact",
        "checkpoint_write_is_atomic",
        "sync_writes_results_branch_and_asset",
        "drive_mirror_copies_run",
        "token_never_in_argv",
    ],
}

TOKEN_ENV = "PINN_DPV_GH_TOKEN"
SECRET_NAME = "GH_TOKEN"
CHECKPOINT = "checkpoint.pkl"
RESULTS_BRANCH = "results"


# --- Checkpoints ------------------------------------------------------------------
def _to_numpy(tree):
    return jax.tree_util.tree_map(lambda a: np.asarray(a) if hasattr(a, "shape") else a, tree)


def save_checkpoint(run_dir: str, state: dict) -> str:
    """Write `state` atomically to run_dir/checkpoint.pkl; return its sha256.

    The file is first written under a temporary name in the same directory and
    then moved over the old one, so a reader sees the old checkpoint or the new
    one, never a partial file.
    """
    os.makedirs(run_dir, exist_ok=True)
    payload = pickle.dumps(_to_numpy(state), protocol=pickle.HIGHEST_PROTOCOL)
    fd, tmp = tempfile.mkstemp(prefix=".ckpt-", dir=run_dir)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, os.path.join(run_dir, CHECKPOINT))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    digest = hashlib.sha256(payload).hexdigest()
    with open(os.path.join(run_dir, CHECKPOINT + ".sha256"), "w") as f:
        f.write(digest + "\n")
    return digest


def load_checkpoint(run_dir: str):
    """The saved state, or None when the run directory holds no checkpoint."""
    path = os.path.join(run_dir, CHECKPOINT)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        payload = f.read()
    side = path + ".sha256"
    if os.path.exists(side):
        with open(side) as f:
            expected = f.read().strip()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"checkpoint {path} does not match its sha256")
    return pickle.loads(payload)


def append_jsonl(run_dir: str, name: str, record: dict) -> None:
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, name), "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json(run_dir: str, name: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.join(run_dir, name)) or run_dir, exist_ok=True)
    path = os.path.join(run_dir, name)
    fd, tmp = tempfile.mkstemp(prefix=".json-", dir=os.path.dirname(path))
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


# --- Secrets ------------------------------------------------------------------------
def github_token():
    """The token from Colab secrets, Kaggle secrets or the environment; None if absent."""
    try:  # Colab
        from google.colab import userdata  # type: ignore

        tok = userdata.get(SECRET_NAME)
        if tok:
            return tok
    except Exception:  # noqa: BLE001 - not on Colab, or the secret is not granted
        pass
    try:  # Kaggle
        from kaggle_secrets import UserSecretsClient  # type: ignore

        tok = UserSecretsClient().get_secret(SECRET_NAME)
        if tok:
            return tok
    except Exception:  # noqa: BLE001
        pass
    return os.environ.get(TOKEN_ENV) or None


def _askpass_env(token):
    """Environment that makes git read the token from TOKEN_ENV through GIT_ASKPASS."""
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    if token is None:
        return env, None
    fd, script = tempfile.mkstemp(prefix="askpass-", suffix=".sh")
    with os.fdopen(fd, "w") as f:
        # The script holds no secret: it prints a username or the env var.
        f.write('#!/bin/sh\ncase "$1" in *sername*) echo x-access-token;; '
                f'*) printf "%s" "${TOKEN_ENV}";; esac\n')
    os.chmod(script, 0o700)
    env.update({"GIT_ASKPASS": script, TOKEN_ENV: token})
    return env, script


def _git(args, cwd, env):
    return subprocess.run(["git", *args], cwd=cwd, env=env, check=True,
                          capture_output=True, text=True)


# --- Results branch ------------------------------------------------------------------
def push_results(run_dir: str, run_id: str, remote_url: str, token=None,
                 files=("run.json", "metrics.jsonl", "eval.jsonl", "summary.json"),
                 author=("pinn-dpv worker", "pinn-dpv@users.noreply.github.com")) -> str:
    """Commit the light files of a run to `results:runs/<run_id>/` and push; return the commit.

    The branch is orphan (it shares no history with the code). A concurrent
    push from another worker is absorbed by fetching and retrying.
    """
    env, script = _askpass_env(token)
    env.update({"GIT_AUTHOR_NAME": author[0], "GIT_AUTHOR_EMAIL": author[1],
                "GIT_COMMITTER_NAME": author[0], "GIT_COMMITTER_EMAIL": author[1]})
    work = tempfile.mkdtemp(prefix="results-")
    try:
        _git(["init", "-q"], work, env)
        _git(["remote", "add", "origin", remote_url], work, env)
        for attempt in range(5):
            has = subprocess.run(["git", "ls-remote", "--exit-code", "--heads", "origin", RESULTS_BRANCH],
                                 cwd=work, env=env, capture_output=True, text=True).returncode == 0
            if has:
                _git(["fetch", "-q", "--depth", "1", "origin", RESULTS_BRANCH], work, env)
                _git(["checkout", "-q", "-B", RESULTS_BRANCH, "FETCH_HEAD"], work, env)
            else:
                _git(["checkout", "-q", "--orphan", RESULTS_BRANCH], work, env)
            dest = os.path.join(work, "runs", run_id)
            os.makedirs(dest, exist_ok=True)
            for name in files:
                src = os.path.join(run_dir, name)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(dest, name))
            _git(["add", "-A"], work, env)
            if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=work, env=env).returncode == 0 \
                    and has:
                return _git(["rev-parse", "HEAD"], work, env).stdout.strip()
            _git(["commit", "-q", "-m", f"results: {run_id}"], work, env)
            push = subprocess.run(["git", "push", "-q", "origin", f"HEAD:{RESULTS_BRANCH}"],
                                  cwd=work, env=env, capture_output=True, text=True)
            if push.returncode == 0:
                return _git(["rev-parse", "HEAD"], work, env).stdout.strip()
            time.sleep(2 * (attempt + 1))
            _git(["reset", "-q", "--hard"], work, env)
        raise RuntimeError("results push failed after 5 attempts")
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if script:
            os.remove(script)


# --- Checkpoint as a release asset ------------------------------------------------------
class LocalReleases:
    """Release store on a directory: one subdirectory per tag (for tests and dry runs)."""

    def __init__(self, root: str):
        self.root = root

    def upload(self, tag: str, path: str, name: str) -> str:
        dest = os.path.join(self.root, tag)
        os.makedirs(dest, exist_ok=True)
        tmp = os.path.join(dest, "." + name)
        shutil.copy2(path, tmp)
        os.replace(tmp, os.path.join(dest, name))
        return os.path.join(dest, name)

    def download(self, tag: str, name: str, dest: str):
        src = os.path.join(self.root, tag, name)
        if not os.path.exists(src):
            return None
        shutil.copy2(src, dest)
        return dest


class GitHubReleases:
    """Release store on GitHub: one release per run tag, the asset replaced on each upload."""

    API = "https://api.github.com"
    UPLOADS = "https://uploads.github.com"

    def __init__(self, repo: str, token=None):
        self.repo, self.token = repo, token

    def _req(self, method, url, data=None, ctype="application/json"):
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Accept", "application/vnd.github+json")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        if data is not None:
            req.add_header("Content-Type", ctype)
        with urllib.request.urlopen(req, timeout=600) as r:
            body = r.read()
            return json.loads(body) if body else None

    def _release(self, tag):
        try:
            return self._req("GET", f"{self.API}/repos/{self.repo}/releases/tags/{tag}")
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
        body = json.dumps({"tag_name": tag, "name": tag, "prerelease": True,
                           "body": "Latest checkpoint of a PINN-DPV run."}).encode()
        return self._req("POST", f"{self.API}/repos/{self.repo}/releases", body)

    def upload(self, tag: str, path: str, name: str) -> str:
        rel = self._release(tag)
        for asset in rel.get("assets", []):
            if asset["name"] == name:
                self._req("DELETE", f"{self.API}/repos/{self.repo}/releases/assets/{asset['id']}")
        with open(path, "rb") as f:
            data = f.read()
        out = self._req("POST", f"{self.UPLOADS}/repos/{self.repo}/releases/{rel['id']}/assets?name={name}",
                        data, "application/octet-stream")
        return out["browser_download_url"]

    def download(self, tag: str, name: str, dest: str):
        url = f"https://github.com/{self.repo}/releases/download/{tag}/{name}"
        try:
            with urllib.request.urlopen(url, timeout=600) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f)
            return dest
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise


def release_tag(run_id: str) -> str:
    return f"ckpt-{run_id}"


def push_checkpoint(run_dir: str, run_id: str, releases) -> str:
    """Upload the current checkpoint (and its sha256) as assets of the run's release."""
    path = os.path.join(run_dir, CHECKPOINT)
    url = releases.upload(release_tag(run_id), path, CHECKPOINT)
    releases.upload(release_tag(run_id), path + ".sha256", CHECKPOINT + ".sha256")
    return url


def fetch_checkpoint(run_dir: str, run_id: str, releases) -> bool:
    """Restore the run's latest checkpoint from its release into run_dir, if one exists."""
    os.makedirs(run_dir, exist_ok=True)
    got = releases.download(release_tag(run_id), CHECKPOINT, os.path.join(run_dir, CHECKPOINT))
    if got:
        releases.download(release_tag(run_id), CHECKPOINT + ".sha256",
                          os.path.join(run_dir, CHECKPOINT + ".sha256"))
    return bool(got)


# --- Drive mirror ------------------------------------------------------------------------
def mirror(run_dir: str, mirror_root, run_id: str):
    """Copy the run directory to mirror_root/run_id (Drive on Colab); None when no mirror."""
    if not mirror_root or not os.path.isdir(mirror_root):
        return None
    dest = os.path.join(mirror_root, run_id)
    os.makedirs(dest, exist_ok=True)
    for name in os.listdir(run_dir):
        src = os.path.join(run_dir, name)
        if os.path.isfile(src) and not name.startswith("."):
            tmp = os.path.join(dest, "." + name)
            shutil.copy2(src, tmp)
            os.replace(tmp, os.path.join(dest, name))
    return dest


def restore(run_dir: str, run_id: str, mirror_root=None, releases=None):
    """Bring the newest checkpoint back: local first, then the Drive mirror, then the release."""
    if os.path.exists(os.path.join(run_dir, CHECKPOINT)):
        return "local"
    if mirror_root and os.path.exists(os.path.join(mirror_root, run_id, CHECKPOINT)):
        os.makedirs(run_dir, exist_ok=True)
        for name in os.listdir(os.path.join(mirror_root, run_id)):
            shutil.copy2(os.path.join(mirror_root, run_id, name), os.path.join(run_dir, name))
        return "mirror"
    if releases is not None and fetch_checkpoint(run_dir, run_id, releases):
        return "release"
    return None


class Syncer:
    """The callback a training loop calls: local checkpoint always, remotes on a schedule.

    `remote_url` is the git URL of the code repository (its `results` branch
    receives the light files); `releases` stores the checkpoint asset.
    Remote failures are logged into the run and never stop training.
    """

    def __init__(self, run_dir, run_id, remote_url=None, releases=None, mirror_root=None,
                 token=None, every_seconds=600.0):
        self.run_dir, self.run_id = run_dir, run_id
        self.remote_url, self.releases, self.mirror_root = remote_url, releases, mirror_root
        self.token, self.every = token, every_seconds
        self._last = -float("inf")

    def __call__(self, state, voltage_done: bool = False):
        save_checkpoint(self.run_dir, state)
        mirror(self.run_dir, self.mirror_root, self.run_id)
        now = time.monotonic()
        if voltage_done or now - self._last >= self.every:
            self._last = now
            self.push(voltage_done)

    def push(self, with_results: bool = True):
        try:
            if self.releases is not None:
                push_checkpoint(self.run_dir, self.run_id, self.releases)
            if with_results and self.remote_url:
                push_results(self.run_dir, self.run_id, self.remote_url, self.token)
        except Exception as e:  # noqa: BLE001 - a remote outage must not lose the run
            append_jsonl(self.run_dir, "sync_errors.jsonl",
                         {"time": time.time(), "error": type(e).__name__, "detail": str(e)[:300]})
