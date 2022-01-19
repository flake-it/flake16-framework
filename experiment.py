#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import shlex
import pickle
import random
import sqlite3
import requests
import itertools
import numpy as np
import subprocess as sp

from scipy import stats
from coverage import numbits
from shap import TreeExplainer
from multiprocessing import Pool
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


LOG_FILE = "log.txt"
SHAP_FILE = "shap.pkl"
TESTS_FILE = "tests.json"
SCORES_FILE = "scores.pkl"
SUBJECTS_FILE = "subjects.txt"
REQUIREMENTS_FILE = "requirements.txt"

DATA_DIR = "data"
STDOUT_DIR = "stdout"
WORK_DIR = os.path.join("/", "home", "user")
SUBJECTS_DIR = os.path.join(WORK_DIR, "subjects")
CONT_DATA_DIR = os.path.join(WORK_DIR, DATA_DIR)
HOST_DATA_DIR = os.path.join(os.getcwd(), DATA_DIR)

CONT_TIMEOUT = 7200
N_PROC = os.cpu_count()
PIP_VERSION = "pip==21.2.1"
IMAGE_NAME = "flake16framework"
NON_FLAKY, OD_FLAKY, FLAKY = 0, 1, 2
PIP_INSTALL = ["pip", "install", "-I", "--no-deps"]
N_RUNS = {"baseline": 2500, "shuffle": 2500, "testinspect": 1}

PLUGIN_BLACKLIST = (
    "-p", "no:cov", "-p", "no:flaky", "-p", "no:xdist", "-p", "no:sugar",
    "-p", "no:replay", "-p", "no:forked",  "-p", "no:ordering", 
    "-p", "no:randomly", "-p", "no:flakefinder", "-p", "no:random_order",
    "-p", "no:rerunfailures",
)

PLUGINS = (
    os.path.join(WORK_DIR, "showflakes"), os.path.join(WORK_DIR, "testinspect")
)

FEATURE_NAMES = (
    "Covered Lines", "Covered Changes", "Source Covered Lines", 
    "Execution Time", "Read Count", "Write Count", "Context Switches",
    "Max. Threads", "Max. Memory", "AST Depth", "Assertions", 
    "External Modules", "Halstead Volume", "Cyclomatic Complexity", 
    "Test Lines of Code", "Maintainability"
)

CONFIG_GRID = (
    {
        "NOD": FLAKY,
        "OD": OD_FLAKY
    },
    {
        "Flake16": range(len(FEATURE_NAMES)),
        "FlakeFlagger": (0, 1, 2, 3, 10, 11, 14)
    },
    {
        "None": None,
        "Scaling": StandardScaler(),
        "PCA": Pipeline([("s", StandardScaler()), ("p", PCA(random_state=0))])
    },
    {
        "None": None,
        "Tomek Links": TomekLinks(),
        "SMOTE": SMOTE(random_state=0),
        "ENN": EditedNearestNeighbours(),
        "SMOTE ENN": SMOTEENN(random_state=0),
        "SMOTE Tomek": SMOTETomek(random_state=0)
    },
    {
        "Extra Trees": ExtraTreesClassifier(random_state=0),
        "Random Forest": RandomForestClassifier(random_state=0),
        "Decision Tree": DecisionTreeClassifier(random_state=0)
    }
)


def iter_subjects():
    with open(SUBJECTS_FILE, "r") as fd:
        for line in fd:
            repo, *rest = line.strip().split(",")
            yield repo.split("/", 1)[1], repo, *rest


def setup_project(proj, url, sha, package_dir):
    proj_dir = os.path.join(SUBJECTS_DIR, proj, proj)
    venv_dir = os.path.join(SUBJECTS_DIR, proj, "venv")
    requirements_file = os.path.join(SUBJECTS_DIR, proj, REQUIREMENTS_FILE)
    
    env = os.environ.copy()
    env["PATH"] = os.path.join(venv_dir, "bin") + ":" + env["PATH"]

    sp.run(["virtualenv", venv_dir], check=True)
    sp.run(["git", "clone", url, proj_dir], check=True)
    sp.run(["git", "reset", "--hard", sha], cwd=proj_dir, check=True)

    package_dir = os.path.join(proj_dir, package_dir)
    sp.run([*PIP_INSTALL, PIP_VERSION], env=env, check=True)
    sp.run([*PIP_INSTALL, "-r", requirements_file], env=env, check=True)
    sp.run([*PIP_INSTALL, *PLUGINS, "-e", package_dir], env=env, check=True)


def setup_image():
    args = []
    os.makedirs(CONT_DATA_DIR, exist_ok=True)

    for proj, repo, sha, package_dir, *_ in iter_subjects():
        args.append((proj, f"https://github.com/{repo}", sha, package_dir))

    with Pool(processes=N_PROC) as pool:
        pool.starmap(setup_project, args)


def manage_container(cont_name, *commands):
    proj, mode, _ = cont_name.split("_", 2)
    proj_dir = os.path.join(SUBJECTS_DIR, proj, proj)
    data_file = os.path.join(CONT_DATA_DIR, cont_name)
    bin_dir = os.path.join(SUBJECTS_DIR, proj, "venv", "bin")

    env = os.environ.copy()
    env["PATH"] = bin_dir + ":" + env["PATH"]

    for cmd in commands[:-1]:
        sp.run(shlex.split(cmd), cwd=proj_dir, env=env, check=True)

    sp.run(
        [
            *shlex.split(commands[-1]), *PLUGIN_BLACKLIST, "--set-exitstatus",
            *{
                "testinspect": [f"--testinspect={data_file}"],
                "baseline": [f"--record-file={data_file}.tsv"],
                "shuffle": [f"--record-file={data_file}.tsv", "--shuffle"],
            }.get(mode)
        ],
        timeout=CONT_TIMEOUT, cwd=proj_dir, check=True, env=env
    )


def run_container(args):
    cont_name, commands = args
    stdout_file = os.path.join(STDOUT_DIR, cont_name)

    with open(stdout_file, "a") as fd:
        proc = sp.run(
            [
                "docker", "run", "-it", 
                f"-v={HOST_DATA_DIR}:{CONT_DATA_DIR}:rw", "--rm", "--init", 
                "--cpus=1", f"--name={cont_name}", IMAGE_NAME, "python3", 
                "experiment.py", "container", cont_name, *commands
            ],
            stdout=fd
        )

    succeeded = proc.returncode == 0
    message = "succeeded" if succeeded else "failed"
    return f"{message}: {cont_name}", (succeeded, cont_name)


def iter_containers(run_modes):
    for proj, _, _, _, *commands in iter_subjects():
        for mode in set(run_modes):
            for run_n in range(N_RUNS[mode]):
                yield f"{proj}_{mode}_{run_n}", commands


def manage_pool(pool, fn, args):
    n_finish = 0
    t_start = time.time()

    random.shuffle(args)
    sys.stdout.write(f"0/{len(args)} 0/?\r")

    for message, result in pool.imap_unordered(fn, args):
        n_finish += 1
        n_remain = len(args) - n_finish

        t_elapse = time.time() - t_start
        t_remain = t_elapse / n_finish * n_remain

        t_elapse = round(t_elapse / 60)
        t_remain = round(t_remain / 60)

        sys.stdout.write(f"{message}\n\r")
        sys.stdout.write(f"{n_finish}/{n_remain} {t_elapse}/{t_remain}\r")

        yield result


def run_experiment(*run_modes):
    log = []
    args = []
    exitstatus = 0
    
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(STDOUT_DIR, exist_ok=True)

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as fd:
            for cont_name in fd:
                log.append(cont_name.strip())

    for cont_name, commands in iter_containers(run_modes):
        if cont_name not in log:
            args.append((cont_name, commands))

    with Pool(processes=N_PROC) as pool:
        for succeeded, cont_name in manage_pool(pool, run_container, args):
            if succeeded:
                with open(LOG_FILE, "a") as fd:
                    fd.write(f"{cont_name}\n")
            else:
                exitstatus = 1

    sys.exit(exitstatus)


def iter_data_dir():
    for file_name in os.listdir(DATA_DIR):
        proj, mode, rest = file_name.split("_", 2)
        run_n, ext = rest.split(".", 1)

        yield os.path.join(DATA_DIR, file_name), proj, mode, int(run_n), ext


def iter_tsv(fd, n_split):
    for line in fd:
        yield line.strip().split("\t", n_split)


def get_test_data_nid(collated_proj, nid):
    test_data = collated_proj[0]
    return test_data.setdefault(nid, [{}, {}, None, None])


def update_collated_runs(fd, mode, run_n, collated_proj):
    for outcome, nid in iter_tsv(fd, 1):
        runs_nid = get_test_data_nid(collated_proj, nid)[0]
        runs_mode = runs_nid.setdefault(mode, [0, 0, None, None])
        runs_mode[0] += 1

        if "failed" in outcome:
            runs_mode[1] += 1

            if runs_mode[2] is None:
                runs_mode[2] = run_n
            else:
                runs_mode[2] = min(runs_mode[2], run_n)
        else:
            if runs_mode[3] is None:
                runs_mode[3] = run_n
            else:
                runs_mode[3] = min(runs_mode[3], run_n)


def update_collated_cov(con, proj, collated_proj):
    nodeids = {}
    cur = con.cursor()
    cur.execute("SELECT id, context FROM context")

    for context_id, nid in cur.fetchall():
        nodeids[context_id] = nid

    files = {}
    cur.execute("SELECT id, path FROM file")
    proj_dir = os.path.join(SUBJECTS_DIR, proj, proj)

    for file_id, file_name in cur.fetchall():
        files[file_id] = os.path.relpath(file_name, start=proj_dir)

    cur.execute("SELECT context_id, file_id, numbits FROM line_bits")

    for context_id, file_id, nb in cur.fetchall():
        cov_nid = get_test_data_nid(collated_proj, nodeids[context_id])[1]
        cov_nid[files[file_id]] = set(numbits.numbits_to_nums(nb))


def update_collated_rusage(fd, collated_proj):
    for *rusage, nid in iter_tsv(fd, 6):
        test_data_nid = get_test_data_nid(collated_proj, nid)
        test_data_nid[2] = [float(x) for x in rusage]


def update_collated_static(fd, collated_proj):
    test_fn_ids, *collated_proj[1:] = pickle.load(fd)

    for nid, fid in test_fn_ids.items():
        test_data_nid = get_test_data_nid(collated_proj, nid)
        test_data_nid[3] = fid


def get_collated():
    collated = {}

    for file_name, proj, mode, run_n, ext in iter_data_dir():
        collated_proj = collated.setdefault(proj, [{}, None, None, None])

        if mode in {"baseline", "shuffle"}:
            with open(file_name, "r") as fd:
                update_collated_runs(fd, mode, run_n, collated_proj)
        elif mode == "testinspect":
            if ext == "sqlite3":
                with sqlite3.connect(file_name) as con:
                    update_collated_cov(con, proj, collated_proj)
            elif ext == "tsv":
                with open(file_name, "r") as fd:
                    update_collated_rusage(fd, collated_proj)
            elif ext == "pkl":
                with open(file_name, "rb") as fd:
                    update_collated_static(fd, collated_proj)

    return collated


def get_req_runs_label_nid(runs_nid):
    runs_baseline = runs_nid.get("baseline", [0, 0, None, None])
    runs_shuffle = runs_nid.get("shuffle", [0, 0, None, None])

    if runs_baseline[0] != N_RUNS["baseline"] or (
        runs_shuffle[0] != N_RUNS["shuffle"]
    ):
        return 0, None

    if runs_baseline[1] == 0:
        if runs_shuffle[1] == 0:
            return 0, NON_FLAKY
        else:
            return runs_shuffle[2], OD_FLAKY
    elif runs_baseline[1] == runs_baseline[0]:
        if runs_shuffle[1] == runs_shuffle[0]:
            return 0, NON_FLAKY
        else:
            return runs_shuffle[3], OD_FLAKY
    else:
        return max(runs_baseline[2], runs_baseline[3]), FLAKY


def get_features_nid_cov(cov_nid, test_files, churn):
    n_lines = n_changes = n_src_lines = 0

    for file_name, cov_file in cov_nid.items():
        n_lines += len(cov_file)
        churn_file = churn.get(file_name, {})
        n_changes += sum(churn_file.get(l_no, 0) for l_no in cov_file)

        if file_name not in test_files:
            n_src_lines += len(cov_file)

    return n_lines, n_changes, n_src_lines


def write_tests():
    collated = get_collated()
    tests = {}

    for proj in sorted(collated.keys(), key=lambda s: s.lower()):
        if not all(collated[proj]):
            continue

        test_data, test_fn_data, test_files, churn = collated[proj]
        tests_proj = {}

        for nid in sorted(test_data.keys(), key=lambda s: s.lower()):
            if not all(test_data[nid]):
                continue

            runs_nid, cov_nid, rusage_nid, fid = test_data[nid]
            req_runs_nid, label_nid = get_req_runs_label_nid(runs_nid)

            if label_nid is None:
                continue

            tests_proj[nid] = (
                req_runs_nid, label_nid, 
                *get_features_nid_cov(cov_nid, test_files, churn), *rusage_nid, 
                *test_fn_data[fid]
            )

        if tests_proj: 
            tests[proj] = tests_proj

    with open(TESTS_FILE, "w") as fd:
        json.dump(tests, fd, indent=4)


def load_feat_lab_proj(flaky_label, feature_set):
    with open(TESTS_FILE, "r") as fd:
        tests = json.load(fd)

    features, labels, projects = [], [], []

    for proj, tests_proj in tests.items():
        projects += [proj] * len(tests_proj)

        for (_, label_nid, *features_nid) in tests_proj.values():
            features.append(features_nid)
            labels.append(label_nid)

    features = np.array(features)[:,feature_set]
    labels = np.array(labels) == flaky_label
    projects = np.array(projects)

    return features, labels, projects


def div_none(a, b):
    return a / b if b else None


def get_prf(fp, fn, tp):
    p = div_none(tp, tp + fp)
    r = div_none(tp, tp + fn)

    if p is None or r is None:
        f = None
    else:
        f = div_none(2 * p * r, p + r)

    return p, r, f


def get_scores(config_keys):
    config_vals = [CONFIG_GRID[i][k] for i, k in enumerate(config_keys)]
    flaky_label, feature_set, preprocessing, balancing, model = config_vals
    features, labels, projects = load_feat_lab_proj(flaky_label, feature_set)
    fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    if preprocessing is not None:
        features = preprocessing.fit_transform(features)

    t_train = t_test = 0
    scores, scores_total = {proj: [0] * 6 for proj in projects}, [0] * 6

    for i, (train, test) in enumerate(fold.split(features, labels)):
        features_train, labels_train = features[train], labels[train]
        features_test, labels_test = features[test], labels[test]
        projects_test = projects[test]

        if balancing is not None:
            features_train, labels_train = balancing.fit_resample(
                features_train, labels_train
            )

        t_start = time.time()
        model.fit(features_train, labels_train)
        t_train += time.time() - t_start

        t_start = time.time()
        labels_pred = model.predict(features_test)
        t_test += time.time() - t_start

        for j, labels_test_j in enumerate(labels_test):
            k = int(2 * labels_test_j + labels_pred[j]) - 1

            if k == -1:
                continue

            scores[projects_test[j]][k] += 1
            scores_total[k] += 1

    for scores_proj in [*scores.values(), scores_total]:
        scores_proj[3:] = get_prf(*scores_proj[:3])

    return ", ".join(config_keys), (
        config_keys, t_train / 10, t_test / 10, scores, scores_total
    )


def write_scores():
    args = list(itertools.product(*[d.keys() for d in CONFIG_GRID]))

    with Pool(processes=N_PROC) as pool:
        result = manage_pool(pool, get_scores, args)
        scores = {config_keys: rest for config_keys, *rest in result}

    with open(SCORES_FILE, "wb") as fd:
        pickle.dump(scores, fd)


def get_shap(config_keys):
    config_vals = [CONFIG_GRID[i][k] for i, k in enumerate(config_keys)]
    flaky_label, feature_set, preprocessing, balancing, model = config_vals
    features, labels, _ = load_feat_lab_proj(flaky_label, feature_set)

    if preprocessing is not None:
        features = preprocessing.fit_transform(features)

    if balancing is not None:
        model.fit(*balancing.fit_resample(features, labels))
    else:
        model.fit(feature, labels)

    return TreeExplainer(model).shap_values(features)[0]


def write_shap():
    with Pool(processes=N_PROC) as pool:
        shap = pool.map(
            get_shap, (
                ("NOD", "Flake16", "Scaling", "SMOTE Tomek", "Extra Trees"),
                ("OD", "Flake16", "Scaling", "SMOTE", "Random Forest")
            )
        )

    with open(SHAP_FILE, "wb") as fd:
        pickle.dump(shap, fd)  


def get_n_stars(repo):
    info = requests.get(f"https://api.github.com/repos/{repo}").json()
    return info.get("stargazers_count", -1)


def get_req_runs_plot_coords(req_runs):
    coords = [[100 * (i + 1), 0] for i in range(25)]

    for c in coords:
        for runs, freq in req_runs.items():
            c[1] += (runs <= c[0]) * freq

    return " ".join(f"({x},{y / coords[24][1]})" for x, y in coords)


def write_req_runs_plot(req_runs_nod, req_runs_od):
    with open("req-runs.tex", "w") as fd:
        coords = get_req_runs_plot_coords(req_runs_nod)
        fd.write(f"\\addplot[mark=x,only marks] coordinates {{{coords}}};\n")
        fd.write("\\addlegendentry{NOD}\n")

        coords = get_req_runs_plot_coords(req_runs_od)
        fd.write(f"\\addplot[mark=o,only marks] coordinates {{{coords}}};\n")
        fd.write("\\addlegendentry{OD}")


def get_top_tables(scores):
    configs = [[] for _ in range(4)]

    for config_keys in scores:
        flaky_type, feature_set, *rest = config_keys
        t_train, t_test, _, (*_, f) = scores[config_keys]
        i = 2 * (flaky_type == "OD") + (feature_set == "Flake16")
        configs[i].append((*rest, t_train, t_test, f))

    for i in range(4):
        configs[i] = [c for c in configs[i] if c[-1] is not None]
        configs[i] = sorted(configs[i], key=lambda c: -c[-1])

    tab_nod = [[configs[0][i] + configs[1][i] for i in range(10)]]
    tab_od = [[configs[2][i] + configs[3][i] for i in range(10)]]
    return tab_nod, tab_od


def get_comparison_table(scores_orig, scores_ext):
    orig, orig_total = scores_orig[2:]
    ext, ext_total = scores_ext[2:]
    tab = []

    for proj, orig_proj in orig.items():
        if all(all(x is not None for x in y) for y in (orig_proj, ext[proj])):
            tab.append([proj, *orig_proj, *ext[proj]])

    return [tab, [["{\\bf Total}", *orig_total, *ext_total]]]


def get_shap_table(shap_nod, shap_od):
    shap_nod = sorted(
        zip(FEATURE_NAMES, abs(shap_nod).mean(axis=0)), key=lambda x: -x[1]
    )

    shap_od = sorted(
        zip(FEATURE_NAMES, abs(shap_od).mean(axis=0)), key=lambda x: -x[1]
    )
    
    return [[shap_nod[i] + shap_od[i] for i in range(len(FEATURE_NAMES))]]


def cellfn_default(cell):
    if isinstance(cell, str):
        return cell
    elif isinstance(cell, float):
        return "%.2f" % cell
    elif isinstance(cell, (int, np.int64)):
        return "-" if cell == 0 else str(cell)


def cellfn_corr(cell):
    if isinstance(cell, str):
        return cell
    elif isinstance(cell, float):
        return "\\cellcolor{gray!%d} %.2f" % (int(50 * abs(cell)), cell)


def cellfn_shap(cell):
    if isinstance(cell, str):
        return cell
    elif isinstance(cell, float):
        return "%.3f" % cell


def write_table(table_file, tab, rowcol=True, cellfn=cellfn_default):
    with open(table_file, "w") as fd:
        for i, tab_i in enumerate(tab):
            i and fd.write("\\midrule\n")

            for j, tab_j in enumerate(tab_i):
                rowcol and j % 2 and fd.write("\\rowcolor{gray!20}\n")
                fd.write(" & ".join([cellfn(c) for c in tab_j]) + " \\\\\n")


def write_figures():
    with open(TESTS_FILE, "r") as fd:
        tests = json.load(fd)

    tab_tests = [[], [["{\\bf Total}", *[0] * 4]]]
    req_runs_nod, req_runs_od = {}, {}
    features = []

    for i, (proj, repo, *_) in enumerate(iter_subjects()):
        tab_tests[0].append([repo, get_n_stars(repo), len(tests[proj]), 0, 0])

        for (req_runs, label_nid, *features_nid) in tests[proj].values(): 
            if label_nid == FLAKY:
                tab_tests[0][i][3] += 1
                req_runs_nod[req_runs] = req_runs_nod.get(req_runs, 0) + 1
            elif label_nid == OD_FLAKY:
                tab_tests[0][i][4] += 1
                req_runs_od[req_runs] = req_runs_od.get(req_runs, 0) + 1

            features.append(features_nid)
            
        for j in range(1, 5):
            tab_tests[1][0][j] += tab_tests[0][i][j]

    write_table("tests.tex", tab_tests)
    write_req_runs_plot(req_runs_nod, req_runs_od)

    corr = stats.spearmanr(features).correlation
    tab_corr = [[[f_i, *corr[i]] for i, f_i in enumerate(FEATURE_NAMES)]]
    write_table("corr.tex", tab_corr, rowcol=False, cellfn=cellfn_corr)

    with open(SCORES_FILE, "rb") as fd:
        scores = pickle.load(fd)

    tab_nod_top, tab_od_top = get_top_tables(scores)    
    write_table("nod-top.tex", tab_nod_top)
    write_table("od-top.tex", tab_od_top)

    tab_nod_comp = get_comparison_table(
        scores[("NOD", "FlakeFlagger", "None", "Tomek Links", "Extra Trees")],
        scores[("NOD", "Flake16", "PCA", "SMOTE", "Extra Trees")]
    )

    write_table("nod-comp.tex", tab_nod_comp)

    tab_od_comp = get_comparison_table(
        scores[("OD", "FlakeFlagger", "None", "SMOTE Tomek", "Extra Trees")],
        scores[("OD", "Flake16", "Scaling", "SMOTE", "Random Forest")]
    )

    write_table("od-comp.tex", tab_od_comp)

    with open(SHAP_FILE, "rb") as fd:
        shap_nod, shap_od = pickle.load(fd)

    tab_shap = get_shap_table(shap_nod, shap_od)
    write_table("shap.tex", tab_shap, cellfn=cellfn_shap)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command, *args = sys.argv[1:]

        if command == "setup":
            setup_image()
        elif command == "container":
            manage_container(*args)
        elif command == "run":
            run_experiment(*args)
        elif command == "tests":
            write_tests()
        elif command == "scores":
            write_scores()
        elif command == "shap":
            write_shap()
        elif command == "figures":
            write_figures()
        else:
            raise ValueError("Unrecognized command given")
    else:
        raise ValueError("No command given")