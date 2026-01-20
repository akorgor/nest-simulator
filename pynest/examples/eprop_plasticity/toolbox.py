import csv
import json
import math
import shutil
from collections import OrderedDict as odict
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import nest
import numpy as np
import pandas as pd
from cycler import cycler


class Tools:
    def __init__(self, cfg, file_path):
        cfg["file_name"] = Path(file_path).name
        self.cfg = cfg
        self.load_cfg()
        np.random.seed(self.cfg["seed"])
        self.init_results_dir()
        self.save_cfg()
        self.init_plotting()

    def init_plotting(self):
        self.colors = odict(
            blue="#2854c5",
            red="#e04b40",
            green="#25aa2c",
            gold="#f9c643",
            gray="#696969",
            orange="#f8933d",
            black="#000000",
            purple="#6f1970",
            lightorange="#fab377",
            lightred="#e98179",
            lightgreen="#66c36b",
            lightgray="#d3d3d3",
            lightblue="#b8dcfd",
            lightyellow="#fcdd91",
            mediumgray="#b8b8b8",
            pink="#fe5895",
            yellow="#f9f871",
        )

        plt.rcParams.update(
            dict(
                [
                    ("axes.spines.right", False),
                    ("axes.spines.top", False),
                    ("axes.prop_cycle", cycler(color=list(self.colors.values()))),
                    ("figure.dpi", 300),
                    ("font.family", "DejaVu Sans"),
                ]
            )
        )

    def init_results_dir(self):
        self.results_dir = Path(self.cfg["results_dir"])

        if self.cfg["remove_results_dir"] and self.results_dir.exists():
            shutil.rmtree(self.results_dir)

        self.recordings_dir = self.results_dir / "recordings"
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.cfg["recordings_dir"] = str(self.recordings_dir.resolve())

        if self.cfg["do_plotting"]:
            self.figures_dir = self.results_dir / "figures"
            self.figures_dir.mkdir(parents=True, exist_ok=True)
            self.cfg["figures_dir"] = str(self.figures_dir.resolve())

        with open(self.recordings_dir / "learning_performance.csv", "w") as f:
            f.write("iteration,phase,loss,error\n")

    def deep_update(self, orig, new):
        for key, val in new.items():
            if isinstance(orig.get(key), dict) and isinstance(val, Mapping):
                self.deep_update(orig[key], val)
            else:
                orig[key] = val

    def load_cfg(self):
        cfg_path = Path(__file__).parent / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                self.deep_update(self.cfg, json.load(f))

    def save_cfg(self):
        with open(self.recordings_dir / "config_derived.json", "w") as file:
            json.dump(self.cfg, file, indent=4)

    def constrain_weights(self, nrns, params_syn_base, params_common_syn_eprop):
        nrns_inp, nrns_rec, nrns_out = nrns
        weight_dicts = [
            dict(
                nrns_pre=nrns_inp,
                nrns_post=nrns_rec,
                constrain_sign=self.cfg["constrain_weights_sign_in"],
                constrain_dale=self.cfg["constrain_weights_dale_in"],
            ),
            dict(
                nrns_pre=nrns_rec,
                nrns_post=nrns_rec,
                constrain_sign=self.cfg["constrain_weights_sign_rec"],
                constrain_dale=self.cfg["constrain_weights_dale_rec"],
            ),
            dict(
                nrns_pre=nrns_rec,
                nrns_post=nrns_out,
                constrain_sign=self.cfg["constrain_weights_sign_out"],
                constrain_dale=self.cfg["constrain_weights_dale_out"],
            ),
        ]

        pop_pre_arr = np.array([], dtype=int)
        pop_post_arr = np.array([], dtype=int)
        weights_arr = np.array([])

        for weight_dict in weight_dicts:
            if weight_dict["constrain_sign"] or weight_dict["constrain_dale"]:
                conns = nest.GetConnections(weight_dict["nrns_pre"], weight_dict["nrns_post"])
                conns_dict = conns.get()
                conns_dict["source"] = np.array(conns_dict["source"], dtype=int)
                conns_dict["target"] = np.array(conns_dict["target"], dtype=int)
                conns_dict["weight"] = np.array(conns_dict["weight"])

                if weight_dict["constrain_dale"]:
                    source_unique = np.unique(conns_dict["source"])
                    proportion_inh = 1.0 / (1.0 + self.cfg["exc_to_inh_ratio"])
                    sources_inh = np.random.choice(
                        source_unique, int(len(source_unique) * proportion_inh), replace=False
                    )
                    new_weights = []
                    for source, weight in zip(conns_dict["source"], np.abs(conns_dict["weight"])):
                        if source in sources_inh:
                            weight *= -1.0
                        new_weights.append(weight)
                    conns_dict["weight"] = np.array(new_weights)

                nest.Disconnect(conns)

                weights_arr = np.append(weights_arr, conns_dict["weight"])
                pop_pre_arr = np.append(pop_pre_arr, conns_dict["source"])
                pop_post_arr = np.append(pop_post_arr, conns_dict["target"])

        for sign_dict in [dict(Wmin=0.0, Wmax=100.0), dict(Wmin=-100.0, Wmax=0.0)]:
            sign = np.sign(sign_dict["Wmin"])
            label = "positive" if sign >= 0.0 else "negative"

            params_common = params_common_syn_eprop.copy()
            params_common["optimizer"].update(sign_dict)
            params_common["weight"] = sign_dict["Wmin"]

            base_synapse_model = params_syn_base["synapse_model"]
            synapse_model = f"{base_synapse_model}_{label}"
            nest.CopyModel(base_synapse_model, synapse_model, params_common)

            if len(weights_arr) > 0:
                idc = np.where(weights_arr >= 0.0 if sign >= 0.0 else weights_arr < 0.0)

                params_base = params_syn_base.copy()
                params_base["synapse_model"] = synapse_model
                params_base["weight"] = weights_arr[idc]
                params_base["delay"] = np.ones_like(params_base["weight"]) * params_syn_base["delay"]

                nest.Connect(pop_pre_arr[idc], pop_post_arr[idc], conn_spec="one_to_one", syn_spec=params_base)

    def set_synapse_defaults(self, eta):
        for synapse_model in nest.synapse_models:
            if synapse_model.startswith("eprop_synapse"):
                nest.SetDefaults(synapse_model, dict(optimizer=dict(eta=eta)))

    def save_node_ids(self, pop_dict):
        path = self.recordings_dir / "node_ids.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["id", "label"])
            for label, v in pop_dict.items():
                nrn_ids = v.get("global_id")
                if isinstance(nrn_ids, int):
                    w.writerow([nrn_ids, label])
                else:
                    for nid in nrn_ids:
                        w.writerow([nid, label])

    def save_weights(self, pop_pre, pop_post, label):
        path = self.recordings_dir / f"weights_{label}.csv"

        conns = nest.GetConnections(pop_pre, pop_post)
        data = conns.get(["source", "target", "weight"])

        rename = {
            "source": "sender",
            "target": "receiver",
        }

        keys = list(data.keys())
        out_keys = [rename.get(k, k) for k in keys]
        n = len(next(iter(data.values()))) if keys else 0

        with open(path, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(out_keys)
            for i in range(n):
                w.writerow([data[k][i] for k in keys])

        del data, conns

    def save_recordings(self, recorder_label, duration):
        out_main = self.recordings_dir / f"{recorder_label}.csv"
        out_sub = self.recordings_dir / f"{recorder_label}_subset.csv"
        wrote_main = False
        wrote_sub = False

        t_margin = 50.0
        seq = duration["sequence"]
        task = duration["task"]

        rename = {
            "senders": "sender",
            "time_ms": "time",
            "times": "time",
            "weights": "weight",
            "receptors": "receptor",
            "ports": "port",
            "targets": "receiver",
        }

        def flush_rows(path, rows, fieldnames, wrote_header_flag):
            mode = "a" if wrote_header_flag else "w"
            with open(path, mode, newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                if not wrote_header_flag:
                    w.writeheader()
                w.writerows(rows)

        for fname in sorted(self.recordings_dir.glob(f"*{recorder_label}*")):
            if not (fname.name.endswith(".dat") or fname.name.endswith(".csv")):
                continue

            with open(fname, newline="") as f:
                if fname.name.endswith(".dat"):
                    for _ in range(2):
                        next(f, None)
                    reader = csv.DictReader(f, delimiter="\t")
                else:
                    reader = csv.DictReader(f)

                in_fields = reader.fieldnames or []
                out_fields = []
                seen = set()
                for k in in_fields:
                    kk = rename.get(k, k)
                    if kk not in seen:
                        out_fields.append(kk)
                        seen.add(kk)

                if not out_fields:
                    fname.unlink()
                    continue

                main_rows = []
                sub_rows = []

                for row in reader:
                    out = {rename.get(k, k): v for k, v in row.items()}

                    t = out.get("time", "")
                    try:
                        t = float(t) if t != "" and t is not None else None
                    except ValueError:
                        t = None

                    main_rows.append(out)

                    if t is not None:
                        if (t < seq + t_margin) or (t >= task - seq - t_margin):
                            sub_rows.append(out)

            if main_rows:
                flush_rows(out_main, main_rows, out_fields, wrote_main)
                wrote_main = True

            if sub_rows:
                flush_rows(out_sub, sub_rows, out_fields, wrote_sub)
                wrote_sub = True

            del main_rows, sub_rows
            fname.unlink()

    def get_events(self, prefix="", save=False):
        files = sorted(self.recordings_dir.glob(f"{prefix}*multimeter_out*.dat"))
        if not files:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

        senders = []
        readout = []
        target = []

        out_path = self.recordings_dir / f"{prefix}_multimeter_out.csv"
        wrote_header = False

        for fname in files:
            with open(fname, newline="") as f:
                reader = csv.DictReader((line for line in f if not line.startswith("#")), delimiter="\t")

                if save:
                    with open(out_path, "a", newline="") as fo:
                        w = csv.DictWriter(fo, fieldnames=reader.fieldnames)
                        if not wrote_header:
                            w.writeheader()
                            wrote_header = True
                        for row in reader:
                            senders.append(int(row["sender"]))
                            readout.append(float(row["readout_signal"]))
                            target.append(float(row["target_signal"]))
                            w.writerow(row)
                else:
                    for row in reader:
                        senders.append(int(row["sender"]))
                        readout.append(float(row["readout_signal"]))
                        target.append(float(row["target_signal"]))

            fname.unlink()

        return (
            np.asarray(senders, dtype=np.int64),
            np.asarray(readout, dtype=np.float64),
            np.asarray(target, dtype=np.float64),
        )

    def clear_events(self, prefix):
        for path in sorted(self.recordings_dir.glob(f"{prefix}*multimeter_out*.dat")):
            path.unlink()

    def load_data(self, label):
        return pd.read_csv(self.recordings_dir / f"{label}.csv", engine="c")

    def make_serializable(self, obj):
        if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, dict):
            return dict((k, self.make_serializable(v)) for k, v in obj.items())
        if isinstance(obj, list):
            return [self.make_serializable(i) for i in obj]
        return obj

    def save_kernel_status(self, kernel_status):
        with open(self.recordings_dir / "kernel_status.json", "w") as f:
            json.dump(self.make_serializable(kernel_status), f, indent=4)

    def save_performance(self, iteration, loss, errors, phase_label):
        path = self.recordings_dir / "learning_performance.csv"
        with open(path, "a", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            for l, e in zip(loss, errors):
                w.writerow([iteration, phase_label, l, e])
                iteration += 1

    def verify(self):
        self.loss = self.load_data("learning_performance").loss.values
        # print(self.cfg["file_name"])
        # for l in self.loss:
        #     print(f"{l:.14f},")
        # exit()

        if self.cfg["file_name"] == "eprop_supervised_classification_evidence-accumulation_bsshslm_2020.py":
            loss_reference = [
                0.74115255000619,
                0.74038818770074,
                0.66578523317777,
                0.66364419332299,
                0.72942896284495,
                0.65825443888416,
            ]
        elif self.cfg["file_name"] == "eprop_supervised_classification_evidence-accumulation.py":
            loss_reference = [
                34.58427289782617,
                36.87835068653019,
                28.89970643558962,
                31.60581680525203,
                36.76571948680768,
                29.90618754038629,
            ]
        elif self.cfg["file_name"] == "eprop_supervised_regression_sine-waves_bsshslm_2020.py":
            loss_reference = [
                101.96435699904158,
                103.46673112620579,
                103.34060707477168,
                103.68024403768638,
                104.41277574875247,
            ]
        elif self.cfg["file_name"] == "eprop_supervised_regression_sine-waves.py":
            loss_reference = [
                107.73732072362752,
                106.42253313316886,
                107.37869441301808,
                108.10839027499375,
                107.76400611943626,
            ]
        elif self.cfg["file_name"] == "eprop_supervised_classification_neuromorphic_mnist.py":
            loss_reference = [
                0.49569090581695,
                0.52751321436889,
                0.51467659566501,
                0.50595422166446,
                0.50532549825770,
                0.49938869752847,
            ]
        elif self.cfg["file_name"] == "eprop_supervised_classification_neuromorphic_mnist_bsshslm_2020.py":
            loss_reference = [
                # 2.29926915739071,
                # 2.30735389920452,
                # 2.31229167547814,
                # 2.30398946726470,
                # 2.30571008112245,
                # 2.30277356036807,
                2.30255632439916,
                2.30276714829263,
                2.30273512725323,
                2.29159414308686,
                2.13508458817967,
                2.85064803699591,
                1.96775491423905,
                2.25084317619386,
                2.15793814437091,
                2.68245474754681,
                2.61180476107622,
                2.91421839715668,
                2.25871717728110,
            ]
        elif self.cfg["file_name"] == "eprop_supervised_regression_lemniscate_bsshslm_2020.py":
            loss_reference = [
                314.30442538643001,
                313.84127193622919,
                312.33971633807948,
                310.66410755892281,
                309.19353500432857,
            ]
        elif self.cfg["file_name"] == "eprop_supervised_regression_handwriting_bsshslm_2020.py":
            loss_reference = [
                91.40191610510351,
                90.53583357361666,
                89.91415022333089,
                88.54544175584950,
                86.98770239575573,
            ]

        n_compare = min(len(self.loss), len(loss_reference))
        verification_successful = np.allclose(self.loss[:n_compare], loss_reference[:n_compare], atol=1e-14, rtol=0)

        if not verification_successful:
            deviation_idc = np.where(self.loss[:n_compare] != loss_reference[:n_compare])[0]
            for deviation_idx in deviation_idc:
                print(f"{deviation_idx}. iteration")
                print(f"{self.loss[deviation_idx]:.16f} loss")
                print(f"{loss_reference[deviation_idx]:.16f} reference loss")
                print(f"{self.loss[deviation_idx]-loss_reference[deviation_idx]:.16f} delta")
        print(verification_successful)
