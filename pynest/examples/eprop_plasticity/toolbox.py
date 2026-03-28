import csv
import json
import math
from collections.abc import Mapping
from pathlib import Path

import nest
import numpy as np
import pandas as pd
from IPython.display import Image


class Tools:
    def __init__(self, cfg, file_path):
        self.file_path = Path(file_path)
        self.cfg = cfg
        self.load_cfg()
        self.initialize_save_dirs()
        self.save_cfg()
        self.loss = []
        self.data_file_list = []

    def initialize_save_dirs(self):
        self.path_recordings_dir = (self.file_path.parent / self.cfg["relative_path_recordings_dir"]).resolve()
        if self.path_recordings_dir.is_dir():
            if any(self.path_recordings_dir.iterdir()):
                if self.cfg["delete_existing_recordings"]:
                    for path in self.path_recordings_dir.iterdir():
                        if path.is_file:
                            path.unlink()
                else:
                    print(
                        "\nWARNING: The recordings directory is not empty. This may cause the run to fail if overwriting is disabled or lead to incorrect results.\n"
                    )
        else:
            self.path_recordings_dir.mkdir()
        if "relative_path_data_dir" in self.cfg:
            self.path_data_dir = (self.file_path.parent / self.cfg["relative_path_data_dir"]).resolve()

    def show_image(self):
        try:
            Image(filename=self.file_path.with_suffix(".png"))
        except:
            pass

    def deep_update(self, orig, new):
        for key, val in new.items():
            if isinstance(orig.get(key), dict) and isinstance(val, Mapping):
                self.deep_update(orig[key], val)
            else:
                orig[key] = val

    def load_cfg(self):
        cfg_path = self.file_path.parent / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                self.deep_update(self.cfg, json.load(f))

    def save_cfg(self):
        with open(self.path_recordings_dir / "config_derived.json", "w") as file:
            json.dump(self.cfg, file, indent=4)

    def sample_recordable_connections(self, nrns_inp, nrns_rec, nrns_out, n_record_w):
        senders_list = []
        receivers_list = []

        rng = np.random.default_rng(self.cfg["seed"])
        for pop_pre, pop_post in (
            (nrns_inp, nrns_rec),
            (nrns_rec, nrns_rec),
            (nrns_rec, nrns_out),
        ):
            conns = nest.GetConnections(pop_pre, pop_post)
            senders = np.asarray(conns.source)
            receivers = np.asarray(conns.target)

            n_conn = len(senders)
            if n_conn == 0:
                continue

            idc = rng.choice(n_conn, size=min(n_record_w, n_conn), replace=False)
            senders_list.append(senders[idc])
            receivers_list.append(receivers[idc])

        return np.unique(np.concatenate(senders_list)), np.unique(np.concatenate(receivers_list))

    def configure_weight_recorder_connections(self, wr, nrns_inp, nrns_rec, nrns_out, n_record_w):
        senders, receivers = self.sample_recordable_connections(nrns_inp, nrns_rec, nrns_out, n_record_w)
        if len(senders) > 0:
            wr.set(senders=senders, targets=receivers)

    def constrain_weights(self, pop_sender, pop_receiver, syn_spec, label):

        weight_sign_fixed = self.cfg[f"weight_sign_fixed_{label}"]
        weight_dale_enforced = self.cfg[f"weight_dale_enforced_{label}"]
        if not (weight_sign_fixed or weight_dale_enforced):
            return

        conns = nest.GetConnections(pop_sender, pop_receiver)
        if len(conns) == 0:
            return
        senders_arr = np.asarray(conns.source)
        receivers_arr = np.asarray(conns.target)
        weights_arr = np.asarray(conns.weight, dtype=float)
        delays_arr = np.asarray(conns.delay, dtype=float)
        synapse_models = np.unique(conns.synapse_model)

        if len(synapse_models) != 1:
            raise ValueError(f"Expected exactly one synapse model, got {list(synapse_models)}")
        base_synapse_model = synapse_models[0]
        existing_synapse_models = set(nest.GetKernelStatus()["synapse_models"])

        if weight_dale_enforced:
            proportion_inh = 1.0 / (1.0 + self.cfg["exc_to_inh_ratio"])
            senders_unique = np.unique(senders_arr)

            n_senders_inh = round(len(senders_unique) * proportion_inh)

            # Better: store this once on self and reuse it.
            rng = np.random.default_rng(self.cfg["seed"])
            senders_inh = set(rng.choice(senders_unique, n_senders_inh, replace=False))

            is_inh = np.isin(senders_arr, list(senders_inh))
            weights_arr = np.where(is_inh, -np.abs(weights_arr), np.abs(weights_arr))

            exc_mask = weights_arr >= 0.0
            inh_mask = ~exc_mask

            for mask, suffix, sign_params in (
                (
                    exc_mask,
                    "exc",
                    {"weight": 0.0, "optimizer": {"Wmin": 0.0, "Wmax": 100.0}},
                ),
                (
                    inh_mask,
                    "inh",
                    {"weight": -100.0, "optimizer": {"Wmin": -100.0, "Wmax": 0.0}},
                ),
            ):
                if not np.any(mask):
                    continue

                synapse_model = f"{base_synapse_model}_{suffix}"

                if synapse_model not in existing_synapse_models:
                    nest.CopyModel(base_synapse_model, synapse_model, sign_params)
                    existing_synapse_models.add(synapse_model)

                nest.Connect(
                    senders_arr[mask],
                    receivers_arr[mask],
                    "one_to_one",
                    {
                        **syn_spec,
                        "synapse_model": synapse_model,
                        "weight": weights_arr[mask],
                        "delay": delays_arr[mask],
                    },
                )

            nest.Disconnect(conns)

    def set_synapse_defaults(self, eta):
        for synapse_model in nest.synapse_models:
            if synapse_model.startswith("eprop_synapse"):
                nest.SetDefaults(synapse_model, dict(optimizer=dict(eta=eta)))

    def save_node_ids(self, pop_dict):
        fname = "node_ids"
        path = self.path_recordings_dir / f"{fname}.csv"
        self.data_file_list.append(fname)
        with open(path, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["id", "label"])
            for label, v in pop_dict.items():
                nrn_ids = v.get("global_id")
                if isinstance(nrn_ids, int) or isinstance(nrn_ids, np.int64):
                    w.writerow([nrn_ids, label])
                else:
                    for nid in nrn_ids:
                        w.writerow([nid, label])

    def save_weights(self, pop_pre, pop_post, label):
        fname = f"weights_{label}"
        path = self.path_recordings_dir / f"{fname}.csv"
        self.data_file_list.append(fname)

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
        out_main = self.path_recordings_dir / f"{recorder_label}.csv"
        fname = f"{recorder_label}_subset"
        out_sub = self.path_recordings_dir / f"{fname}.csv"
        self.data_file_list.append(fname)

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

        for fname in sorted(self.path_recordings_dir.glob(f"*{recorder_label}*")):
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
        files = sorted(self.path_recordings_dir.glob(f"{prefix}*multimeter_out*.dat"))
        if not files:
            empty_i = np.empty(0, dtype=np.int64)
            empty_f = np.empty(0, dtype=np.float64)
            return empty_i, empty_f, empty_f

        senders, times, readout_signals, target_signals = [], [], [], []
        out_path = self.path_recordings_dir / f"{prefix}_multimeter_out.csv"

        if save and out_path.exists():
            out_path.unlink()

        for fname in files:
            with open(fname, newline="") as f:
                reader = csv.DictReader(
                    (line for line in f if not line.startswith("#")),
                    delimiter="\t",
                )

                writer = None
                if save:
                    fo = open(out_path, "a", newline="")
                    writer = csv.DictWriter(fo, fieldnames=reader.fieldnames)
                    if fo.tell() == 0:
                        writer.writeheader()

                for row in reader:
                    senders.append(row["sender"])
                    times.append(row["time_ms"])
                    readout_signals.append(row["readout_signal"])
                    target_signals.append(row["target_signal"])
                    if writer:
                        writer.writerow(row)

                if save:
                    fo.close()
                    fname.unlink()

        senders = np.asarray(senders, dtype=np.int64)
        times = np.asarray(times, dtype=np.float64)
        readout_signals = np.asarray(readout_signals, dtype=np.float64)
        target_signals = np.asarray(target_signals, dtype=np.float64)

        order = np.lexsort((times, senders))
        return senders[order], readout_signals[order], target_signals[order]

    def clear_events(self, prefix):
        for path in sorted(self.path_recordings_dir.glob(f"{prefix}*multimeter_out*.dat")):
            path.unlink()

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
        with open(self.path_recordings_dir / "kernel_status.json", "w") as f:
            json.dump(self.make_serializable(kernel_status), f, indent=4)

    def save_performance(self, iteration, loss, errors=[], phase_label=""):
        fname = "learning_performance"
        path = self.path_recordings_dir / f"{fname}.csv"
        self.data_file_list.append(fname)

        do_append_errors = len(errors) > 0

        if not path.exists():
            df = pd.DataFrame(columns=["iteration", "phase", "loss"])
            if do_append_errors:
                df["error"] = []
            df.to_csv(path, index=False)

        with open(path, "a", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            for i in range(len(loss)):
                row = [iteration]
                if phase_label != "":
                    row.append(phase_label)
                row.append(loss[i])
                if do_append_errors:
                    row.append(errors[i])
                w.writerow(row)
                iteration += 1

    def verify(self):
        self.loss = self.load_data("learning_performance").loss.to_numpy()

        # from datetime import datetime
        # from pathlib import Path
        # fname = (Path.home() / "log" / datetime.now().strftime("%Y-%m-%d_%u_%H-%M-%S")).with_suffix(".txt")
        # with open(fname, "w") as f:
        #     f.write(self.file_path.name + "\n\n")
        #     for l in self.loss:
        #         f.write(f"{l:.13f}\n")
        # exit()

        # print(self.file_path.name)
        # for l in self.loss:
        #     print(f"{l:.14f},")
        # exit()

        loss_map = {
            "eprop_supervised_classification_evidence-accumulation.py": [
                34.58427289782617,
                36.87835068653019,
                28.89970643558962,
                31.60581680525202,
                36.76571948680768,
                29.90618754038629,
            ],
            "eprop_supervised_classification_evidence-accumulation_bsshslm_2020.py": [
                0.70216337067153,
                0.73555530315184,
                0.74035486411103,
                0.68388281528182,
                0.70784112226789,
                0.67269494515383,
            ],
            "eprop_supervised_classification_neuromorphic_mnist.py": [
                10.35322836401918,
                8.33506530601532,
                9.72408176470581,
                9.25152573790276,
                4.67788466496715,
                31.78819420209928,
            ],
            "eprop_supervised_classification_neuromorphic_mnist_bsshslm_2020.py": [
                2.30246587138697,
                2.28945027983528,
                2.15313277459524,
                2.78232640765524,
                1.97565231669283,
                2.24778735962639,
            ],
            "eprop_supervised_regression_handwriting.py": [
                91.20706143549684,
                91.24365648440904,
                91.36386255798166,
                91.15808668390740,
                91.32144527401481,
            ],
            "eprop_supervised_regression_handwriting_bsshslm_2020.py": [
                91.40191610510352,
                90.53583357361666,
                89.91415022333089,
                88.54544175584948,
                86.98770239575573,
            ],
            "eprop_supervised_regression_lemniscate.py": [
                313.97823685007972,
                314.44451082112619,
                314.33470446080099,
                314.34578846087288,
                314.26966562042782,
            ],
            "eprop_supervised_regression_lemniscate_bsshslm_2020.py": [
                314.30442538643001,
                313.84127193622919,
                312.33971633807948,
                310.66410755892286,
                309.19353500432857,
            ],
            "eprop_supervised_regression_sine-waves.py": [
                107.73732072362752,
                106.42253313316886,
                107.37869441301808,
                108.10839027499374,
                107.76400611943626,
            ],
            "eprop_supervised_regression_sine-waves_bsshslm_2020.py": [
                101.96435699904158,
                103.46673112620580,
                103.34060707477168,
                103.68024403768638,
                104.41277574875248,
            ],
        }

        loss_reference = np.array(loss_map.get(self.file_path.name))

        if loss_reference is None:
            print("\nFAILURE: No reference loss.\n")
            return

        n_compare = min(len(self.loss), len(loss_reference))
        verification_successful = np.allclose(self.loss[:n_compare], loss_reference[:n_compare], atol=1e-14, rtol=0.0)

        if not verification_successful:
            deviation_idc = np.where(self.loss[:n_compare] != loss_reference[:n_compare])[0]

            for deviation_idx in deviation_idc:
                print(f"\n{deviation_idx}. iteration")
                print(f"{self.loss[deviation_idx]:.16f} loss")
                print(f"{loss_reference[deviation_idx]:.16f} loss reference")
                print(f"{self.loss[deviation_idx]-loss_reference[deviation_idx]:.16f} difference")
            print(f"\nFAILURE: The loss does not match the reference values.\n")
        else:
            print(f"\nSUCCESS: The loss matches the reference values.\n")

    def read_data(self, fname):
        data = pd.read_csv(self.path_recordings_dir / f"{fname}.csv", engine="c")
        return data

    def load_data(self, fname=""):
        if fname == "":
            data = dict()
            for fname in self.data_file_list:
                data[fname] = self.read_data(fname)
        else:
            data = self.read_data(fname)
        return data
