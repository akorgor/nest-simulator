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
        self._load_cfg()
        self._initialize_save_dirs()
        self._save_cfg()
        self.loss = []
        self.data_file_list = []
        self.neuron_types = dict()
        self.rng = np.random.default_rng(self.cfg["seed"])

    def _initialize_save_dirs(self):
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

    def _deep_update(self, orig, new):
        for key, val in new.items():
            if isinstance(orig.get(key), dict) and isinstance(val, Mapping):
                self._deep_update(orig[key], val)
            else:
                orig[key] = val

    def _validate_keys(self, reference, overrides):
        for key in overrides:
            if key not in reference and not key.startswith("job"):
                raise KeyError(f"Unknown config key: '{key}'")
            if isinstance(overrides[key], dict) and isinstance(reference.get(key), dict):
                self._validate_keys(reference[key], overrides[key])

    def _load_cfg(self):
        cfg_path = self.file_path.parent / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                overrides = json.load(f)
            self._validate_keys(self.cfg, overrides)
            self._deep_update(self.cfg, overrides)

    def _save_cfg(self):
        with open(self.path_recordings_dir / "config_derived.json", "w") as file:
            json.dump(self.cfg, file, indent=4, sort_keys=True)

    def _sample_recordable_connections(self, nrns_inp, nrns_rec, nrns_out, n_record_w):
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
        senders, receivers = self._sample_recordable_connections(nrns_inp, nrns_rec, nrns_out, n_record_w)
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
            signs = []

            for sender in senders_arr:
                sender = int(sender)

                if sender not in self.neuron_types:
                    self.neuron_types[sender] = 1.0 if self.rng.random() < self.cfg["exc_neuron_fraction"] else -1.0

                signs.append(self.neuron_types[sender])

            signs_arr = np.asarray(signs, dtype=float)
            weights_arr = np.abs(weights_arr) * signs_arr

        elif weight_sign_fixed:
            weights_arr = np.where(weights_arr >= 0.0, np.abs(weights_arr), -np.abs(weights_arr))

        exc_mask = weights_arr >= 0.0
        inh_mask = weights_arr < 0.0

        for mask, suffix, sign_params in (
            (
                exc_mask,
                "exc",
                dict(weight=0.0, optimizer=dict(Wmin=0.0, Wmax=100.0)),
            ),
            (
                inh_mask,
                "inh",
                dict(weight=-100.0, optimizer=dict(Wmin=-100.0, Wmax=0.0)),
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
                conn_spec="one_to_one",
                syn_spec={
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

        field_rename_map = dict(
            source="sender",
            target="receiver",
        )

        keys = list(data.keys())
        out_keys = [field_rename_map.get(k, k) for k in keys]
        n = len(next(iter(data.values()))) if keys else 0

        with open(path, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(out_keys)
            for i in range(n):
                w.writerow([data[k][i] for k in keys])

        del data, conns

    def save_recordings(self, recorder_label):

        field_rename_map = dict(
            senders="sender",
            time_ms="time",
            times="time",
            weights="weight",
            receptors="receptor",
            ports="port",
            targets="receiver",
        )

        file_path_main = self.path_recordings_dir / f"{recorder_label}.csv"
        self.data_file_list.append(file_path_main.stem)

        writer_main = None
        file_main = None
        fieldnames_main = None

        flush_every = 10000
        row_counter = 0

        try:
            for file_path_input in sorted(self.path_recordings_dir.glob(f"*{recorder_label}*")):
                if not (file_path_input.name.endswith(".dat") or file_path_input.name.endswith(".csv")):
                    continue

                if file_path_input == file_path_main:
                    continue

                with open(file_path_input, newline="") as file_input:
                    if file_path_input.name.endswith(".dat"):
                        for _ in range(2):
                            next(file_input, None)
                        reader = csv.DictReader(file_input, delimiter="\t")
                    else:
                        reader = csv.DictReader(file_input)

                    fieldnames_input = reader.fieldnames or []
                    fieldnames_output = []
                    fieldnames_seen = set()
                    for fieldname_input in fieldnames_input:
                        fieldname_output = field_rename_map.get(fieldname_input, fieldname_input)
                        if fieldname_output not in fieldnames_seen:
                            fieldnames_output.append(fieldname_output)
                            fieldnames_seen.add(fieldname_output)

                    if not fieldnames_output:
                        file_path_input.unlink()
                        continue

                    if writer_main is None:
                        file_main = open(file_path_main, "w", newline="")
                        writer_main = csv.DictWriter(file_main, fieldnames=fieldnames_output, extrasaction="ignore")
                        writer_main.writeheader()
                        fieldnames_main = fieldnames_output

                    for row in reader:
                        row_renamed = {field_rename_map.get(k, k): v for k, v in row.items()}

                        if fieldnames_output != fieldnames_main:
                            row_main_output = {k: row_renamed.get(k, "") for k in fieldnames_main}
                        else:
                            row_main_output = row_renamed

                        writer_main.writerow(row_main_output)

                        row_counter += 1
                        if row_counter % flush_every == 0:
                            if file_main is not None:
                                file_main.flush()

                file_path_input.unlink()

        finally:
            if file_main is not None:
                file_main.flush()
                file_main.close()

    def get_events(self, prefix="", save=False):
        files = sorted(self.path_recordings_dir.glob(f"{prefix}*multimeter_learning*.dat"))
        if not files:
            empty_i = np.empty(0, dtype=np.int64)
            empty_f = np.empty(0, dtype=np.float64)
            return empty_i, empty_f, empty_f

        senders, times, readout_signals, target_signals = [], [], [], []
        out_path = self.path_recordings_dir / f"{prefix}_multimeter_learning.csv"

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
        for path in sorted(self.path_recordings_dir.glob(f"{prefix}*multimeter_learning*.dat")):
            path.unlink()

    def _make_serializable(self, obj):
        if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, dict):
            return dict((k, self._make_serializable(v)) for k, v in obj.items())
        if isinstance(obj, list):
            return [self._make_serializable(i) for i in obj]
        return obj

    def save_kernel_status(self, kernel_status):
        with open(self.path_recordings_dir / "kernel_status.json", "w") as f:
            json.dump(self._make_serializable(kernel_status), f, indent=4, sort_keys=True)

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

        loss_map = {
            "eprop_supervised_classification_evidence-accumulation.py": [
                34.58427289782617,
                36.70613320098335,
                28.76956055149927,
                31.72556125455047,
                37.86311373441664,
                28.83000346955887,
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
                53.22189789076346,
                11.47202858963620,
                15.48989069361456,
                12.25313574995387,
                11.55505826294500,
                18.69022685114927,
            ],
            "eprop_supervised_classification_neuromorphic_mnist_bsshslm_2020.py": [
                2.30169193352033,
                2.30460790228436,
                2.30318530812726,
                2.35351319497111,
                2.25478261712781,
                2.27606637088642,
            ],
            "eprop_supervised_regression_handwriting.py": [
                91.20706143549684,
                91.24365648440904,
                91.36371455870751,
                91.17114245646556,
                91.33388530139359,
            ],
            "eprop_supervised_regression_handwriting_bsshslm_2020.py": [
                91.40191610510352,
                90.53583357361666,
                89.91415022333089,
                88.54544175584948,
                86.98770239575573,
            ],
            "eprop_supervised_regression_lemniscate.py": [
                313.97780874558964,
                314.44451084333986,
                314.33470446142724,
                314.34578846060418,
                314.26966573356208,
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
                106.35465029286360,
                107.88661829581604,
                107.79733948745920,
                107.82189392351764,
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
        print(self.file_path.name)
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
            for l in self.loss:
                print(f"{l:.14f},")
        else:
            print(f"\nSUCCESS: The loss matches the reference values.\n")

    def _read_data(self, fname):
        data = pd.read_csv(self.path_recordings_dir / f"{fname}.csv", engine="c")
        return data

    def load_data(self, fname=""):
        if fname == "":
            data = dict()
            for fname in self.data_file_list:
                data[fname] = self._read_data(fname)
        else:
            data = self._read_data(fname)
        return data
