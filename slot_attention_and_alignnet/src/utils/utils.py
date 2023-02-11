import os
from pathlib import Path
import haiku as hk


def forward_fn(x, debug, net, cfg):
    return net(cfg)(x, debug)


def denorm(x):
    x = (x / 2.0) + 0.5
    x = x.clip(0, 1)
    return x


def get_run_path(basename, output_dir):
    max_run_num = 0
    for run_folder in os.listdir(output_dir):
        if basename == run_folder[: len(basename)]:
            run_num = int(run_folder.split("_")[-1])
            if run_num > max_run_num:
                max_run_num = run_num
    outpath = Path(output_dir) / f"{basename}_run_{max_run_num+1}"
    os.mkdir(outpath)
    os.mkdir(outpath / "model")
    os.mkdir(outpath / "outputs")
    return outpath


def strfdelta(tdelta):
    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    if d["days"] > 0:
        return "{days} days {hours:02d}:{minutes:02d}:{seconds:02d}".format(**d)
    if d["hours"] > 0:
        return "{hours:02d}:{minutes:02d}:{seconds:02d}".format(**d)
    return "{minutes:02d}:{seconds:02d}".format(**d)


class objdict(dict):
    def __init__(self, dict):
        if dict is not None:
            [self.__setattr__(k, v) for k, v in dict.items()]

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


def rename_treemap_branches(params, rename_tuples):
    """Takes a haiku treemap datastructured (e.g. model params or state) and a list
    of the form [('old branch subname','new branch subname'), ...]

    Returns tree with renamed branches
    """

    if params is not None:  # Loaded model may have no associated state
        params = hk.data_structures.to_mutable_dict(params)
        initial_names = list(params.keys())
        for layer_name in initial_names:
            mapped_name = layer_name
            for (old_name, new_name) in rename_tuples:
                mapped_name = mapped_name.replace(old_name, new_name)

            params[mapped_name] = params[layer_name]
            if mapped_name != layer_name:
                params.pop(layer_name)
    return params


def split_treemap(
    trainable_params, trainable_state, loaded_model, partition_string=None
):
    loaded_params, loaded_state = loaded_model
    if loaded_params is not None:
        if partition_string is not None:
            # NOTE doesn't support fine-tuning - i.e. loaded_params = Frozen if we are partitioning
            trainable_params, _ = hk.data_structures.partition(
                lambda m, n, p: partition_string in m, trainable_params
            )
            trainable_state, _ = hk.data_structures.partition(
                lambda m, n, p: partition_string in m, trainable_state
            )
        else:  # NOTE This assumes resuming from a checkpoint, but no option for pure testing
            trainable_params = loaded_params
            trainable_state = loaded_state
            loaded_params = None
            loaded_state = None
    return trainable_params, trainable_state, loaded_params, loaded_state
