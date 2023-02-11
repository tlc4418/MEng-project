import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pickle
from pathlib import Path
import haiku as hk
import jax
import yaml
from src.dataloaders import DataController, DataLoader

# Imports from other stuff
from src.models import AlignedSlotAttention, SlotAttentionAE
from src.utils import forward_fn, objdict


data_path = "/media/home/thomas/data/"  # CLEVR dataloader will put data here

run_name = "align_sprite"
config_name = "alignnet_base.yaml"
debug = True
model_class = AlignedSlotAttention  # SlotAttentionAE #
# For CLEVR, we will discard labels
extract_relevant_input_fn = (
    lambda batch: batch[0] if isinstance(batch, tuple) else batch
)


def main():
    rngseq = hk.PRNGSequence(7)

    cfg = objdict(
        yaml.safe_load(open(Path("slot_attention_and_alignnet/config") / config_name))
    )

    data_controller = DataController(
        "/media/home/thomas/data/",
        file_name="aai_goals_sequence",
        test_train_split=0,
        load_mode=True,
        shuffle=False,
        batch_size=cfg["batch_size"],
        unbatch=False,
        gzip=False,
    )
    ds_val = DataLoader(data_controller, cfg, split="validation")
    performance_tests = model_class.get_performance_tests(cfg)
    test_sample_inp = extract_relevant_input_fn(next(ds_val))

    global net
    model_vis_fn = model_class.get_visualizers(cfg)
    net = hk.transform_with_state(jax.partial(forward_fn, net=model_class, cfg=cfg))
    with open(
        f"./slot_attention_and_alignnet/runs/align_sprite_run_32/model/params_100000.pkl",
        "rb",
    ) as f:
        params, state = pickle.load(f)

    fixed_net_apply = jax.jit(lambda rng, im: net.apply(params, state, rng, im, True))

    # Plot model specific visualizations
    test_sample_out, _ = fixed_net_apply(next(rngseq), test_sample_inp)
    figs = model_vis_fn(test_sample_inp, test_sample_out)
    for title, (f, caption) in figs.items():
        f.savefig(
            "./slot_attention_and_alignnet/aai_align_tests/"
            f"{title.lower().replace(' ', '_')}.pdf",
            bbox_inches="tight",
            pad_inches=0.1,
        )

    # Evaluate performance
    count = 0
    performance_history = None
    for i, val_batch in enumerate(ds_val): #[(0, test_sample_inp)]:#
        inp = extract_relevant_input_fn(val_batch)
        # inp = jnp.expand_dims(inp, axis=0)
        count += 1
        out, _ = fixed_net_apply(next(rngseq), inp)
        ith_perf_vals = performance_tests(inp, out)
        if i == 0:
            performance_history = ith_perf_vals
        else:
            performance_history = {
                k: v + ith_perf_vals[k]
                for k, v in performance_history.items()
            }

    [
        print(f"Validation Average {k}, {v/count}")
        for k, v in performance_history.items()
    ]
    print("--------------------------------------\n")


if __name__ == "__main__":
    main()
