import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import haiku as hk
import jax
import matplotlib.pyplot as plt
import optax
import yaml

# Project imports
from src.models import AlignedSlotAttention, SlotAttentionAE
from src.utils import (
    forward_fn,
    objdict,
    rename_treemap_branches,
    split_treemap,
    get_run_path,
    strfdelta,
)
from src.dataloaders import DataController, DataLoader


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
    rngseq = hk.PRNGSequence(42)

    cfg = objdict(
        yaml.safe_load(open(Path("slot_attention_and_alignnet/config") / config_name))
    )
    
    data_controller = DataController(
        "/media/home/thomas/data/",
        file_name="aai_goals_sequence",
        test_train_split=0.9,
        load_mode=True,
        shuffle=True,
        batch_size=cfg["batch_size"],
        unbatch=False,
        gzip=False,
    )
    ds_train = DataLoader(data_controller, cfg, split="train")
    ds_val = DataLoader(data_controller, cfg, split="validation")
    test_sample_inp = extract_relevant_input_fn(next(ds_train))

    # Initialize AlignNet model and other params
    performance_tests = model_class.get_performance_tests(cfg)

    global net, opt, model_loss_fn
    model_loss_fn = model_class.get_loss(cfg)
    model_vis_fn = model_class.get_visualizers(cfg)
    net = hk.transform_with_state(jax.partial(forward_fn, net=model_class, cfg=cfg))
    params, state = net.init(
        next(rngseq),
        jax.random.normal(next(rngseq), (cfg.batch_size, *cfg.input_resolution, 3)),
        False,
    )

    # Load SA params
    with open(
        "./slot_attention_and_alignnet/runs/sa_aai_goals_closer_mlp_48_run_6/model/params_220000.pkl",
        "rb",
    ) as f:
        loaded_model = pickle.load(f)

    # This is for loading a pre-trained sub-component of a model - e.g. Slot Attention for AlignNet
    pretrain_partition_string = model_class.pretrain_partition_string
    pretrain_rename_tuples = model_class.pretrain_param_matching_tuples
    params, _, loaded_params, loaded_state = split_treemap(
        params, state, loaded_model, pretrain_partition_string
    )

    loaded_params = rename_treemap_branches(loaded_params, pretrain_rename_tuples)
    loaded_state = rename_treemap_branches(loaded_state, pretrain_rename_tuples)

    """ This does not freeze the loaded parameters - for that, one needs to keep loaded and trainable params 
        separate, and then only use the optimizer to track and update the trainable ones 
    """

    opt = model_class.get_optimizer(cfg)
    opt_state = opt.init(params)

    # Start training
    step = 0
    start = datetime.now()
    print("Started training at " + start.strftime("%d %b %Y, %H:%M:%S"))
    print(f"Model has params: {hk.data_structures.tree_size(params)} trainable params")
    loss_hist = defaultdict(list)
    val_hist = []
    outpath = get_run_path(run_name, "./slot_attention_and_alignnet/runs")
    while step <= cfg.train_steps:
        batch = extract_relevant_input_fn(next(ds_train))

        # Update, freezing loaded_params
        losses, params, state, opt_state = update(
            params,
            state,
            next(rngseq),
            opt_state,
            batch,
            non_trainable_params=loaded_params,
        )
        all_params = hk.data_structures.merge(params, loaded_params)

        if step % 5000 == 0:
            # Plot Losses
            f, ax = plt.subplots(1)
            ax.set_yscale("log")
            ax.set_xlabel("Steps")
            ax.set_ylabel("Loss")
            for k, v in loss_hist.items():
                ax.plot(v, label=k.capitalize())
            ax.legend()
            f.savefig(str(outpath / f"loss_hist_{run_name}.jpg"))

            fixed_net_apply = jax.jit(
                lambda rng, im: net.apply(all_params, state, rng, im, True)
            )

            # Plot model specific visualizations
            test_sample_out, _ = fixed_net_apply(next(rngseq), test_sample_inp)
            figs = model_vis_fn(test_sample_inp, test_sample_out)
            for title, (f, caption) in figs.items():
                f.savefig(
                    outpath
                    / "outputs"
                    / f"{title.lower().replace(' ', '_')}_{step}.pdf",
                    bbox_inches="tight",
                    pad_inches=0.1,
                )

            if step > 20000:
                # Save model
                with open(outpath / f"model/params_{step}.pkl", "wb") as f:
                    pickle.dump((all_params, state), f)
                with open(outpath / f"model/optimizer_{step}.pkl", "wb") as f:
                    pickle.dump(opt_state, f)

                if step % 20000 == 0:
                    # Perform validation of model

                    print("\nValidating Model")
                    mse_tot = 0
                    count = 0
                    performance_history = None
                    for i, val_batch in enumerate(ds_val):
                        inp = extract_relevant_input_fn(val_batch)
                        count += inp.shape[0]
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

        if step % 100 == 0:
            [loss_hist[k].append(v) for k, v in losses.items()]
            losses_str = ", ".join([f"{k}: {v:.2e}" for k, v in losses.items()])
            print(
                f"Step: {step}, Losses: {losses_str}, Elapsed: "
                + strfdelta(datetime.now() - start)
            )

        step += 1
    with open(outpath / f"loss_hist_{run_name}.pkl", "wb") as f:
        pickle.dump((loss_hist, val_hist), f)


@jax.jit
def update(params, state, rng_key, opt_state, batch, non_trainable_params={}):
    """Learning rule (stochastic gradient descent)."""
    grads, (losses, state) = jax.grad(loss_fn, 0, has_aux=1)(
        params, state, rng_key, batch, non_trainable_params
    )
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return losses, params, state, opt_state


def loss_fn(params, state, rng_key, batch, non_trainable_params={}):
    p = hk.data_structures.merge(non_trainable_params, params)
    x, state = net.apply(p, state, rng_key, batch, False)
    loss = model_loss_fn(x, batch)
    return loss["total"], (loss, state)


if __name__ == "__main__":
    main()
