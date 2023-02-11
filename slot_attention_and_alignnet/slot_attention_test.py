import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pickle
from pathlib import Path
import haiku as hk
import jax
import yaml

# Project imports
from src.models import AlignedSlotAttention, BGSlotAttentionAE, SlotAttentionAE
from src.utils import ari_score, forward_fn, get_true_masks_from_inp, objdict
from src.dataloaders import DataController, DataLoader, CLEVRHansLoader

ARI_BATCH_SIZE = 0

data_path = "/media/home/thomas/data/"  # CLEVR dataloader will put data here
run_name = "slot_attention_test_load_ari"
config_name = "slot_attention_clevr.yaml"
debug = True
model_class = SlotAttentionAE  # SlotAttentionAE #
# For CLEVR, we will discard labels
extract_relevant_input_fn = (
    lambda batch: batch[0] if isinstance(batch, tuple) else batch
)


def main():
    rngseq = hk.PRNGSequence(21)

    cfg = objdict(
        yaml.safe_load(open(Path("slot_attention_and_alignnet/config") / config_name))
    )

    # Load dataset

    # data_controller = DataController(
    #     "/media/home/thomas/data/",
    #     file_name="aai_random_many_objects",
    #     test_train_split=0.8,
    #     load_mode=True,
    #     shuffle=True,
    #     batch_size=cfg["batch_size"],
    #     unbatch=True,
    #     # gzip=True,
    #     # masks=True
    # )
    # ds_val = DataLoader(data_controller, cfg, masks=False, split="validation")

    # ds_val = CLEVRHansLoader(data_path, cfg, "validation", variant=3, get_masks=True)

    data_controller = DataController(
        "/media/home/thomas/data/",
        file_name="spriteworld_test",
        test_train_split=0,
        load_mode=True,
        shuffle=False,
        batch_size=cfg["batch_size"],
        unbatch=True,
        gzip=True,
        masks=False,
    )
    ds_val = DataLoader(data_controller, cfg, split="validation", masks=False)
    inp_images = next(ds_val)
    test_sample_inp = extract_relevant_input_fn(inp_images)

    # Load model and other params
    performance_tests = model_class.get_performance_tests(cfg)

    global net
    model_vis_fn = model_class.get_visualizers(cfg)
    net = hk.transform_with_state(jax.partial(forward_fn, net=model_class, cfg=cfg))
    with open(
        f"./slot_attention_and_alignnet/runs/sa_sprite_run_4_resume_run_1/model/params_250000.pkl",
        "rb",
    ) as f:
        params, state = pickle.load(f)
    fixed_net_apply = jax.jit(lambda rng, im: net.apply(params, state, rng, im, True))

    # Plot model specific visualizations
    test_sample_out, _ = fixed_net_apply(next(rngseq), test_sample_inp)
    figs = model_vis_fn(test_sample_inp, test_sample_out)
    for title, (f, caption) in figs.items():
        f.savefig(
            "./slot_attention_and_alignnet/aai_sa_tests/"
            f"{title.lower().replace(' ', '_')}.pdf",
            bbox_inches="tight",
            pad_inches=0.1,
        )

    count = 0
    # ari_count = 0
    # ari_total = 0
    # performance_history = None
    for i, inp in enumerate(ds_val):
        #     inp, true_masks = inp
        out, _ = fixed_net_apply(next(rngseq), inp)

        #     # ARI metric
        #     if ari_count < ARI_BATCH_SIZE:
        #         # true_masks = get_true_masks_from_inp(inp)
        #         new_ari, processed = ari_score(
        #             true_masks=true_masks,
        #             pred_masks=out["masks"],
        #             max_process=ARI_BATCH_SIZE - ari_count,
        #         )
        #         ari_total = ((ari_total * ari_count) + (new_ari * processed)) / (
        #             ari_count + processed
        #         )
        #         ari_count += processed

        # Other metrics
        count += 1
        ith_perf_vals = performance_tests(inp, out)
        if i == 0:
            performance_history = ith_perf_vals
        else:
            performance_history = {
                k: v + ith_perf_vals[k] for k, v in performance_history.items()
            }

    [
        print(f"Validation Average {k}, {v/count}")
        for k, v in performance_history.items()
    ]
    # print(f"Adjusted Rand Index over {ARI_BATCH_SIZE} samples: {ari_total}")
    print("--------------------------------------\n")


if __name__ == "__main__":
    main()
