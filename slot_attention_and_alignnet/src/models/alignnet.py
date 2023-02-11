from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from slot_attention_and_alignnet.src.models.slot_attention import (
    CNNDecoder,
    CNNEncoder,
    SlotAttentionModule,
)
from slot_attention_and_alignnet.src.plots.vis_slot_attention import (
    plot_slot_attention,
    plot_slot_attention_horiz_single_align,
)
from slot_attention_and_alignnet.src.plots.vis_alignnet import plot_alignment

from slot_attention_and_alignnet.src.utils.performance_metrics import (
    alignment_score,
    mse,
    get_slot_assignments,
)


class AlignedSlotAttention(hk.Module):
    pretrain_partition_string = (
        "AlignNet"  # Used to merge pretrained backbone with learnable params
    )
    pretrain_param_matching_tuples = [
        ("SlotAttentionAE", "AlignedSlotAttention"),
        ("SlotAttDecoder", "cnn_decoder"),
        ("slot_attention_module", "SlotAttention"),
    ]
    latents_of_interest = ["slots"]

    def __init__(self, cfg, name="AlignedSlotAttention"):
        super().__init__(name=name)
        self.cfg = cfg

    def __call__(self, x, debug=False):
        if hk.running_init():  # Want to initialize all params and states during init
            debug = True
        # x is a (batch of) sequence of images [B, T, H, W, C]
        return_dict = {"reco": None}
        if x.ndim == 4:  # If we are given a single time series, we pad batch dim
            x = jnp.expand_dims(x, 0)
        batch_size = x.shape[0]
        x = CNNEncoder(self.cfg, name=None)(
            x.reshape(-1, *x.shape[-3:])
        )  # , is_training=not debug)
        x = SlotAttentionModule(self.cfg)(
            x, self.cfg.attn_iter, return_dict, debug=debug
        )
        x, permutations = AlignNet(self.cfg, training=not debug)(
            x, return_dict, debug=debug
        )  # .reshape(batch, seq_length, *x.shape[1:]))

        if debug:
            # ALIGN LATENTS FOR SLOT-OBJECT MATCHING
            permutations = permutations.reshape((-1, *permutations.shape[2:]))
            return_dict["attn"] = jnp.einsum(
                "bkK, bIK -> bIk", permutations, return_dict["attn"]
            )
            return_dict["slots"] = jnp.einsum(
                "bkK, bKd -> bKd", permutations, return_dict["slots"]
            )

            # DECODE
            decoder = CNNDecoder(self.cfg, name="cnn_decoder")
            x_dyn = decoder(x["x_dyn_hist"], return_dict, debug=debug)
            return_dict["dyn_masks"] = return_dict["masks"].reshape(
                batch_size, -1, *return_dict["masks"].shape[-4:]
            )

            x_perm = decoder(x["x_perm_hard"], return_dict, debug=debug)
            return_dict["perm_masks"] = return_dict["masks"].reshape(
                batch_size, -1, *return_dict["masks"].shape[-4:]
            )
            return_dict["slotsss"] = return_dict["slot_recos"].reshape(
                batch_size, -1, *return_dict["slot_recos"].shape[-4:]
            )
            del return_dict["masks"]

            x_dyn, x_perm = map(
                lambda y: y.reshape(batch_size, -1, *y.shape[-3:]), (x_dyn, x_perm)
            )
            return_dict["x_dyn"] = x_dyn
            return_dict["x_perm"] = x_perm

            return_dict["rollout"] = decoder(
                x["rollout"][:, 0], return_dict, debug=debug
            )

        return return_dict

    @staticmethod
    def get_loss(_):
        def mse_loss(x, _):
            """compute the loss of the network, including L2."""
            loss = {}
            loss["dynamics"] = mse(
                x["x_dyn_hist"], jax.lax.stop_gradient(x["x_perm_hard"])
            )
            loss["permutations"] = mse(
                jax.lax.stop_gradient(x["x_dyn_hist"]), x["x_perm_soft"]
            )  # Stop grad towards the dynamics model?
            loss["permutation_entropy"] = -0.01 * jnp.mean(
                jnp.sum(x["S_hist"] * jnp.log(x["S_hist"]), axis=(1, 2))
            )

            loss["total"] = (
                loss["dynamics"] + loss["permutations"] + loss["permutation_entropy"]
            )
            return loss

        return mse_loss

    @staticmethod
    def get_optimizer(cfg):
        return optax.rmsprop(cfg["alignnet_lr"])

    @staticmethod
    def get_performance_tests(cfg):
        def test_fn(batch_source, batch_out, **kwargs):
            metrics = {}
            metrics["mse_on_permutations"] = mse(
                batch_source[:, 1:], batch_out["x_perm"]
            ).item()
            metrics["alignment"] = alignment_score(batch_out["perm_masks"])
            return metrics

        return test_fn

    @staticmethod
    def get_visualizers(_):
        def plot_fn(inp, out, **kwargs):
            if inp.ndim < 5:
                inp = jnp.expand_dims(inp, 0)
            T, B, im_w, im_h = inp.shape[:4]
            figures = {}

            # Plot masks
            attn = out["attn"].reshape(T, B, im_w, im_h, -1)[0]
            attn = attn.transpose(0, 3, 1, 2)
            attn_fig, attn_fig_caption = plot_slot_attention_horiz_single_align(
                inp[0],
                attn,
                recos=[out["x_dyn"][0], out["x_perm"][0]],
                masks=[out["dyn_masks"][0], out["perm_masks"][0]],
                slot_recos=out["slotsss"][0],
                show_time_steps=True,
            )
            figures["Slot Attention"] = (attn_fig, attn_fig_caption)

            # Plot alignment
            # slot_object_assignments = get_slot_assignments(
            #     jnp.expand_dims(out["perm_masks"][0], 0)
            # )[0]
            # plot_colors = np.array([[50.0, 170.0, 255.0], [0.0, 140.0, 255.0]])
            # plot_shapes = np.array(["D", "s"])
            # align_fig, align_caption = plot_alignment(
            #     slot_object_assignments, [plot_colors, plot_shapes]
            # )
            # figures["Alignment"] = (align_fig, align_caption)

            return figures

        return plot_fn


class AlignNet(hk.Module):
    def __init__(self, cfg, training=True, name="AlignNet"):
        super().__init__(name=name)
        self.cfg = cfg
        self.permuter = SinkhornTransformer(cfg)  # Transformer + SinkHorn
        self.training = training

    def __call__(self, x: jnp.ndarray, return_dict, debug=False) -> jnp.ndarray:
        """Apply AligNet module to align input sequence of embedded entities

        Args:
            x (jnp.ndarray): [Batch, Sequence, Slots, SlotDims]
                             Slot-wise object embeddings over batched sequences

        Returns:
            jnp.ndarray: [Batch, Sequence, Slots, SlotDims]
                         Aligned Slot-wise sequences
        """
        x = x.reshape(
            (-1, self.cfg.sequence_length, self.cfg.slots, self.cfg.slot_size)
        )
        x = x.transpose((1, 0, 2, 3))  # Now : [Time, Batch, Slots, SlotDim]

        RNN = hk.LSTM(self.cfg.slot_size, "dynamics_lstm")
        c = RNN.initial_state(
            x.shape[1] * self.cfg.slots
        )  # Dynamics independent acros slots
        x_dyn = x[0]
        x_dyn_hist = (
            [jnp.array(x[0])] if debug else []
        )  # embeddings if return_decoded, else loss
        x_perm_hist = []
        x_perm_hist_soft = []
        S_hist = []
        S_hist_hard = []
        T, B, K = x.shape[:3]
        for t in range(T - 1):
            # Additive Dynamics
            dynamics, c = RNN(x_dyn.reshape((-1, self.cfg.slot_size)), c)
            x_dyn = x_dyn + dynamics.reshape((-1, self.cfg.slots, self.cfg.slot_size))

            # Permutation
            S = self.permuter(x_dyn, x[t + 1])
            S_hist.append(S)

            # For inference AND training dynamics, we use hard aligned dynamics
            hard_S = jax.nn.one_hot(jnp.argmax(S, axis=-1), self.cfg.slots)
            x_perm_hard = jnp.einsum("bkK, bKd -> bkd", hard_S, x[t + 1])

            if debug:
                S_hist_hard.append(hard_S)
                if t == 2:  # Perform one 10 step rollout
                    rollout = [x[i] for i in range(3)] + [x_dyn]
                    c_temp = c
                    for _ in range(10):
                        dynamics_temp, c_temp = RNN(
                            rollout[-1].reshape((-1, self.cfg.slot_size)), c_temp
                        )
                        rollout.append(
                            rollout[-1]
                            + dynamics_temp.reshape(
                                (-1, self.cfg.slots, self.cfg.slot_size)
                            )
                        )
                    return_dict["rollout"] = jnp.stack(rollout)

            if not debug:
                # For training transformer we'll use soft permutation
                x_perm_soft = jnp.einsum("bkK, bKd -> bkd", S, x[t + 1])
                x_perm_hist_soft.append(x_perm_soft)
            x_dyn_hist.append(jnp.array(x_dyn))
            x_perm_hist.append(jnp.array(x_perm_hard))

            x_dyn = x_perm_hard
        if not debug:  # Previously, not return decoded
            return_dict["S_hist"] = jnp.stack(S_hist)
            return_dict["x_perm_soft"] = (
                jnp.stack(x_perm_hist_soft)
                .transpose((1, 0, 2, 3))
                .reshape((-1, self.cfg.slots, self.cfg.slot_size))
            )

        return_dict["x_dyn_hist"] = (
            jnp.stack(x_dyn_hist)
            .transpose((1, 0, 2, 3))
            .reshape((-1, self.cfg.slots, self.cfg.slot_size))
        )
        return_dict["x_perm_hard"] = (
            jnp.stack(x_perm_hist)
            .transpose((1, 0, 2, 3))
            .reshape((-1, self.cfg.slots, self.cfg.slot_size))
        )

        return (
            return_dict,
            jnp.concatenate(
                (jnp.identity(K).tile([1, B, 1, 1]), jnp.stack(S_hist_hard)), axis=0
            )
            if debug
            else None,
        )

    def get_loss(self, x_dyn, x_perm_soft, x_perm_hard, S):
        dyn_mse_loss = jnp.mean((x_dyn - x_perm_soft) ** 2)
        perm_mse_loss = jnp.mean((x_dyn - x_perm_hard) ** 2)
        perm_entropy_loss = -jnp.mean(jnp.sum(S * jnp.log(S), axis=(1, 2)))
        return dyn_mse_loss, perm_mse_loss, perm_entropy_loss


class SinkhornTransformer(hk.Module):
    # Generates soft permutation matrix which is used to align current slots with prev_slots+dynamics
    def __init__(self, cfg, name=None):
        super().__init__(name=name)
        self.cfg = cfg

    def __call__(self, x_dyn, x_obs):
        batch_size = x_dyn.shape[0]

        # Add set and positional information
        pos_information = jnp.expand_dims(
            jnp.tile(jnp.linspace(-1, 1, num=self.cfg.slots), (batch_size, 1)), -1
        )
        x_dyn = jnp.concatenate(
            [x_dyn, -jnp.ones((*x_dyn.shape[:-1], 1)), pos_information], axis=-1
        )
        x_obs = jnp.concatenate(
            [x_obs, jnp.ones((*x_obs.shape[:-1], 1)), pos_information], axis=-1
        )

        # Apply transformer layers and get similarity matrix from final one
        P = Transformer(self.cfg)(x_dyn, x_obs)

        # Apply sinkhorn operator iteratively
        B, N = P.shape[
            :2
        ]  # NOTE Assuming transformer saliency matrix is square (not needed for sinkhorn)
        S = jnp.exp(P / self.cfg.sinkhorn_temp)
        for _ in range(self.cfg.sinkhorn_iter):
            # Normalize rows and then columns
            S = S / (jnp.einsum("bnN,bNm->bnm", S, jnp.ones((B, N, N))))
            S = S / (jnp.einsum("bnN,bNm->bnm", jnp.ones((B, N, N)), S))
        return S


class Transformer(hk.Module):
    def __init__(self, C, name=None):
        super().__init__(name=name)
        self._num_layers = C["transformer_layers"]
        self._dropout_rate = C["transformer_dropout"]
        self._num_heads = C["transformer_multiattn_heads"]

    def __call__(self, x_dyn, x_obs, is_training: bool = True) -> jnp.ndarray:
        init_scale = 2.0 / self._num_layers
        dropout_rate = self._dropout_rate if is_training else 0.0
        key_size = x_dyn.shape[-1]
        h_attn = None
        for i in range(self._num_layers):
            h_dyn_norm = layer_norm(x_dyn, name=f"h{i}_ln_1_dyn")
            h_obs_norm = layer_norm(x_obs, name=f"h{i}_ln_1_obs")
            # Note -> We only want the similarity matrix from the last layer
            h_attn = MultiHeadAttention(
                num_heads=self._num_heads,
                key_size=key_size // self._num_heads,
                w_init_scale=init_scale,
                name=f"h{i}_attn",
            )(
                h_dyn_norm,
                h_obs_norm,
                h_obs_norm,
                return_similarity=(i == self._num_layers - 1),
            )
            if i < self._num_layers - 1:
                h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)
                x_obs = x_obs + h_attn
                h_obs_norm = layer_norm(x_obs, name=f"h{i}_ln_2")
                h_dense = DenseBlock(init_scale, name=f"h{i}_mlp")(h_obs_norm)
                h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense)
                h_dense_dyn = DenseBlock(init_scale, name=f"h{i}_mlp")(h_dyn_norm)
                h_dense_dyn = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense_dyn)
                x_obs = x_obs + h_dense
                x_dyn = x_dyn + h_dense_dyn
        return h_attn


def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
    """Apply a unique LayerNorm to x with default settings."""
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=name)(x)


class DenseBlock(hk.Module):
    """A 2-layer MLP which widens then narrows the input."""

    def __init__(
        self, init_scale: float, widening_factor: int = 4, name: Optional[str] = None
    ):
        super().__init__(name=name)
        self._init_scale = init_scale
        self._widening_factor = widening_factor

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        hiddens = x.shape[-1]
        initializer = hk.initializers.VarianceScaling(self._init_scale)
        x = hk.Linear(self._widening_factor * hiddens, w_init=initializer)(x)
        x = jax.nn.gelu(x)
        return hk.Linear(hiddens, w_init=initializer)(x)


# Taken from haiku core
class MultiHeadAttention(hk.Module):
    """Multi-headed attention mechanism.

    As described in the vanilla Transformer paper:
      "Attention is all you need" https://arxiv.org/abs/1706.03762
    """

    def __init__(
        self,
        num_heads: int,
        key_size: int,
        w_init_scale: float,
        query_size: Optional[int] = None,
        value_size: Optional[int] = None,
        model_size: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.query_size = query_size or key_size
        self.value_size = value_size or key_size
        self.model_size = model_size or key_size * num_heads
        self.w_init = hk.initializers.VarianceScaling(w_init_scale)

    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        return_similarity: Optional[bool] = False,
    ) -> jnp.ndarray:
        """Compute (optionally masked) MHA with queries, keys & values."""
        query_heads = self._linear_projection(query, self.query_size, "query")
        key_heads = self._linear_projection(key, self.key_size, "key")
        value_heads = self._linear_projection(value, self.value_size, "value")

        attention_logits = jnp.einsum("bthd,bThd->bhtT", query_heads, key_heads)
        if mask is not None:
            attention_logits -= 1e10 * (1.0 - mask)

        attention_weights = jax.nn.softmax(attention_logits)
        sqrt_key_size = jnp.sqrt(self.key_size)  # They used np , dtype=key.dtype)
        attention_weights = attention_weights / sqrt_key_size

        if return_similarity:
            return attention_weights.sum(
                axis=1, keepdims=False
            )  

        attention = jnp.einsum("bhtT,bThd->bthd", attention_weights, value_heads)
        # Concatenate attention matrix of all heads into a single vector.
        attention_vec = jnp.reshape(attention, (*query.shape[:2], -1))

        output = hk.Linear(self.model_size, w_init=self.w_init)(attention_vec)
        return output

    @hk.transparent
    def _linear_projection(
        self, x: jnp.ndarray, head_size: int, name: Optional[str] = None
    ) -> jnp.ndarray:
        y = hk.Linear(self.num_heads * head_size, w_init=self.w_init, name=name)(x)
        return y.reshape((*x.shape[:2], self.num_heads, head_size))
