import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from slot_attention_and_alignnet.src.plots import (
    plot_slot_attention,
    plot_slot_attention_horiz,
    plot_slot_attention_horiz_single,
)
from slot_attention_and_alignnet.src.utils import mse


class BGSlotAttentionAE(hk.Module):
    pretrain_partition_string = (
        None  # When loading, all params are used from checkpoint
    )
    latents_of_interest = ["slots"]

    def __init__(self, cfg, name="BGSlotAttentionAE"):
        super().__init__(name=name)
        self.cfg = cfg

    def __call__(self, x, debug=False):
        # x is (batch_size, H, W, C)
        return_dict = {}
        _, im_w, im_h = x.shape[:3]
        x = CNNEncoder(self.cfg)(x)
        bg_x, x = SlotAttentionModule(self.cfg)(
            x, self.cfg["attn_iter"], return_dict, debug=debug
        )
        x, alphas = CNNDecoder(self.cfg)(x, return_dict, debug=debug, combine=False)
        bg_x = BGDecoder()(
            bg_x.reshape((-1, self.cfg.slot_size)), is_training=not debug
        )

        x = x.reshape((-1, self.cfg.slots, im_w, im_h, 3))
        alphas = alphas.reshape((-1, self.cfg.slots, im_w, im_h, 1))
        bg_x = bg_x.reshape((-1, 1, im_w, im_h, 3))

        bg_mask = 1 - alphas.max(axis=1, keepdims=True)
        alphas = jax.nn.softmax(jnp.concatenate((bg_mask, alphas), axis=1), axis=1)
        x = jnp.concatenate((bg_x, x), axis=1) * alphas
        return_dict["slot_recos"] = x
        x = x.sum(axis=1)

        if debug:
            return_dict["masks"] = alphas

        return_dict["reco"] = x
        return return_dict

    @staticmethod
    def get_visualizers(_):
        def plot_fn(inp, out):
            batch_size, im_w, im_h = inp.shape[:3]
            figures = {}

            attn = out["attn"].reshape(batch_size, im_w, im_h, -1)
            attn = attn.transpose(0, 3, 1, 2)
            attn_fig, attn_fig_caption = plot_slot_attention_horiz(
                inp,
                attn,
                recos=out["reco"],
                masks=out["masks"],
                # slot_recos=out["slot_recos"],
                show_time_steps=False,
            )
            figures["Slot Attention"] = (attn_fig, attn_fig_caption)
            return figures

        return plot_fn

    @staticmethod
    def get_loss(_):
        def mse_loss(x, batch):
            """compute the loss of the network, including L2."""
            return {"total": jnp.mean((x["reco"] - batch) ** 2)}

        return mse_loss

    @staticmethod
    def get_optimizer(cfg):
        warm_up_poly = optax.polynomial_schedule(
            init_value=1 / cfg["warmup_iter"],
            end_value=1,
            power=1,
            transition_steps=cfg["warmup_iter"],
        )
        exp_decay = optax.exponential_decay(
            init_value=cfg["learning_rate"],
            transition_steps=cfg["decay_steps"],
            decay_rate=cfg["lr_decay_rate"],
            transition_begin=0,
        )  # config['warmup_iter'])
        opt = optax.chain(
            # clip_by_global_norm(max_norm),
            optax.scale_by_adam(
                b1=cfg["adam_beta_1"], b2=cfg["adam_beta_2"], eps=cfg["adam_eps"]
            ),
            optax.scale_by_schedule(warm_up_poly),
            optax.scale_by_schedule(exp_decay),
            optax.scale(-1),
        )
        return opt

    @staticmethod
    def get_performance_tests(cfg):
        def test_fn(batch_source, batch_out):
            metrics = {}
            metrics["mse"] = mse(batch_source, batch_out["reco"]).item()
            return metrics

        return test_fn


class BGDecoder(hk.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, is_training):
        use_bias = True
        init = None
        x = hk.Linear(8 * 8 * 8)(x)
        x = jax.nn.relu(x)
        x = hk.Reshape((8, 8, 8))(x)
        x = hk.Conv2DTranspose(
            16, 3, stride=2, padding="SAME", with_bias=use_bias, w_init=init
        )(x)
        x = jax.nn.relu(x)
        x = hk.Conv2DTranspose(
            32, 3, stride=2, padding="SAME", with_bias=use_bias, w_init=init
        )(x)
        x = jax.nn.relu(x)
        x = hk.Conv2DTranspose(
            64, 3, stride=2, padding="SAME", with_bias=use_bias, w_init=init
        )(x)
        x = jax.nn.relu(x)
        x = hk.Conv2DTranspose(
            3, 3, stride=2, padding="SAME", with_bias=True, w_init=init
        )(x)
        return x


class CNNDecoder(hk.Module):
    def __init__(self, C, name="SlotAttDecoder", out_channels=4):
        super().__init__(name=name)
        self.C = C
        self.num_slots = C["slots"]
        channels, kernels, strides = (
            C["decoder_cnn_channels"],
            C["decoder_cnn_kernels"],
            C["decoder_cnn_strides"],
        )
        glorot_uniform_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")

        deconv_layers = []
        if C["extra_deconv_layers"]:
            deconv_layers = [
                hk.Conv2DTranspose(channels[3], 5, 2, padding="SAME"),
                jax.nn.relu,
                hk.Conv2DTranspose(channels[4], 5, 2, padding="SAME"),
                jax.nn.relu,
            ]
        deconv_layers.extend(
            [
                hk.Conv2DTranspose(
                    channels[0],
                    kernels[0],
                    stride=strides[0],
                    padding="SAME",
                    w_init=glorot_uniform_init,
                ),
                jax.nn.relu,
                hk.Conv2DTranspose(
                    channels[1],
                    kernels[1],
                    stride=strides[1],
                    padding="SAME",
                    w_init=glorot_uniform_init,
                ),
                jax.nn.relu,
                hk.Conv2DTranspose(
                    channels[2],
                    kernels[2],
                    stride=strides[2],
                    padding="SAME",
                    w_init=glorot_uniform_init,
                ),
                jax.nn.relu,
                hk.Conv2DTranspose(
                    out_channels,
                    kernels[3],
                    padding="SAME",
                    stride=strides[3],
                    w_init=glorot_uniform_init,
                ),
            ]
        )

        self.deconvolutions = hk.Sequential(deconv_layers)

        self.pos_embed = SoftPositionEmbed(C["slot_size"], C["spatial_broadcast_dims"])

    def __call__(self, x, return_dict, num_obj_override=0, debug=False, combine=True):
        x = self.tile_grid(x)
        x = self.pos_embed(x)
        x = self.deconvolutions(x)
        x, alphas = self.decollapse_and_split(
            x, num_obj_override if num_obj_override else self.num_slots
        )

        if combine:  # Add flag so decoder can be used elsewhere manually
            if debug and "obj_decoded" in return_dict:
                return_dict["obj_decoded"] = x
            alphas = jax.nn.softmax(alphas, axis=1)  # Softmax across slots
            x = jnp.sum(x * alphas, axis=1, keepdims=False)  # Sum across slots

            if debug:
                return_dict["masks"] = alphas
            return x

        return x, alphas

    def decollapse_and_split(self, x, num_slots):
        # Decollapse batches and split alpha from color channels
        x = jnp.reshape(
            x, (x.shape[0] // num_slots, num_slots, *x.shape[1:])
        )  # Decollapse batches from slots
        x, alphas = jnp.array_split(x, [x.shape[-1] - 1], -1)
        # x, alphas = x[:,:,:,:,0], x[:,:,:,:,1] #jnp.split(x, 0, axis=-1)
        return x, alphas

    def tile_grid(self, x):
        # takes slots (batch, k, d) and returns (batch*k, w, h, d)
        # i.e. collapse batches (for computation/layer applicability?) and copy slot information wxh times, wtf?
        # maybe this general representational mapping format is sensible - grid cells and conceptual spaces eichenbaum hmmm
        x = jnp.reshape(x, (x.shape[0] * x.shape[1], 1, 1, x.shape[-1]))
        return jnp.tile(x, [1, *self.C["spatial_broadcast_dims"], 1])


class SlotAttentionModule(hk.Module):
    """Slot Attention Module - Iteratively perform dot product attention over inputs
    Inputs are (32*32, hidden_dim) and slots are (num_slots, slot_dim)
    """

    def __init__(self, C, name="SlotAttention"):
        super().__init__(name=name)
        self.w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
        self.num_slots = C["slots"]
        self.slot_size = C["slot_size"]
        self.attn_eps = C["attention_eps"]
        self.mlp_hidden_size = C["mlp_hidden_size"]

        # Layer norm for slots after and before attention + GRU (+MLP)
        self.layer_norm_1 = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True, eps=1e-03
        )
        self.layer_norm_2 = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True, eps=1e-03
        )
        self.layer_norm_bg_1 = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True, eps=1e-03
        )
        self.layer_norm_bg_2 = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True, eps=1e-03
        )
        # Slot update function is learned by GRU - #hidden_states = slot_dim
        self.mlp = hk.Sequential(
            [
                hk.Linear(self.mlp_hidden_size, w_init=self.w_init),
                jax.nn.relu,  # MLP + Residual Connection improves output
                hk.Linear(self.slot_size, w_init=self.w_init),  # esp. without relu...
            ]
        )
        self.bg_mlp = hk.Sequential(
            [
                hk.Linear(self.mlp_hidden_size, w_init=self.w_init),
                jax.nn.relu,  # MLP + Residual Connection improves output
                hk.Linear(self.slot_size, w_init=self.w_init),  # esp. without relu...
            ]
        )

    def weighted_mean(self, x, weights):
        weights = weights / jnp.sum(weights, axis=-2, keepdims=True)
        return jnp.einsum("bnk,bnd->bkd", weights, x)

    def __call__(self, x, T: int, return_dict, debug=False):
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, eps=1e-03)(x)

        mu = hk.get_parameter(
            "mu",
            [self.slot_size],
            init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
        )
        logstd = hk.get_parameter(
            "logstd",
            [self.slot_size],
            init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
        )
        mu_bg = hk.get_parameter(
            "mu_bg",
            [self.slot_size],
            init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
        )
        logstd_bg = hk.get_parameter(
            "logstd_bg",
            [self.slot_size],
            init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
        )

        # Initialize slots from common distribution - Affine transform of norm. dist
        slots = mu + jnp.exp(logstd) * jax.random.normal(
            hk.next_rng_key(),
            shape=(x.shape[0], self.num_slots, self.slot_size),
            dtype=jnp.float32,
        )
        bg_slot = mu_bg + jnp.exp(logstd_bg) * jax.random.normal(
            hk.next_rng_key(), shape=(x.shape[0], 1, self.slot_size), dtype=jnp.float32
        )

        if debug:
            return_dict["slot_mus"] = mu
            return_dict["slot_logstds"] = logstd

        k = hk.Linear(self.slot_size, w_init=self.w_init, with_bias=False, name="k")(x)
        v = hk.Linear(self.slot_size, w_init=self.w_init, with_bias=False)(x)
        attn = None
        updates = None

        update_fn = hk.GRU(self.slot_size, name="gru")
        q_mlp = hk.Linear(self.slot_size, w_init=self.w_init, with_bias=False)
        bg_update_fn = hk.GRU(self.slot_size, name="bg_gru")
        bg_q_mlp = hk.Linear(self.slot_size, w_init=self.w_init, with_bias=False)

        for _ in range(T):  # Iteratively applt slot attention
            slots_prev = slots
            slot_bg_prev = bg_slot
            slots = self.layer_norm_1(slots)
            bg_slot = self.layer_norm_bg_1(bg_slot)

            # Attention
            q = q_mlp(slots)
            bg_q = bg_q_mlp(bg_slot)
            q = jnp.concatenate((bg_q, q), axis=1) / jnp.sqrt(
                self.slot_size
            )  # bg slot competes with rest
            attn_logits = jnp.einsum(
                "bnd,bkd->bnk", k, q
            )  # Apply softmax attention and normalize over slots
            attn = jax.nn.softmax(attn_logits, axis=-1)

            # Weighted Mean
            updates = self.weighted_mean(v, attn[:, :, 1:] + self.attn_eps)
            bg_update = self.weighted_mean(
                v, jnp.expand_dims(attn[:, :, 0], -1) + self.attn_eps
            )

            # Slot Update
            slots, _ = update_fn(
                updates.reshape(-1, self.slot_size),
                slots_prev.reshape(-1, self.slot_size),
            )
            slots = slots.reshape(-1, self.num_slots, self.slot_size)
            slots += self.mlp(self.layer_norm_2(slots))
            bg_slot, _ = bg_update_fn(
                bg_update.reshape(-1, self.slot_size),
                slot_bg_prev.reshape(-1, self.slot_size),
            )
            bg_slot = bg_slot.reshape(-1, 1, self.slot_size)
            bg_slot += self.bg_mlp(self.layer_norm_bg_2(bg_slot))

        if debug:
            return_dict["slots"] = jnp.concatenate((bg_slot, slots), axis=1)  # slots
            return_dict["attn"] = attn

        return bg_slot, slots


class CNNEncoder(hk.Module):
    def __init__(self, C, name=None):
        super().__init__(name=name)
        glorot_uniform_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")

        channels = C["encoder_cnn_channels"]
        kernels = C["encoder_cnn_kernels"]
        strides = C["encoder_cnn_strides"]

        hidden_size = channels[-1]
        self.cnn_layers = hk.Sequential(
            [
                hk.Conv2D(
                    channels[0],
                    kernels[0],
                    stride=strides[0],
                    padding="SAME",
                    w_init=glorot_uniform_init,
                    with_bias=True,
                ),
                jax.nn.relu,
                hk.Conv2D(
                    channels[1],
                    kernels[1],
                    stride=strides[1],
                    padding="SAME",
                    w_init=glorot_uniform_init,
                    with_bias=True,
                ),
                jax.nn.relu,
                hk.Conv2D(
                    channels[2],
                    kernels[2],
                    stride=strides[2],
                    padding="SAME",
                    w_init=glorot_uniform_init,
                    with_bias=True,
                ),
                jax.nn.relu,
                hk.Conv2D(
                    hidden_size,
                    kernels[3],
                    stride=strides[3],
                    padding="SAME",
                    w_init=glorot_uniform_init,
                    with_bias=True,
                ),
            ]
        )

        self.pos_embed = SoftPositionEmbed(hidden_size, C["hidden_res"])

        self.linears = hk.Sequential(
            [  # i.e. 1x1 convolution (shared 32 neurons across all locations)
                hk.Reshape((-1, hidden_size)),  # Flatten spatial dim (works with batch)
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, eps=1e-03),
                hk.Linear(32, w_init=glorot_uniform_init),
                jax.nn.relu,
                hk.Linear(32, w_init=glorot_uniform_init),
            ]
        )

    def __call__(self, x):
        x = self.cnn_layers(x)
        x = self.pos_embed(x)
        x = self.linears(x)
        return x


# Modified from https://github.com/google-research/google-research/tree/master/slot_attention
class SoftPositionEmbed(hk.Module):
    """Adds soft positional embedding with learnable projection."""

    def __init__(self, hidden_size, resolution, name=None):
        """Builds the soft position embedding layer.
        args:
            hidden_size: Size of input feature dimension.
            resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__(name=name)
        self.grid = build_grid(resolution)

        glorot_uniform_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
        self.linear = hk.Linear(
            hidden_size, w_init=glorot_uniform_init, name="soft_pos_emb_linear"
        )

    def __call__(self, x):
        return x + self.linear(self.grid)


# Taken from https://github.com/google-research/google-research/tree/master/slot_attention
# Assigns 4vec to each point in (W,H) grid with fractional distance to borders: (Top, Right, Bottom, Left)
# @jax.partial(jax.jit, static_argnums=0)
def build_grid(resolution):
    ranges = [np.linspace(0.0, 1.0, num=res) for res in resolution]
    grid = np.meshgrid(
        *ranges, sparse=False, indexing="ij"
    )  # NOTE I changed the indexing from ij to align with cnn feature maps?
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(jnp.float32)
    return jnp.concatenate([grid, 1.0 - grid], axis=-1)
