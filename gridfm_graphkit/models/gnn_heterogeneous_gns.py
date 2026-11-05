import math

import torch
from torch import nn
from torch_geometric.nn import HeteroConv, TransformerConv

from gridfm_graphkit.models.coed_transformer_conv import CoEDTransformerConv
from gridfm_graphkit.io.registries import MODELS_REGISTRY
from gridfm_graphkit.io.param_handler import get_physics_decoder
from torch_scatter import scatter_add
from gridfm_graphkit.models.utils import (
    ComputeBranchFlow,
    ComputeNodeInjection,
    ComputeNodeResiduals,
    bound_with_sigmoid,
)
from gridfm_graphkit.datasets.globals import (
    # Bus feature indices
    VM_H,
    VA_H,
    MIN_VM_H,
    MAX_VM_H,
    # Output feature indices
    VM_OUT,
    PG_OUT_GEN,
    # Generator feature indices
    PG_H,
    MIN_PG,
    MAX_PG,
)


@MODELS_REGISTRY.register("GNS_heterogeneous")
class GNS_heterogeneous(nn.Module):
    """
    Heterogeneous version of your Transformer-based GNN for buses and generators.
    - Expects node features as dict: x_dict = {"bus": Tensor[num_bus, bus_feat], "gen": Tensor[num_gen, gen_feat]}
    - Expects edge_index_dict and edge_attr_dict with keys:
        ("bus","connects","bus"), ("gen","connected_to","bus"), ("bus","connected_to","gen")
      (edge_attr only needed for bus-bus currently; other relations can be None)
    - Keeps the physics residual idea but splits it into bus-step and gen-step residuals.
    """

    def __init__(self, args) -> None:
        super().__init__()
        self.num_layers = args.model.num_layers
        self.hidden_dim = args.model.hidden_size
        self.input_bus_dim = args.model.input_bus_dim
        self.input_gen_dim = args.model.input_gen_dim
        self.output_bus_dim = args.model.output_bus_dim
        self.output_gen_dim = args.model.output_gen_dim
        self.edge_dim = args.model.edge_dim
        self.heads = args.model.attention_head
        self.task = args.task.task_name
        self.dropout = getattr(args.model, "dropout", 0.0)

        # --- CoED (Continuous Edge Directions) --------------------------------
        # One angle per undirected branch, per layer.  Per-layer is deliberate,
        # not an option: the projected edge representation is built once outside
        # the layer loop, so a single shared theta would apply the same prior at
        # every layer and compound with depth.
        #
        # theta is parameterised as theta = (pi/2) * sigmoid(theta_raw) rather
        # than clamped.  This keeps it inside [0, pi/2] by construction, so no
        # post-step projection hook is needed and the module stays usable from
        # any training loop.  theta_raw = 0 gives exactly theta = pi/4, where
        # log(cos^2) == log(sin^2) is a constant on every edge and cancels in
        # the softmax: CoED is then provably a no-op, so epoch 0 matches the
        # baseline bit for bit.  Weight decay on theta_raw pulls towards 0, i.e.
        # towards that neutral angle, so it needs no special parameter group.
        self.coed = getattr(args.model, "coed", False)
        self.num_undirected_branches = getattr(
            args.model,
            "num_undirected_branches",
            None,
        )
        # How the two directed copies of a branch are ordered in edge_index:
        #   "contiguous"  -> [all forward, all reverse]   (one block-diagonal
        #                    mega-graph, e.g. the PPC parquet adapter)
        #   "interleaved" -> [fwd, rev] per graph, repeated (PyG's own
        #                    DataLoader collating one HeteroData per scenario)
        # There is no safe default: picking the wrong one binds theta to the
        # wrong branches *silently*, which reads as a null result rather than an
        # error, so it must be stated explicitly.
        self.coed_edge_order = getattr(args.model, "coed_edge_order", None)
        if self.coed:
            if not self.num_undirected_branches:
                raise ValueError(
                    "model.coed is enabled but model.num_undirected_branches is "
                    "unset. CoED needs one angle per undirected branch, so the "
                    "branch count of the single fixed topology must be known up "
                    "front.",
                )
            if self.coed_edge_order not in ("contiguous", "interleaved"):
                raise ValueError(
                    "model.coed requires model.coed_edge_order to be either "
                    "'contiguous' ([all forward, all reverse], as the PPC "
                    "parquet adapter builds it) or 'interleaved' ([fwd, rev] per "
                    "graph, as PyG's DataLoader builds it). Guessing wrong binds "
                    f"theta to the wrong branches without error; got "
                    f"{self.coed_edge_order!r}.",
                )
            self.theta_raw = nn.Parameter(
                torch.zeros(self.num_layers, self.num_undirected_branches),
            )
        else:
            self.theta_raw = None

        # projections for each node type
        self.input_proj_bus = nn.Sequential(
            nn.Linear(self.input_bus_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        self.input_proj_gen = nn.Sequential(
            nn.Linear(self.input_gen_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        self.input_proj_edge = nn.Sequential(
            nn.Linear(self.edge_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        # a small physics MLP that will take residuals (real, imag) and return a correction
        self.physics_mlp = nn.Sequential(
            nn.Linear(2, self.hidden_dim * self.heads),
            nn.LeakyReLU(),
        )

        # Build hetero layers: HeteroConv of TransformerConv per relation
        self.layers = nn.ModuleList()
        self.norms_bus = nn.ModuleList()
        self.norms_gen = nn.ModuleList()
        for i in range(self.num_layers):
            # in-channels depend on whether it is first layer (hidden_dim) or subsequent (hidden_dim * heads)
            in_bus = self.hidden_dim if i == 0 else self.hidden_dim * self.heads
            in_gen = self.hidden_dim if i == 0 else self.hidden_dim * self.heads
            out_dim = self.hidden_dim  # TransformerConv will output hidden_dim (per head reduction in HeteroConv call)

            # relation -> conv module mapping
            bus_bus_conv = CoEDTransformerConv if self.coed else TransformerConv
            conv_dict = {
                ("bus", "connects", "bus"): bus_bus_conv(
                    in_bus,
                    out_dim,
                    heads=self.heads,
                    edge_dim=self.hidden_dim,
                    dropout=self.dropout,
                    beta=True,
                ),
                ("gen", "connected_to", "bus"): TransformerConv(
                    in_gen,
                    out_dim,
                    heads=self.heads,
                    dropout=self.dropout,
                    beta=True,
                ),
                ("bus", "connected_to", "gen"): TransformerConv(
                    in_bus,
                    out_dim,
                    heads=self.heads,
                    dropout=self.dropout,
                    beta=True,
                ),
            }

            hetero_conv = HeteroConv(conv_dict, aggr="sum")
            self.layers.append(hetero_conv)

            # Norms for node representations (note: after HeteroConv each node type will have size out_dim * heads)
            self.norms_bus.append(nn.LayerNorm(out_dim * self.heads))
            self.norms_gen.append(nn.LayerNorm(out_dim * self.heads))

        # Separate shared MLPs to produce final bus/gen outputs (predictions y)
        self.mlp_bus = nn.Sequential(
            nn.Linear(self.hidden_dim * self.heads, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.output_bus_dim),
        )

        self.mlp_gen = nn.Sequential(
            nn.Linear(self.hidden_dim * self.heads, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.output_gen_dim),
        )

        # mask param (kept similar to your original)
        self.activation = nn.LeakyReLU()
        self.branch_flow_layer = ComputeBranchFlow()
        self.node_injection_layer = ComputeNodeInjection()
        self.node_residuals_layer = ComputeNodeResiduals()
        self.physics_decoder = get_physics_decoder(args)

        # In StateEstimation, the gen output head, physics_mlp, and the gen
        # branch of the final hetero conv layer never contribute to the loss.
        # Freeze those parameters so DDP doesn't reject them as unused. The
        # modules stay on the model so existing checkpoints still load.
        if self.task == "StateEstimation":
            for p in self.mlp_gen.parameters():
                p.requires_grad = False
            for p in self.physics_mlp.parameters():
                p.requires_grad = False
            last = self.num_layers - 1
            for p in self.norms_gen[last].parameters():
                p.requires_grad = False
            last_conv = self.layers[last].convs[("bus", "connected_to", "gen")]
            for p in last_conv.parameters():
                p.requires_grad = False

        # container for monitoring residual norms per layer and type
        self.layer_residuals = {}

    def _theta_logit(self, layer: int, num_directed_edges: int) -> torch.Tensor:
        """Per-directed-edge log directional weight for one layer.

        CoED gives each undirected branch an angle ``theta`` whose two directed
        copies carry complementary weights ``cos^2(theta)`` and
        ``sin^2(theta)``, summing to one.  This returns their logs, laid out to
        match the order the directed edges appear in ``edge_index``.

        The layout depends on who built the batch, hence
        ``model.coed_edge_order``:

        * ``contiguous`` -- the branch list of the whole batch is traversed once
          forward and once in reverse, giving ``[all forward, all reverse]``.
          This is what the PPC parquet adapter produces, because it assembles
          one block-diagonal mega-graph per batch.
        * ``interleaved`` -- each graph contributes its own
          ``[forward, reverse]`` block and the blocks are concatenated, giving
          ``[fwd_1, rev_1, fwd_2, rev_2, ...]``.  This is what PyG's own
          ``DataLoader`` produces from one ``HeteroData`` per scenario.

        Using the wrong one misplaces half the edges and raises nothing, so the
        divisibility check below is the only guard against a batch whose branch
        count does not match ``num_undirected_branches`` -- which also catches
        out-of-service branches (the adapter drops dead edges, so the count
        shrinks) and batches that mix topologies.
        """
        theta = (math.pi / 2.0) * torch.sigmoid(self.theta_raw[layer])
        cos2 = torch.cos(theta).pow(2).clamp(min=1e-8)
        sin2 = torch.sin(theta).pow(2).clamp(min=1e-8)
        log_cos2, log_sin2 = cos2.log(), sin2.log()

        n_branches = log_cos2.numel()
        if num_directed_edges % (2 * n_branches) != 0:
            raise ValueError(
                f"CoED needs one fixed, fully in-service topology per batch: got "
                f"{num_directed_edges} directed bus-bus edges, which is not a "
                f"multiple of 2 * num_undirected_branches = {2 * n_branches}. "
                f"Causes: a batch mixing networks with different branch counts, "
                f"out-of-service branches dropped by the edge builder, or a "
                f"mismatched model.num_undirected_branches.",
            )
        n_graphs = num_directed_edges // (2 * n_branches)

        if self.coed_edge_order == "contiguous":
            # [all forward, all reverse] over the concatenated branch list.
            return torch.cat([log_cos2.repeat(n_graphs), log_sin2.repeat(n_graphs)])
        # [fwd, rev] per graph, that block repeated once per graph.
        return torch.cat([log_cos2, log_sin2]).repeat(n_graphs)

    def forward(self, batch, return_embeddings: bool = False):
        """
        Accepts a PyG HeteroData batch and extracts the required tensors.

        batch: HeteroData/Batch containing:
            x_dict: {"bus": Tensor[num_bus, bus_feat], "gen": Tensor[num_gen, gen_feat]}
            edge_index_dict: keys like ("bus","connects","bus"), ("gen","connected_to","bus"), ("bus","connected_to","gen")
            edge_attr_dict: same keys -> edge attributes (bus-bus requires G,B)
            mask_dict: dict mapping node/bus types to mask tensors

        When ``return_embeddings`` is ``True``, returns ``(predictions, embeddings)``
        instead of ``predictions`` alone. Each embedding is the latent tensor fed
        directly into the corresponding prediction head: ``embeddings["bus"]`` is
        the ``h_bus`` consumed by ``mlp_bus`` and ``embeddings["gen"]`` is the
        ``h_gen`` consumed by ``mlp_gen`` on the final layer. (Note the last-layer
        ``physics_mlp`` update to ``h_bus`` is intentionally skipped, so the
        returned bus embedding is exactly the head input, not a post-update state.)
        """
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        edge_attr_dict = batch.edge_attr_dict
        mask_dict = batch.mask_dict

        self.layer_residuals = {}

        # 1) initial projections
        h_bus = self.input_proj_bus(x_dict["bus"])  # [num_bus, hidden_dim]
        h_gen = self.input_proj_gen(x_dict["gen"])  # [num_gen, hidden_dim]

        num_bus = x_dict["bus"].size(0)
        _, gen_to_bus_index = edge_index_dict[("gen", "connected_to", "bus")]
        bus_edge_index = edge_index_dict[("bus", "connects", "bus")]
        bus_edge_attr = edge_attr_dict[("bus", "connects", "bus")]

        edge_attr_proj_dict = {}
        for key, edge_attr in edge_attr_dict.items():
            if edge_attr is not None:
                edge_attr_proj_dict[key] = self.input_proj_edge(edge_attr)
            else:
                edge_attr_proj_dict[key] = None

        bus_mask = mask_dict["bus"][:, VM_H : VA_H + 1]
        gen_mask = mask_dict["gen"][:, : (PG_H + 1)]
        bus_fixed = x_dict["bus"][:, VM_H : VA_H + 1]
        gen_fixed = x_dict["gen"][:, : (PG_H + 1)]

        # iterate layers
        num_bus_edges = bus_edge_index.size(1)
        for i, conv in enumerate(self.layers):
            conv_kwargs = {}
            if self.coed:
                conv_kwargs["theta_logit_dict"] = {
                    ("bus", "connects", "bus"): self._theta_logit(
                        i,
                        num_bus_edges,
                    ),
                }
            out_dict = conv(
                {"bus": h_bus, "gen": h_gen},
                edge_index_dict,
                edge_attr_proj_dict,
                **conv_kwargs,
            )
            out_bus = out_dict["bus"]  # [Nb, hidden_dim * heads]
            out_gen = out_dict["gen"]  # [Ng, hidden_dim * heads]

            out_bus = self.activation(self.norms_bus[i](out_bus))
            out_gen = self.activation(self.norms_gen[i](out_gen))

            # skip connection
            h_bus = h_bus + out_bus if out_bus.shape == h_bus.shape else out_bus
            h_gen = h_gen + out_gen if out_gen.shape == h_gen.shape else out_gen

            # Decode bus and generator predictions
            bus_temp = self.mlp_bus(h_bus)  # [Nb, 2]  -> Vm, Va
            gen_temp = self.mlp_gen(h_gen)  # [Ng, 1]  -> Pg

            if self.task == "StateEstimation":
                if i == self.num_layers - 1:
                    Pft, Qft = self.branch_flow_layer(
                        bus_temp,
                        bus_edge_index,
                        bus_edge_attr,
                    )
                    P_in, Q_in = self.node_injection_layer(
                        Pft,
                        Qft,
                        bus_edge_index,
                        num_bus,
                    )
                    output_temp = self.physics_decoder(
                        P_in,
                        Q_in,
                        bus_temp,
                        x_dict["bus"],
                        None,
                        None,
                    )

            else:
                bus_temp = torch.where(bus_mask, bus_temp, bus_fixed)
                gen_temp = torch.where(gen_mask, gen_temp, gen_fixed)

                if self.task == "OptimalPowerFlow":
                    bus_temp[:, VM_OUT] = bound_with_sigmoid(
                        bus_temp[:, VM_OUT],
                        x_dict["bus"][:, MIN_VM_H],
                        x_dict["bus"][:, MAX_VM_H],
                    )
                    gen_temp[:, PG_OUT_GEN] = bound_with_sigmoid(
                        gen_temp[:, PG_OUT_GEN],
                        x_dict["gen"][:, MIN_PG],
                        x_dict["gen"][:, MAX_PG],
                    )

                Pft, Qft = self.branch_flow_layer(
                    bus_temp,
                    bus_edge_index,
                    bus_edge_attr,
                )
                P_in, Q_in = self.node_injection_layer(
                    Pft,
                    Qft,
                    bus_edge_index,
                    num_bus,
                )
                agg_bus = scatter_add(
                    gen_temp.squeeze(),
                    gen_to_bus_index,
                    dim=0,
                    dim_size=num_bus,
                )
                output_temp = self.physics_decoder(
                    P_in,
                    Q_in,
                    bus_temp,
                    x_dict["bus"],
                    agg_bus,
                    mask_dict,
                )
                residual_P, residual_Q = self.node_residuals_layer(
                    P_in,
                    Q_in,
                    output_temp,
                    x_dict["bus"],
                )

                bus_residuals = torch.stack([residual_P, residual_Q], dim=-1)

                # Save and project residuals to latent space.
                # layer_residuals is consumed by LayeredWeightedPhysicsLoss for
                # every layer, so it is recorded unconditionally.
                self.layer_residuals[i] = torch.linalg.norm(
                    bus_residuals,
                    dim=-1,
                ).mean()
                # On the last layer the updated h_bus is never read again: the
                # loop ends, predictions come from output_temp/gen_temp, and the
                # exported embedding is the tensor that fed mlp_bus. Skipping the
                # update avoids a dead forward pass and makes h_bus itself the
                # correct embedding to return (no snapshot/clone needed).
                if i < self.num_layers - 1:
                    h_bus = h_bus + self.physics_mlp(bus_residuals)

        predictions = {"bus": output_temp, "gen": gen_temp}
        if not return_embeddings:
            return predictions
        return predictions, {"bus": h_bus, "gen": h_gen}
