"""TransformerConv with a learnable per-edge directional bias (CoED).

Continuous Edge Directions (CoED, Pahng & Hormoz 2024) assign each *undirected*
edge a learnable angle ``theta`` in ``[0, pi/2]``.  It induces a complementary
pair of weights for the two directed copies of that edge,

    w_forward = cos^2(theta),    w_reverse = sin^2(theta),

which sum to one: ``theta = 0`` routes entirely source->target, ``pi/2``
entirely target->source, and ``pi/4`` is the symmetric (undirected) case.

The reference implementation turns those weights into a normalised graph
convolution.  Here they enter as an additive term on the attention logits
instead, so the direction competes inside the softmax alongside the learned
edge bias::

    alpha_ij = softmax_j( <q_i, k_j> / sqrt(d) + log w_ij )

``log w`` is supplied pre-computed as ``theta_logit`` so the caller controls how
theta maps onto edges (see ``GNS_heterogeneous._theta_logit``); this module only
adds it.

Two implementation notes, both load-bearing:

* The ``# propagate_type:`` comment in :meth:`forward` is **not** documentation.
  ``MessagePassing`` reads the class source to generate ``propagate``'s
  signature; without that annotation the inherited signature is used and
  ``propagate`` rejects ``theta_logit`` with a ``TypeError``.
* For the same reason this class must live in an importable ``.py`` file.
  Defined in a REPL or a heredoc, ``inspect.getsource`` fails and the
  annotation is never seen.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import TransformerConv
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor
from torch_geometric.utils import softmax
from typing import Optional, Tuple, Union


class CoEDTransformerConv(TransformerConv):
    """:class:`~torch_geometric.nn.TransformerConv` plus an additive logit bias.

    Identical to the parent when ``theta_logit`` is ``None`` -- bit-identical,
    not merely similar -- so a model built with this class but run without
    directions reproduces the plain ``TransformerConv`` baseline exactly.

    Args:
        theta_logit (Tensor, optional): one value per directed edge, added to
            the attention logits before the softmax.  Shape ``[num_edges]`` or
            ``[num_edges, 1]``; broadcast across heads.
    """

    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        theta_logit: OptTensor = None,
        return_attention_weights: Optional[bool] = None,
    ) -> Union[
        Tensor,
        Tuple[Tensor, Tuple[Tensor, Tensor]],
        Tuple[Tensor, SparseTensor],
    ]:
        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        # propagate_type: (query: Tensor, key: Tensor, value: Tensor, edge_attr: OptTensor, theta_logit: OptTensor)
        out = self.propagate(
            edge_index,
            query=query,
            key=key,
            value=value,
            edge_attr=edge_attr,
            theta_logit=theta_logit,
        )

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout="coo")
        else:
            return out

    def message(
        self,
        query_i: Tensor,
        key_j: Tensor,
        value_j: Tensor,
        edge_attr: OptTensor,
        theta_logit: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(
                -1,
                self.heads,
                self.out_channels,
            )
            key_j = key_j + edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)

        if theta_logit is not None:
            # One bias per directed edge, shared across heads.  Added before the
            # softmax so direction competes with the content and edge-bias terms
            # rather than rescaling the normalised weights afterwards.
            alpha = alpha + theta_logit.view(-1, 1)

        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out = out + edge_attr

        out = out * alpha.view(-1, self.heads, 1)
        return out
