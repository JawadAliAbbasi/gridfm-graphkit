"""Adapters that feed the gridfm-graphkit models from external data pipelines.

``ppc_parquet_coed`` drives :class:`GNS_heterogeneous` from the PPC branch-row
parquet corpus, with CoED (learnable continuous edge directions) available.
It imports a few helpers from the PIGNN-Attn-LS-PPC driver, so that directory
must be importable -- which it is when running its ``train_valid_test_*.py``
scripts, since they live there.
"""
