# SPDX-License-Identifier: Apache-2.0
"""No-op stub."""


class MedusaProposer:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Medusa proposer is not supported in the lean build")
