from __future__ import annotations

import dataclasses


@dataclasses.dataclass(eq=True, frozen=True)
class MagnusGL6Method:
    dt: float

    def replace(self, **changes) -> MagnusGL6Method:
        return dataclasses.replace(self, **changes)
