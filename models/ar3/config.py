"""AR3 configuration. Spec: temp/ar3.md."""

from __future__ import annotations

from typing import Any, Dict

from models.ar2_5.config import FL_AR25Config


class FL_AR3Config(FL_AR25Config):
    model_type = "fl_ar3"
    _YAML_REQUIRED = FL_AR25Config._YAML_REQUIRED | frozenset(
        {
            "align_topk",
            "align_power",
            "align_mass_power",
            "align_logit_coef",
            "align_loss_coef",
            "align_warmup_ratio",
            "align_query_prob",
        }
    )

    def __init__(
        self,
        name: str = "ar3",
        align_topk: int = 8,
        align_power: float = 2.0,
        align_mass_power: float = 1.0,
        align_logit_coef: float = 1.0,
        align_loss_coef: float = 0.1,
        align_warmup_ratio: float = 0.1,
        align_query_prob: float = 0.25,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.align_topk = int(align_topk)
        self.align_power = float(align_power)
        self.align_mass_power = float(align_mass_power)
        self.align_logit_coef = float(align_logit_coef)
        self.align_loss_coef = float(align_loss_coef)
        self.align_warmup_ratio = float(align_warmup_ratio)
        self.align_query_prob = float(align_query_prob)

    def backbone_kwargs(self) -> Dict[str, Any]:
        kw = super().backbone_kwargs()
        kw.update(
            {
                "align_topk": self.align_topk,
                "align_power": self.align_power,
                "align_mass_power": self.align_mass_power,
                "align_logit_coef": self.align_logit_coef,
                "align_loss_coef": self.align_loss_coef,
                "align_warmup_ratio": self.align_warmup_ratio,
                "align_query_prob": self.align_query_prob,
            }
        )
        return kw


CONFIG_CLS = FL_AR3Config
