"""Share the exact GPU implementation and launch choices with validation."""
from importlib import import_module

_CONFIGS = {
    'streamed': {},
    'padded': dict(check_warps=16, variable_warps=8, syndrome_warps=8,
                   variable_bs=32, variable_vs=16),
    'combined': dict(check_warps=16, variable_warps=8, syndrome_warps=8,
                     variable_bs=32, variable_vs=8),
}
_MODULES = dict(streamed='gpu_prepass', padded='gpu_padded', combined='gpu_combined')


def launch_config(implementation):
    return dict(_CONFIGS[implementation])


def make_first_leg(H, pri, *, implementation='streamed', **kwargs):
    decoder = import_module(_MODULES[implementation]).ExactFirstLeg
    return decoder(H, pri, **launch_config(implementation), **kwargs)
