from .dcn import (
    ModulatedDeformConv2dPackPlugin,
    ModulatedDeformConv2dPackPlugin2,
)

# Quantized versions are only available if pytorch_quantization is installed
try:
    from .dcn import (
        ModulatedDeformConv2dPackQ,
        ModulatedDeformConv2dPackPluginQ,
        ModulatedDeformConv2dPackPluginQ2,
    )
except ImportError:
    pass  # Quantized DCN variants not available
