"""Constants used by Vitessce config builders.

This module intentionally contains both shared constants and domain-specific
defaults. Domain-specific defaults are safe to change when adapting configs,
as long as usages stay internally consistent.
"""

# Vitessce component identifiers used by this package.
SPATIAL_VIEW = "spatialBeta"
LAYER_CONTROLLER_VIEW = "layerControllerBeta"

# Shared image-layer constraints.
MAX_INITIAL_CHANNELS = 6  # Viv currently supports at most 6 visible channels.

# Seq-based transcriptomics (visium_hd) coordination identifiers.
# These are scoped to seq-based configs and are not required by proteomics.
OBS_TYPE_SPOT = "spot"
OBS_TYPE_BIN = "bin"
OBS_COLOR_CELL_SET_SELECTION = "cellSetSelection"
OBS_COLOR_GENE_SELECTION = "geneSelection"

# Seq-based transcriptomics (visium_hd) coordination defaults.
# These are user-facing defaults, not protocol-level requirements; changing
# them is supported and should not break code when used consistently.
FEATURE_TYPE_GENE = "gene"
FEATURE_VALUE_TYPE_EXPRESSION = "expression"
FEATURE_TYPE_QC = "qc"
FEATURE_VALUE_TYPE_QC = "qc_value"
