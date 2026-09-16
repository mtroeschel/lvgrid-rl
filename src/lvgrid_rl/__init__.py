"""lvgrid-rl: training environment for RL agents controlling low-voltage grids.

Layering (see architecture document, section 2):

==============  =========================================================
Package         Layer
==============  =========================================================
``core``        Schemas, protocols, unit convention, information ordering
``data``        L0 data layer: sources, resampling, cache, scenarios
``grid``        L1 grid and physics
``components``  L2 asset and flexibility models
``env``         L3 Gymnasium environment
``agents``      L4 agents
``baselines``   L4 reference methods
``eval``        L5 KPIs and evaluation
``viz``         L5 visualisation
``experiment``  L6 orchestration and reproducibility
==============  =========================================================
"""

__version__ = "0.1.0.dev0"
