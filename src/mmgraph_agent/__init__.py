"""mmgraph-agent: an independent reproduction of MMGraph-Agent.

The package is organised around the three modules described in the paper:

* :mod:`mmgraph_agent.hypergraph` -- zero-API multimodal hypergraph construction
* :mod:`mmgraph_agent.memory`     -- cache-inspired L1/L2 memory
* :mod:`mmgraph_agent.policy`     -- unified action space + GRPO reward

The deterministic, CPU-only parts run with the standard library plus NumPy;
the heavy CLIP / VLM / GRPO pieces are behind optional adapters.
"""

__version__ = "0.1.0"
