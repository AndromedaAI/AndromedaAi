
Andromeda is een open-source serie LLMs gebouwd from scratch voor financiële toepassingen.

Wat is Andromeda?
- **v0.1**: Basis transformer blocks (attention, LayerNorm, FF) in pure NumPy/JAX.
- **Toekomst**: 1.5B MoE model getraind op tick-data + macro dat + persoonlijke trades.
- **Doel**: Een "Fin AGI" die hedgefondsen en traders helpt met edge.

Quick Start
bash
git clone https://github.com/AndromedaAI/andromeda-core.git
cd andromeda-core
python andromeda.py
