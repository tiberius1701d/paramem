"""paramem.config — configuration classification, drift-detection, and the
knowledge-graph taxonomy (entity/relation types, anonymizer prefix vocabulary)
loaded from ``configs/schema.yaml``. A leaf package: no dependency on
``paramem.graph`` or ``paramem.server``, so every tier — including
``paramem.cloud`` — can import it directly.
"""
