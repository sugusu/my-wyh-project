# S0 Label Definition

Status: protocol recorded, labels not generated.

Candidate labels must be computed by exact triangle nearest-point distance to verified GT surfaces. Metric scenes use TRUE at distance <= 0.002 m and FALSE at distance >= 0.010 m. Non-metric or very small scenes use TRUE at <= 0.002 times scene bounding-box diagonal and FALSE at >= 0.010 times scene bounding-box diagonal. Intermediate candidates are IGNORE.

This run did not generate labels because official source, selected scenes, candidate exports, and verified GT geometry were unavailable.
