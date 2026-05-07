# Teacher committee package — dev/refresh only, not imported at inference time.
# All public symbols are imported lazily so that production code (which does
# not install the teacher group) never triggers an ImportError.
