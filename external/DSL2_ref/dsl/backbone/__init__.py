try:
    from .dit import DIT, DITY
    from .dits import DITS  # sahoo
except:  # import fails on non-cuda. Use dummy class and revert to BERT only
    class DIT:
        pass
    class DITS:
        pass
    class DITY:
        pass

try:
    from .modernbert import ModernBERT
except ImportError:
    ModernBERT = None
from .bert import BERT