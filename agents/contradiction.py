"""Contradiction detection between two finding statements (NLI cross-encoder if enabled, else polarity heuristic)."""
import logging
import re

from settings import settings

log = logging.getLogger(__name__)
POS = r"improv|outperform|increase|better|effective|benefit|gain|higher|superior|enhanc|succe"
NEG = r"fail|no significant|not significant|worse|degrad|decreas|ineffective|no improvement|does not|did not|do not|limited gain|underperform|lower|harm|hurt"


class ContradictionDetector:
    def __init__(self):
        self._nli, self._tried = None, False

    def _load(self):
        if not self._tried:
            self._tried = True
            if settings.USE_NLI:
                try:
                    from sentence_transformers import CrossEncoder
                    self._nli = CrossEncoder(settings.NLI_MODEL)
                except Exception as e:
                    log.warning("NLI model unavailable: %s", e)
        return self._nli

    @staticmethod
    def polarity(text):
        t = (text or "").lower()
        pos, neg = len(re.findall(POS, t)), len(re.findall(NEG, t))
        return 0 if pos == neg else (1 if pos > neg else -1)

    def score(self, a, b):
        """Probability-like score in [0,1] that claims a and b contradict."""
        if not a or not b:
            return 0.0
        nli = self._load()
        if nli:
            import numpy as np
            logits = np.array(nli.predict([(a, b)])[0])
            p = np.exp(logits - logits.max())
            p /= p.sum()
            return float(p[0])  # label order: contradiction, entailment, neutral
        pa, pb = self.polarity(a), self.polarity(b)
        return 0.7 if pa and pb and pa != pb else 0.0
