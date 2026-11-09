import hashlib
import html
import re
from datetime import datetime

STOP = set("""a an the of in on for and or to with by from at as is are was were be been this that these those we our
their its it using use used based via into over under between among than then also can may not no which such has have
had their they more most other new paper study approach results show shown propose proposed method methods""".split())


def strip_tags(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def norm_doi(d):
    d = (d or "").lower().strip()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
    return d or None


def tokens(s):
    return [w for w in re.findall(r"[a-z0-9][a-z0-9\-]+", (s or "").lower()) if w not in STOP and len(w) > 2]


def pid_for(doi, title):
    return "p" + hashlib.sha1((norm_doi(doi) or norm_title(title)).encode()).hexdigest()[:10]


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text or "") if len(s.strip()) > 20]


def clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(x)))


def uniq(seq):
    seen, out = set(), []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def this_year():
    return datetime.now().year
