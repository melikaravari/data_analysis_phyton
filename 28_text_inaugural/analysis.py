"""
Project 28 - Text as data: how has the language of US presidential inaugural addresses changed, 1789-2025?

Data: the Inaugural Address Corpus in NLTK (60 speeches from Washington 1789 to 2025; compiled by C-Span; public domain government speeches).
Loaded with nltk.download("inaugural"); the text is not stored in this repository.

Questions
  1. How do length, sentence length, readability and lexical diversity change over time (regression with robust standard errors)?
  2. Which words distinguish the eras (weighted log-odds)?
  3. Which topics (non-negative matrix factorisation of text chunks) appear, and how does their share change over time?
  4. Can the year of a speech be predicted from its words (ridge regression, leave-one-out cross-validation), and which words carry the information?

Run:  python analysis.py      (needs numpy pandas scipy statsmodels scikit-learn nltk matplotlib)
Outputs: results.txt and figures/*.png in this folder.
"""
import re
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)
out = open(HERE / "results.txt", "w")


def log(*a):
    print(*a)
    print(*a, file=out)


nltk.download("inaugural", quiet=True)
from nltk.corpus import inaugural  # noqa: E402

rows = []
for fid in inaugural.fileids():
    year, name = fid[:-4].split("-")
    rows.append({"year": int(year), "president": name, "text": inaugural.raw(fid)})
df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
log(f"{len(df)} speeches, {df.year.min()}-{df.year.max()}, {len(df.president.unique())} different presidents; {df.text.str.split().str.len().sum():,} words in total\n")

# ----------------------------------------------------------------------------
# 1. Simple text statistics over time
# ----------------------------------------------------------------------------
def words(t):
    return re.findall(r"[A-Za-z']+", t.lower())


def sentences(t):
    return [s for s in re.split(r"(?<=[.!?])\s+", t.replace("\n", " ")) if len(s.split()) > 2]


def syllables(w):  # rough heuristic: count vowel groups
    w = re.sub(r"e$", "", w)
    return max(1, len(re.findall(r"[aeiouy]+", w)))


def mattr(tokens, window=500):  # moving-average type-token ratio: lexical diversity independent of text length
    if len(tokens) < window:
        return np.nan
    return np.mean([len(set(tokens[i:i + window])) / window for i in range(0, len(tokens) - window + 1, 50)])


stat = []
for t in df.text:
    w, s = words(t), sentences(t)
    syl = sum(syllables(x) for x in w)
    stat.append({"n_words": len(w), "sent_len": len(w) / len(s), "fk_grade": 0.39 * len(w) / len(s) + 11.8 * syl / len(w) - 15.59, "mattr": mattr(w)})
df = pd.concat([df, pd.DataFrame(stat)], axis=1)
df["decade10"] = (df.year - 1900) / 10
log("Descriptives: median speech length {:.0f} words (shortest {} in {}, longest {} in {}); median sentence length {:.0f} words".format(
    df.n_words.median(), df.n_words.min(), df.loc[df.n_words.idxmin(), "year"], df.n_words.max(), df.loc[df.n_words.idxmax(), "year"], df.sent_len.median()))

rows = []
for v, lab in [("n_words", "words per speech"), ("sent_len", "words per sentence"), ("fk_grade", "Flesch-Kincaid grade (approx.)"), ("mattr", "lexical diversity (MATTR-500)")]:
    m = smf.ols(f"{v} ~ decade10", df.dropna(subset=[v])).fit(cov_type="HC3")
    rows.append({"measure": lab, "change per decade": m.params["decade10"], "95% CI": f"[{m.conf_int().loc['decade10', 0]:.2f}, {m.conf_int().loc['decade10', 1]:.2f}]",
                 "p": m.pvalues["decade10"], "R2": m.rsquared, "mean in 1789-1850": df.loc[df.year <= 1850, v].mean(), "mean since 1950": df.loc[df.year >= 1950, v].mean()})
tr = pd.DataFrame(rows).round(3)
log("\nLinear time trends (robust HC3 standard errors)")
log(tr.to_string(index=False))
e = tr.set_index("measure")
log(f"Sentences became much shorter: about {e.loc['words per sentence', 'mean in 1789-1850']:.0f} words on average in 1789-1850 versus {e.loc['words per sentence', 'mean since 1950']:.0f} since 1950 "
    f"({e.loc['words per sentence', 'change per decade']:.1f} words per decade in a linear fit), and the approximate reading grade fell from {e.loc['Flesch-Kincaid grade (approx.)', 'mean in 1789-1850']:.0f} to "
    f"{e.loc['Flesch-Kincaid grade (approx.)', 'mean since 1950']:.0f}. Speech length shows no significant linear trend (p = {e.loc['words per speech', 'p']:.2f}; a few very long early speeches), "
    f"and lexical diversity is nearly flat ({e.loc['lexical diversity (MATTR-500)', 'mean in 1789-1850']:.3f} to {e.loc['lexical diversity (MATTR-500)', 'mean since 1950']:.3f}). "
    "Linear fits are rough descriptions, not models of the process.\n")
log("Mean sentence length by era (words): " + ", ".join(f"{k} {v:.0f}" for k, v in df.groupby(pd.cut(df.year, [1788, 1850, 1900, 1950, 2000, 2030], labels=['1789-1850', '1853-1900', '1901-1950', '1953-2000', '2001-2025'])).sent_len.mean().items()) + "\n")

# ----------------------------------------------------------------------------
# 2. Distinctive words by era: weighted log-odds with an informative Dirichlet prior (Monroe, Colaresi & Quinn 2008)
# ----------------------------------------------------------------------------
df["era"] = pd.cut(df.year, [1788, 1850, 1900, 1950, 2000, 2030], labels=["1789-1850", "1853-1900", "1901-1950", "1953-2000", "2001-2025"])
cv = CountVectorizer(stop_words="english", token_pattern=r"[A-Za-z]{3,}", min_df=3)
C = cv.fit_transform(df.text)
vocab = np.array(cv.get_feature_names_out())
prior = np.asarray(C.sum(0)).ravel() + 0.01
a0 = prior.sum()
log("Most distinctive words by era (z-score of the weighted log-odds ratio against all other eras; |z| > 1.96 is conventionally large)")
for era in df.era.cat.categories:
    mask = (df.era == era).values
    y1, y2 = np.asarray(C[mask].sum(0)).ravel(), np.asarray(C[~mask].sum(0)).ravel()
    n1, n2 = y1.sum(), y2.sum()
    d_ = np.log((y1 + prior) / (n1 + a0 - y1 - prior)) - np.log((y2 + prior) / (n2 + a0 - y2 - prior))
    z = d_ / np.sqrt(1 / (y1 + prior) + 1 / (y2 + prior))
    top = np.argsort(-z)[:8]
    log(f"  {era} ({int(mask.sum())} speeches): " + ", ".join(f"{vocab[i]} ({z[i]:.1f})" for i in top))
log("")

# ----------------------------------------------------------------------------
# 3. Topics: NMF on chunks of 8 consecutive sentences (60 whole speeches would be too few documents)
# ----------------------------------------------------------------------------
chunks = []
for _, r in df.iterrows():
    s = sentences(r.text)
    for i in range(0, len(s), 8):
        part = " ".join(s[i:i + 8])
        if len(part.split()) >= 40:
            chunks.append({"year": r.year, "text": part})
ch = pd.DataFrame(chunks)
log(f"{len(ch)} text chunks of about 8 sentences")
tf = TfidfVectorizer(stop_words="english", token_pattern=r"[A-Za-z]{3,}", min_df=8, max_df=0.4)
T = tf.fit_transform(ch.text)
terms = np.array(tf.get_feature_names_out())
B = (CountVectorizer(vocabulary=terms, binary=True, token_pattern=r"[A-Za-z]{3,}").fit_transform(ch.text)).toarray()


def umass(top_idx):
    s_ = 0
    for i in range(1, len(top_idx)):
        for j in range(i):
            co = (B[:, top_idx[i]] * B[:, top_idx[j]]).sum()
            s_ += np.log((co + 1) / B[:, top_idx[j]].sum())
    return s_ / (len(top_idx) * (len(top_idx) - 1) / 2)


cohs = {}
models = {}
for k in (4, 6, 8, 10, 12):
    m = NMF(k, init="nndsvda", random_state=0, max_iter=500).fit(T)
    models[k] = m
    cohs[k] = np.mean([umass(np.argsort(-c)[:10]) for c in m.components_])
log("Topic coherence (UMass, higher/less negative is better) by number of topics: " + ", ".join(f"k = {k}: {v:.2f}" for k, v in cohs.items()))
K = 8  # chosen for interpretability; coherence differences between k are small
nmf = models[K]
W = nmf.transform(T)
W = W / W.sum(1, keepdims=True).clip(1e-12)
labels = {}
log(f"Using k = {K} topics. Top words and average share per era:")
ch["era"] = pd.cut(ch.year, [1788, 1850, 1900, 1950, 2000, 2030], labels=df.era.cat.categories)
share = pd.DataFrame(W, columns=[f"T{i + 1}" for i in range(K)]).groupby(ch.era.values).mean() * 100
for i, c in enumerate(nmf.components_):
    top = ", ".join(terms[np.argsort(-c)[:8]])
    log(f"  T{i + 1}: {top}  | share by era (%): " + ", ".join(f"{share.iloc[j, i]:.0f}" for j in range(len(share))))
log("(Topic names are my interpretation of the word lists; shares are rows of eras 1789-1850, 1853-1900, 1901-1950, 1953-2000, 2001-2025.)\n")

fig, ax = plt.subplots(figsize=(8, 4))
share.T.plot(kind="bar", ax=ax, width=0.8)
ax.set_ylabel("average topic share (%)")
ax.set_title("Topic shares by era")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(HERE / "figures" / "topics_by_era.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------------
# 4. Predict the year of a speech from its words (ridge regression, leave-one-out)
# ----------------------------------------------------------------------------
pipe = make_pipeline(TfidfVectorizer(stop_words="english", token_pattern=r"[A-Za-z]{3,}", min_df=3, sublinear_tf=True), RidgeCV(alphas=np.logspace(-1, 2, 12)))
pred = cross_val_predict(pipe, df.text, df.year, cv=LeaveOneOut())
mae = np.abs(pred - df.year).mean()
base = np.abs(df.year - df.year.median()).mean()
r2 = 1 - ((df.year - pred) ** 2).sum() / ((df.year - df.year.mean()) ** 2).sum()
log(f"Predicting the year from the words (60 speeches, leave-one-out): mean absolute error {mae:.1f} years (always guessing the median year: {base:.1f}); cross-validated R2 = {r2:.2f}")
pipe.fit(df.text, df.year)
coef = pipe[-1].coef_
vocab2 = np.array(pipe[0].get_feature_names_out())
log("Words most associated with later speeches: " + ", ".join(vocab2[np.argsort(-coef)[:10]]))
log("Words most associated with earlier speeches: " + ", ".join(vocab2[np.argsort(coef)[:10]]) + "\n")

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
for ax, v, lab in zip(axes[:2], ["sent_len", "fk_grade"], ["words per sentence", "Flesch-Kincaid grade (approx.)"]):
    ax.scatter(df.year, df[v], s=14)
    from statsmodels.nonparametric.smoothers_lowess import lowess
    sm_ = lowess(df[v], df.year, frac=0.4)
    ax.plot(sm_[:, 0], sm_[:, 1], "r-")
    ax.set_title(lab)
axes[2].scatter(df.year, pred, s=14)
axes[2].plot([1789, 2025], [1789, 2025], "k--", lw=0.8)
axes[2].set_xlabel("actual year")
axes[2].set_ylabel("predicted year (leave-one-out)")
axes[2].set_title("Predicting the year from the words")
fig.tight_layout()
fig.savefig(HERE / "figures" / "trends_and_prediction.png", dpi=150)
plt.close(fig)

log("Caveats: 60 speeches is a small corpus; the speeches are written for different audiences and media; stop-word removal and the bag-of-words")
log("representation ignore word order and meaning; topic models are sensitive to preprocessing and k, and topic labels are subjective; the readability")
log("formula and syllable counter are approximations designed for modern English; the year-prediction model may partly learn the era of a few memorable")
log("events (wars, crises) rather than general language change.")
out.close()
print("\nDone. See results.txt and figures/.")
