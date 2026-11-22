# Project 28: Text as data - how has the language of US inaugural addresses changed, 1789-2025?

**Data:** the Inaugural Address Corpus in NLTK: 60 speeches from Washington (1789) to 2025, 140,987 words (public-domain speeches; loaded with `nltk.download("inaugural")`, text not
stored here).

## Questions
1. How do speech length, sentence length, readability and lexical diversity change over time?
2. Which words distinguish the eras?
3. Which topics appear, and how do their shares change?
4. Can the year of a speech be predicted from its words?

## Method
Regression with HC3 standard errors on simple text statistics (Flesch-Kincaid grade from a rough syllable counter, moving-average type-token ratio); weighted log-odds with an informative
Dirichlet prior (Monroe, Colaresi and Quinn 2008) per era; non-negative matrix factorisation (k = 8, chosen with UMass coherence and interpretability) on 683 chunks of about 8
sentences; ridge regression of the year on TF-IDF features with leave-one-out cross-validation.

## Results
- **Sentences became much shorter:** 44 words on average in 1789-1850, 32 in 1853-1900, 24 in 1901-1950, 21 in 1953-2000 and 19 since 2001; the approximate reading grade fell from 21 to 10.
  Speech length shows no significant trend (p = 0.38) and lexical diversity is almost flat (0.509 to 0.492).
- **Era vocabulary tracks history:** 1789-1850 "union, powers, foreign, state"; 1853-1900 "constitution, laws, slavery, partisan"; 1901-1950 "business, economic, industrial, international";
  1953-2000 "world, let, new, help, century"; 2001-2025 "america, thank, americans, story, jobs".
- **Topics:** a "states, union, constitution, powers" topic falls from 40% of the text in 1789-1850 to 4% in 2001-2025, while a "new, america, world, nation" topic rises from 2% to 27% (topic
  labels are my interpretation of the word lists); tariff and revenue language peaks in 1853-1950 and peace/war language in 1901-1950.
- **The year can be predicted from the words:** leave-one-out mean absolute error 20.4 years (60.0 years for always guessing the median), cross-validated R-squared 0.84.

## Limitations
- Small corpus of 60 speeches written for different audiences and media; bag-of-words ignores order and meaning; topic models depend on preprocessing and k; the readability formula and syllable
  counter are approximations for modern English; the year model may partly learn memorable events.

## Possible extensions
Word embeddings, speaker party and sentiment analysis, change-point detection in sentence length, comparison with State of the Union addresses, German political speeches (Bundestag corpus).
