# defect-detection

Surface defect detection with confidence thresholding and a human review handoff.

The idea is the one you actually want on an inspection line. A classifier looks
at an image of a manufactured surface and decides whether it is defective or
clean. When the classifier is confident it makes the call automatically. When it
is unsure it hands the case to a person instead of guessing. That second part,
the review queue, is what keeps a model honest in production: the borderline
cases are exactly the ones a human should see.

## What is in the data

There is no download. The images are synthetic and generated on the fly so the
whole thing runs on a CPU in seconds and is fully reproducible from a seed.

A clean sample is a smooth, regular texture built from a few sine gratings plus
a little Gaussian noise, which stands in for the steady pattern of a material
surface. A defective sample takes a clean texture and stamps a localized anomaly
onto it: a bright or dark blob, or a thin scratch. That mirrors what a real
defect looks like, a sharp local break in an otherwise regular pattern.

## How the detector works

Rather than a heavy network, the detector reads a small set of texture features
from each image and fits a logistic regression on top of them. The features are
chosen around the fact that a defect is local: the extremes of intensity, the
shape of the intensity histogram, the high end of the gradient magnitude, and
the fraction of pixels that are strong outliers against a local median. A blob
edge or a scratch lights up the gradient and outlier features, which is the
signal the classifier learns from.

Logistic regression hands back a probability for the defective class. Confidence
is read straight off that probability as its distance from the 0.5 boundary, so
a prediction near 0.5 is treated as uncertain and a prediction near 0 or 1 is
treated as confident.

## The review handoff

`route_predictions` takes the probabilities and a confidence threshold and
splits every prediction into two streams. Anything at or above the threshold is
decided automatically. Anything below it lands in a `ReviewQueue`, sorted with
the most uncertain case first so a human works the worst cases first. Raise the
threshold and more cases go to review, which is the dial you turn when the cost
of a wrong automatic decision is high.

## Results

These come from an actual run on this machine, not from a reference number.

Training on 160 images and testing on a held out 80, the detector reached 0.963
accuracy, well clear of the 0.5 chance rate on a balanced two class problem. At a
confidence threshold of 0.75 it decided 68 of the 80 test cases automatically and
routed the remaining 12 to the review queue. The reviewed cases were the low
confidence ones, with probabilities sitting close to 0.5, which is exactly the
behavior you want.

Run it yourself:

```
python demo.py
```

## Layout

```
src/
  data.py      synthetic textured images, clean and defective
  features.py  texture feature extraction
  model.py     the logistic regression detector
  review.py    confidence routing and the review queue
tests/         pytest behavior tests
demo.py        end to end run with a short report
```

## Tests

```
python -m pytest tests/ -q
```

The tests check behavior, not magic numbers. The data is balanced and
reproducible, defect images carry stronger gradient and outlier signals than
clean ones, the classifier clearly beats chance on a held out split, every
prediction is routed exactly once, the review queue captures the low confidence
predictions, and raising the threshold sends more cases to review.

## Install

```
pip install -r requirements.txt
```
