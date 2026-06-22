"""Run the full pipeline end to end and print a short report.

    python demo.py
"""

from src.data import generate_dataset
from src.model import DefectDetector
from src.review import route_predictions


def main() -> None:
    train_imgs, train_lbls = generate_dataset(n_per_class=80, seed=0)
    test_imgs, test_lbls = generate_dataset(n_per_class=40, seed=1)

    model = DefectDetector(seed=0).fit(train_imgs, train_lbls)
    accuracy = model.score(test_imgs, test_lbls)

    probs = model.predict_proba(test_imgs)
    result = route_predictions(probs, confidence_threshold=0.75)

    print(f"test images        : {len(test_lbls)}")
    print(f"test accuracy      : {accuracy:.3f}")
    print(f"auto decided       : {result.n_auto}")
    print(f"sent for review    : {result.n_review}")
    if result.n_review:
        print("most uncertain cases (index, probability, confidence):")
        for item in result.review_queue.items[:5]:
            print(
                f"  index {item.index:>3}  "
                f"p={item.probability:.3f}  conf={item.confidence:.3f}"
            )


if __name__ == "__main__":
    main()
