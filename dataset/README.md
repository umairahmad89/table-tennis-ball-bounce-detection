# OpenTTGames Dataset

This dataset was created using the OpenTTGames dataset, by extracting ball coordinates from both the `train` and `test` data.

## Dataset Format

There are three events in the OpenTTGames dataset: `bounce`, `net`, and `empty` events. In this dataset, we have used the `net` and `empty` events as `not_bounce` events and assigned them the class label `0`, while the `bounce` event is assigned the class label `1`.

Here is an example of how each entry in the dataset is structured:

[
[
[x0, y0], [x1, y1], [x2, y2], ..., [x8, y8], [0/1], ....
]
]

- Each entry in the dataset contains a sequence of 9 ball coordinates represented as pairs `[x, y]`.
- The last element in the sequence is the class label, where `0` represents a `not_bounce` event, and `1` represents a `bounce` event.

This format provides a clear understanding of how the dataset is structured and the meaning of the class labels.

