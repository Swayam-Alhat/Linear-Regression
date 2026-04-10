# Decision Trees

## What is a Decision Tree?

A decision tree is an ML algorithm that learns patterns from data and predicts outcomes for new unseen data. It forms a tree-like structure made of nodes and branches — essentially a flowchart of questions that leads to a prediction.

Each node in the tree represents a feature (like Glucose, Age, Insulin). Each branch represents a split based on that feature's value. At the bottom are leaf nodes — these give the final prediction.

---

## How the Algorithm Learns

The core idea is simple:

> **Find the feature and threshold that best separates the data into two classes (0 and 1), then repeat on each resulting subset.**

This is done by measuring **Gini impurity** — a score that tells how mixed a group is. A Gini of `0` means the group is perfectly pure (all one class). A Gini of `0.5` means it's perfectly mixed (50/50 split).

### Gini Impurity Formula

$$Gini = 1 - (p_0^2 + p_1^2)$$

Where $p_0$ and $p_1$ are the proportions of class 0 and class 1 in the node.

When evaluating a split, the Gini scores of both child nodes are combined into a **weighted Gini**:

$$Weighted\ Gini = \frac{n_{left}}{n_{total}} \times Gini_{left} + \frac{n_{right}}{n_{total}} \times Gini_{right}$$

The split with the **lowest weighted Gini** wins.

### Step-by-Step: How the Tree is Built

**Step 1 — Find the best split**

For every feature:

- Sort its unique values and compute candidate thresholds (midpoints between consecutive values)
- For each threshold, split the data into left (≤ threshold) and right (> threshold)
- Calculate weighted Gini for that split
- Pick the threshold with the lowest weighted Gini → this is the **best split for that feature**

Repeat for all features. The feature + threshold with the globally lowest weighted Gini becomes the **node**.

**Step 2 — Split and recurse**

The data is divided into two subsets based on the winning split. For each subset, the algorithm checks:

- Is this node **pure enough**? (≥ 90% of samples belong to one class) → assign that class as a leaf node, stop
- Is this node **too small**? (fewer than 5 samples after a split) → assign the majority class, stop
- Otherwise → go back to **Step 1** and find the best split for this subset

This recursion continues until every branch ends at a leaf node.

### Example

Take the root split: **Glucose ≤ 154.5**

The algorithm tested every threshold for every feature. Splitting on Glucose at 154.5 produced the lowest weighted Gini across all candidates — meaning this single question separates diabetic vs non-diabetic patients most cleanly. Patients with Glucose > 154.5 are very likely to be diabetic, so the right branch almost immediately reaches a leaf.

---

## Implementation

The implementation has three functions:

### `is_pure_enough(data)`

Checks whether a node's data is pure enough to stop splitting. It finds the most frequent class in the target column and checks if it makes up ≥ 90% of the samples. Returns the result and the majority class.

```python
def is_pure_enough(data):
    most_frequent_value = data.iloc[:,-1].value_counts().idxmax()
    ocurrence = data.iloc[:,-1].value_counts().max()
    is_pure = ((ocurrence / len(data)) * 100) >= 90
    return [is_pure, most_frequent_value]
```

### `get_best_feature_split(node_data)`

For each feature, generates candidate split thresholds as midpoints between sorted unique values. Then calculates weighted Gini for every threshold and picks the best one per feature. Finally compares across all features and returns the globally best `[feature, split_value]`.

```python
split_arr = (sorted_feature[:-1] + sorted_feature[1:]) / 2
```

```python
weighted_gini = (len(left) / len(data)) * gini_left + (len(right) / len(data)) * gini_right
```

### `build_tree(node_data)`

The recursive engine. Checks stopping conditions first (purity, min size), then calls `get_best_feature_split()`, splits the data, and recurses on both halves. Each internal node is stored as a Python dict:

```python
node = {
    "feature": best_feature,
    "split_value": best_split_value,
    "left_node": build_tree(left_node_data),   # recursive
    "right_node": build_tree(right_node_data)  # recursive
}
```

Leaf nodes are just an integer — `0` or `1`.

### Trained Tree Structure

```
[Glucose <= 154.5?]
├── YES → [Age <= 28.5?]
│   ├── ✅ No Diabetes (0)
│   └── NO  → [Insulin <= 142.5?]
│       ├── YES → [Glucose <= 100.5?]
│       │   ├── ✅ No Diabetes (0)
│       │   └── NO  → [Age <= 47.0?]
│       │       ├── YES → [BloodPressure <= 79.0?]
│       │       │   ├── ✅ No Diabetes (0)
│       │       │   └── 🔴 Diabetes (1)
│       │       └── ✅ No Diabetes (0)
│       └── NO  → [BloodPressure <= 77.0?]
│           ├── ✅ No Diabetes (0)
│           └── 🔴 Diabetes (1)
└── NO  → [Insulin <= 544.0?]
    ├── 🔴 Diabetes (1)
    └── ✅ No Diabetes (0)
```

The tree reads naturally: a patient with Glucose > 154.5 and Insulin ≤ 544 is predicted diabetic. A patient with Glucose ≤ 154.5 and Age ≤ 28.5 is predicted non-diabetic. Every path ends at a clear prediction.

To print this tree from your own trained tree, add this function:

```python
import numbers

def print_tree(node, prefix="", is_left=None):
    if isinstance(node, numbers.Number):
        label = "✅ No Diabetes (0)" if node == 0 else "🔴 Diabetes (1)"
        connector = "├── " if is_left is True else "└── " if is_left is False else ""
        print(f"{prefix}{connector}{label}")
        return

    feature = node["feature"]
    split = node["split_value"]

    if is_left is None:
        print(f"[{feature} <= {split}?]")
        child_prefix = ""
    elif is_left:
        print(f"{prefix}├── YES → [{feature} <= {split}?]")
        child_prefix = prefix + "│   "
    else:
        print(f"{prefix}└── NO  → [{feature} <= {split}?]")
        child_prefix = prefix + "    "

    print_tree(node["left_node"],  child_prefix, is_left=True)
    print_tree(node["right_node"], child_prefix, is_left=False)

print_tree(final_decision_tree)
```

---

## Pruning

After the tree is built, some internal nodes end up with both children predicting the same class. These nodes add complexity without changing any prediction — they're useless splits.

**Example from this implementation:**

Before pruning, this subtree existed:

```
[SkinThickness <= 34.5?]
├── ✅ No Diabetes (0)
└── ✅ No Diabetes (0)
```

Both children predict `0`, so the split on SkinThickness is pointless. Pruning collapses this entire subtree into a single leaf: `0`.

### How `prune_tree()` Works

It walks the tree **bottom-up** using recursion. At each internal node, it first recurses into both children (pruning them first), then checks: are both children now leaf nodes with the same value? If yes, replace the entire node with that value.

```python
def prune_tree(node):
    if isinstance(node, numbers.Number):
        return node

    node["left_node"] = prune_tree(node["left_node"])
    node["right_node"] = prune_tree(node["right_node"])

    if isinstance(node["left_node"], numbers.Number) and isinstance(node["right_node"], numbers.Number):
        if node["left_node"] == node["right_node"]:
            return node["left_node"]   # collapse

    return node
```

The bottom-up order matters — a node can only be pruned after its children are already pruned.

---

## Accuracy

Tested on held-out data (samples 373 onwards from the filtered Pima Indians dataset):

| Metric              | Value      |
| ------------------- | ---------- |
| Test samples        | 161        |
| Correct predictions | 129        |
| **Accuracy**        | **80.12%** |

**Dataset:** Pima Indians Diabetes dataset. Features used: Glucose, BloodPressure, SkinThickness, Insulin, Age. Removed: Pregnancies, BMI, DiabetesPedigreeFunction. Rows with zero values in any used feature were filtered out.
