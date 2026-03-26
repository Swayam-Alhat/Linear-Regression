# What Gradient descent actually does

Gradient descent calculates how a loss function changes when a parameter changes slightly — and by how much. This tells us in which direction and by how much to update each parameter to reduce the loss.

For linear regression, the loss function is **MSE**. We want to find values of `m` and `b` that minimize it. (**_As we know MSE is Mean Error, so obviously, it should decrease_**)

We calculate the gradient of `m` — how **MSE** changes when `m` changes:

- If `grad(m)` is positive → **MSE** increases as `m` increases. So `m` should decrease
- If `grad(m)` is negative → **MSE** decreases as `m` increases. So `m` should increase

The update formula handles this automatically

```
m = m - learning_rate * grad(m)
```

When `grad(m)` is positive, we subtract it from `m`, making `m` smaller, which brings **MSE** down.

When `grad(m)` is negative, we add (because substracting a number (m) with negative number (grad(m) actually sums up ) it to `m`, making `m` larger, which brings **MSE** down

**_The learning rate controls how large each step is_**.

The same update is applied to `b`.

> For logistic regression, the loss function is Log Loss (negative log likelihood) instead of MSE. The gradient descent loop is identical — only the loss function and its gradient formulas differ.

## Gradient descent for Logistic regression

Forget gradient descent for a second.

Lets say,

You have Likelihood. It outputs 0.3 right now. You want it to output close to 1.

The only way to change Likelihood's output is to **change m and b**. There's no other way.

So the real question is — **in which direction should m change, so that Likelihood goes up?**

To answer that, you need to know: _how does Likelihood change when m changes?_

That's the gradient. `grad(m)` = how Likelihood changes when m increases slightly.

- If `grad(m)` is positive → Likelihood goes **up** when m increases → so **increase m**
- If `grad(m)` is negative → Likelihood goes **down** when m increases → so **decrease m**

So to increase Likelihood, the update should be:

```
m = m + lr * grad(m)
```

Notice the **plus**. This is called **gradient ascent** — climbing uphill instead of downhill.

---

## Now the problem

Gradient descent update formula is hardcoded as:

```
m = m - lr * grad(m)
```

That minus is permanent. You cannot change it. The whole algorithm, all the code, all the math is built around this minus.

So if you try to use gradient descent directly on Likelihood, it will move m in the **wrong direction** and make Likelihood smaller, not larger.

---

## The -1 trick

If you multiply Likelihood by -1, something interesting happens to the gradient:

```
grad of (-Likelihood) = -grad of (Likelihood)
```

The gradient just flips sign.

So now:

- Original grad(m) was positive → Likelihood was going up → you wanted to increase m
- New grad(m) is **negative** → gradient descent subtracts a negative → **increases m** ✓

The minus in the update formula and the minus from multiplying by -1 **cancel each other out**. Result is the same direction you wanted.

---

## In one line

Multiplying by -1 flips the gradient sign. This cancels with the minus in the update formula. So gradient descent ends up moving m in the direction that **increases** Likelihood — even though it thinks it's minimizing.
