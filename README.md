# Handwritten Digit Classifier

A fully connected neural network written from scratch in plain Java, with no ML or math libraries, that classifies handwritten digits from the [MNIST](https://yann.lecun.com/exdb/mnist/) dataset with **94–96% accuracy** (measured on the 60,000-image training set).

## Authorship

The network is written by hand: the linear algebra, activation functions, forward pass, backpropagation math, and training loop design are my own. AI (Claude) was used for:

- the `MNISTLoader` class, which parses the IDX data files
- the `Makefile`
- reviewing and debugging my math, and restructuring how `train()` accumulates gradients per layer and applies the update

## Math implemented

- **Linear algebra from scratch:** matrix-vector products, transposes, outer products, and elementwise operations on `float` arrays (`LinearAlgebra`).
- **Layers:** each layer computes an affine map **z = Wa + b** followed by an activation. The default architecture is 784 → 16 → 16 → 10, and the number and width of hidden layers are configurable.
- **Sigmoid hidden activation** with derivative **σ′(z) = σ(z)(1 − σ(z))**, evaluated at the cached pre-activation z.
- **Softmax output layer**, made numerically stable by subtracting the max logit before exponentiating.
- **Cross-entropy loss** against one-hot labels. Combined with softmax, the output error simplifies to **∂L/∂z = p − y**.
- **Backpropagation** using the error recurrence **δ⁽ⁱ⁾ = (W⁽ⁱ⁺¹⁾)ᵀ δ⁽ⁱ⁺¹⁾ ⊙ σ′(z⁽ⁱ⁾)**, with parameter gradients **∂L/∂W⁽ⁱ⁾ = δ⁽ⁱ⁾ ⊗ a⁽ⁱ⁻¹⁾** and **∂L/∂b⁽ⁱ⁾ = δ⁽ⁱ⁾**.
- **Mini-batch gradient descent:** the training set is shuffled each epoch, split into batches of 32, and each batch's gradients are averaged before a step of **θ ← θ − η∇L**.
- **Initialization:** weights are drawn from a Gaussian with standard deviation 0.01 to break symmetry between neurons; biases start at zero.
- **Input normalization:** pixel values are scaled from 0–255 to 0–1.

## Running

Place the MNIST IDX files in `data/` (`train-images.idx3-ubyte`, `train-labels.idx1-ubyte`), then run from the project root:

```bash
make run
```

This compiles to `out/` and runs `Main`, which trains for 10 epochs (learning rate 1.0, batch size 32) and prints each prediction followed by the total accuracy. Requires a JDK and GNU make. `make clean` removes the compiled output.
