class Perceptron {
  constructor(size, learningRate = 0.1) {
    this.weights = new Array(size).fill(0);
    this.bias = 0;
    this.lr = learningRate;
  }

  predict(inputs) {
    let sum = this.bias;

    for (let i = 0; i < inputs.length; i++) {
      sum += this.weights[i] * inputs[i];
    }

    return sum >= 0 ? 1 : 0;
  }

  train(inputs, label) {
    const prediction = this.predict(inputs);
    const error = label - prediction;

    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] += this.lr * error * inputs[i];
    }

    this.bias += this.lr * error;
  }
}