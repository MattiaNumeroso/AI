import numpy as np
import matplotlib.pyplot as plt

class SingleLayerPerceptron:
    """Single Layer Perceptron from scratch"""

    def __init__(self, input_size, learning_rate=0.01):
        """
        Create the perceptron
        Args:
            input_size: number of input features
            learning_rate for weight updates
        """

        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def forward(self, x):
        """
        Forward propagation -> weighted sum + bias
        """

        return np.dot(x, self.weights) + self.bias

    def train(self, X, y, epochs=100):
        """
        Train the perceptron using MSE 
        Args:
            X: input data (N, D) where D is number of features and N number of samples
            y: target (N,)
            epochs: number of training epochs

        Returns:
            list of MSE errors per epoch to plot then learning curve
        """
        errors = []

        for epoch in range(epochs):
            total_error = 0

            for i in range(len(X)):
                # Forward propagation
                prediction = self.forward(X[i])

                # Error -> SQUARED ERROR
                error = y[i] - prediction
                total_error += error ** 2

                # Weight update
                self.weights += self.learning_rate * error * X[i] 
                self.bias += self.learning_rate * error

            # Salva errore medio dell'epoca
            mse = total_error / len(X)
            errors.append(mse)

            if epoch % 100 == 0:
                print(f"Epoca {epoch}, MSE: {mse:.6f}")

        return errors


def main():

    print("Function to approximate: f(x) = x*2 + 2")
    print("=" * 20)

    #Training data [X,y] where y = X*2 + 2
    np.random.seed(42)
    X_train = np.linspace(-5, 5, 100).reshape(-1, 1)
    y_train = X_train[:, 0] * 2 + 2

    # Create the model 
    slp = SingleLayerPerceptron(input_size=1, learning_rate=0.01)

    print("\n Training the SLP...")
    errors = slp.train(X_train, y_train, epochs= 100)

    # Test predictions
    predictions = np.array([slp.forward(x) for x in X_train])

    # Get final MSE
    final_mse = np.mean((y_train - predictions) ** 2)
    print(f"\n final MSE: {final_mse:.6f}")

    # Plot the error curve and predictions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Learning curve
    ax1.plot(errors, linewidth=2, color='blue')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('MSE', fontsize=12)
    ax1.set_title('Learning curve Single Layer Perceptron', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: Function approximation
    ax2.plot(X_train, y_train, 'b-', linewidth=2, label='f(x) = x² + 2 (target)', alpha=0.7)
    ax2.plot(X_train, predictions, 'r--', linewidth=2, label='Prediction SLP', alpha=0.7)
    ax2.scatter(X_train[::5], y_train[::5], color='blue', s=30, alpha=0.5, zorder=3)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_title('Function approximation', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/mattia/Desktop/RNN/AI/SLP/slp_results.png', dpi=150, bbox_inches='tight')
    print(f"\PNG saved in: /home/mattia/Desktop/RNN/AI/SLP/slp_results.png")
    plt.show()

if __name__ == "__main__":
    main()
