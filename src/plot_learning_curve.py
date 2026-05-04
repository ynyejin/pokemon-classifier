import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5]
train_loss = [2567.0744, 2278.9646, 2063.2607, 1801.5039, 1539.2719]

plt.figure()
plt.plot(epochs, train_loss, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Learning Curve - ResNet50 Scratch")
plt.grid(True)
plt.savefig("assets/learning_curve.png", dpi=200, bbox_inches="tight")
plt.show()