from mnist import MNIST


mnist = MNIST()

# Train set is lazily loaded into memory and cached afterward
mnist.train_set.images  # (60000, 784)
mnist.train_set.labels  # (60000, 10)

# Test set is lazily loaded into memory and cached afterward
mnist.test_set.images   # (10000, 784)
mnist.test_set.labels   # (10000, 10)

# Yield minibatches from the shuffled train set
for images, labels in mnist.train_set.minibatches(batch_size=256):
    pass
