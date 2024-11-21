# Building GUI
zig build -Doptimize=ReleaseFast

# Building training application
zig build -Doptimize=ReleaseFast

# Architecture
## Results
Correctly predicts 59971/60000 (99.95%) of training samples.

Correctly predicts 9854/1000   (98.54%) of test samples.

## Design
28x28 (784) normalized pixel luminance -> 512 neuron hidden layer with 50% dropout -> 10 neuron output layer
Uses ReLU for the hidden layer and softmax for the output layer with cross entropy loss as error function
