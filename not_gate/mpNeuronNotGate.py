import torch

# Define the MP Neuron python class implementing NOT gate
class MPNeuronNOT(torch.nn.Module):
    def __init__(self):
        super(MPNeuronNOT, self).__init__()
        self.weight = torch.tensor([-1.0])  # Fixed weight for NOT gate

    def forward(self, x: torch.Tensor) -> float:
        # Apply threshold condition directly
        if x * self.weight < 0:
            return 0.0
        else:
            return 1.0

# Instantiate the NOT neuron
mp_neuron_not = MPNeuronNOT()

# Test different the 2 different inputs for the NOT gate
test_inputs = [torch.Tensor([0]), torch.Tensor([1])]
for input_data in test_inputs:
    output = mp_neuron_not(input_data)
    print(f"Output of NOT Gate ({int(input_data.item())}): {output}")
