import torch
import torch.nn as nn

from collections import OrderedDict

def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()

class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # For compatibility with Torchmeta
        self.named_meta_parameters = self.named_parameters
        self.meta_parameters = self.parameters

    def update_params(self, loss, params=None, step_size=0.5, first_order=False):
        """Apply one step of gradient descent on the loss function `loss`, with 
        step-size `step_size`, and returns the updated parameters of the neural 
        network.
        """
        if params is None:
            params = OrderedDict(self.named_meta_parameters())

        grads = torch.autograd.grad(loss, params.values(),
                                    create_graph=not first_order)

        #print('loss:', loss)
        updated_params = OrderedDict()
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = param - step_size * grad
            #print(f"(grad) {name:<23}: Min={grad.min().item():<6.2f} Max={grad.max().item():<6.2f} Mean={grad.mean().item():<6.2f} StdDev={grad.std().item():<6.2f} Sum={param.sum().item():<6.2f}")

        #for key, tensor in updated_params.items():
            #print(f"{key:<23}: Min={tensor.min().item():<6.2f} Max={tensor.max().item():<6.2f} Mean={tensor.mean().item():<6.2f} StdDev={tensor.std().item():<6.2f} Sum={tensor.sum().item():<6.2f}")
        #print('-'*30)

        return updated_params
