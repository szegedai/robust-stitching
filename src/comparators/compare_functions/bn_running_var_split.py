

# import numpy as np
# from src.comparators.general import ActivationComparator

# class BnActivation(ActivationComparator):

#     def __init__(self, dataset, batch_size=50, n_epochs=1, n_iters=-1, percentile=90):
#         super().__init__(dataset, batch_size, n_epochs, n_iters)
#         self.percentile = percentile

#     def similarity(self, frank_act, model2_act):
#         dist = (frank_act - model2_act) ** 2
#         dist_below = dist[:, ~self.bn_mask]
#         dist_above = dist[:, self.bn_mask]
#         return dist_below, dist_above

#     def _get_similarity_on_layer(self, model1, model2, layer):
#         similarities = self._get_running_similarities(model1,
#                                                               model2,
#                                                               layer)

#         x1 = np.mean([x[0] for x in similarities])
#         x2 = np.mean([x[1] for x in similarities])
#         return x1, x2

#     def _rearrange_activations(self, activations):
#         if len(activations.shape) > 2:
#             activations = np.transpose(activations, axes=[0, 2, 3, 1])
#             n_channels = activations.shape[-1]
#             flat_activations = activations.reshape(-1, n_channels)
#         else:
#             flat_activations = activations
#         return list(flat_activations)


#     def _get_running_similarities(self, frank_model, model2, layer):

#         self.bn_mask = self._get_bn_mask(frank_model, layer)
#         return super()._get_running_similarities(frank_model, model2, layer)

#     def _get_bn_mask(self, frank_model, stop_layer):
        
#         layer_list = [name for (name, _) in frank_model.named_modules()]
#         bn_layer = layer_list[layer_list.index(stop_layer)+3]
#         running_vars = getattr(frank_model, bn_layer).running_var
#         running_vars = running_vars.detach().cpu().numpy()
#         threshold = np.percentile(running_vars, self.percentile)
#         return running_vars > threshold