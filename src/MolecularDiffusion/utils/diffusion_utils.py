import torch


def compute_mean_mad_from_dataloader(props, task_names):
    property_norms = {}
    for i, property_key in enumerate(task_names):
        values = props[i]
        mean = torch.mean(values)
        ma = torch.abs(values - mean)
        mad = torch.mean(ma)
        property_norms[property_key] = {}
        property_norms[property_key]["mean"] = mean
        property_norms[property_key]["mad"] = mad
        property_norms[property_key]["max"] = torch.max(values)
        property_norms[property_key]["min"] = torch.min(values)
    return property_norms


# %% in task
def prepare_context(task_names, minibatch, property_norms, normalization_method="maxmin"):
    batch_size, n_nodes, _ = minibatch["coords"].size()
    node_mask = minibatch["node_mask"].unsqueeze(2)
    context_node_nf = 0
    context_list = []
    for key in task_names:
        properties = minibatch[key]
        if normalization_method is not None:
            if normalization_method == "mad":
                properties = (properties - property_norms[key]["mean"]) / property_norms[key][
                    "mad"
                ]
            elif normalization_method == "maxmin": # [-1, 1]
                properties = 2*(properties - property_norms[key]["min"]) / (  
                    property_norms[key]["max"] - property_norms[key]["min"]
                ) - 1
            elif "value" in normalization_method: # "value_n where n is the value to normalize"
                value = float(normalization_method.split("_")[1])
                properties = properties / value
            else:
                raise ValueError(f"Unknown normalization method: {normalization_method}")
    
        if len(properties.size()) == 1:
            # Global feature.
            assert properties.size() == (batch_size,)
            reshaped = properties.view(batch_size, 1, 1).repeat(1, n_nodes, 1)
            context_list.append(reshaped)
            context_node_nf += 1
        elif len(properties.size()) == 2 or len(properties.size()) == 3:
            # Node feature.
            assert properties.size()[:2] == (batch_size, n_nodes)

            context_key = properties

            # Inflate if necessary.
            if len(properties.size()) == 2:
                context_key = context_key.unsqueeze(2)

            context_list.append(context_key)
            context_node_nf += context_key.size(2)
        else:
            raise ValueError("Invalid tensor size, more than 3 axes.")
    # Concatenate
    context = torch.cat(context_list, dim=2)
    # Mask disabled nodes!
    context = context * node_mask
    assert context.size(2) == context_node_nf
    return context

