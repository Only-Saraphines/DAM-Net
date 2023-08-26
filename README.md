# DAM-Net
We present a multitask network based on densenet and attention mechanism(DAM-Net).
The network introduces attention mechanism to enhance feature extraction ability, and uses two brain structure images related to brain atrophy as auxiliary subtasks to dynamically update model parameters, thereby optimizing the classification results of the main task Alzheimer's disease.Fig. 1 illustrates the overview of the proposed network.We use densenet to extract image sharing features, then use attention mechanism to weight image features, and finally use different full classification layers to obtain classification output.
