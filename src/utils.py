def lr_decay(step):
    decay_steps = decay_steps * ceil(step / decay_steps)
    return ((initial_learning_rate - end_learning_rate) * (1 - step / decay_steps) ^ (power)) + end_learning_rate