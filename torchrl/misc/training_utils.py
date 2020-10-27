def update_network_variables(target, source, tau=1.0):
    if tau == 1.0:
        target.load_state_dict(source.state_dict())
    else:
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.mul_(1.0 - tau)
            t.data.add_(s.data * tau)
