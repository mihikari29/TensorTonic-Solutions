def value_iteration_step(
    values: list,
    transitions: list,
    rewards: list,
    gamma: float
) -> list[float]:
    """
    Perform one value-iteration update for every state.
    """
    new_values = []

    for s in range(len(values)):
        action_values = []

        for a in range(len(transitions[s])):
            expected_next_value = sum(
                transitions[s][a][next_s] * values[next_s]
                for next_s in range(len(values))
            )

            q_value = rewards[s][a] + gamma * expected_next_value
            action_values.append(q_value)

        new_values.append(float(max(action_values)))

    return new_values