import optax


def build_scheduler(lr, warmup_steps):
    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value=0.0, end_value=lr, transition_steps=warmup_steps
            ),
            optax.constant_schedule(lr),
        ],
        boundaries=[warmup_steps],
    )
    return schedule
