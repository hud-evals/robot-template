# robot-template

SmolVLA driving the [LIBERO](https://libero-project.github.io/) benchmark over HUD's
`robot` capability. `environment/env.py` declares the env (`env.gym` + one template),
`environment/lerobot_sim.py` adapts lerobot's `make_env` (the task is an episodic
template arg), and `run.py` is the agent and runner.

The sim stack (MuJoCo, LIBERO) lives only in the Docker image; locally you install
the agent side (the policy + the HUD SDK):

```bash
pip install 'hud-python[robot] @ git+https://github.com/hud-evals/hud-python.git' 'lerobot[smolvla]'
docker build -f Dockerfile.hud -t hud-libero-env .
python run.py
```

`python run.py --suite libero_goal -n 10` picks the suite and task count. With
`HUD_API_KEY` set, rollouts stream to the trace viewer on [hud.ai](https://hud.ai);
`hud deploy` hosts the env on the platform instead of local Docker.

Docs: [docs.hud.ai/v6/advanced/robots](https://docs.hud.ai/v6/advanced/robots)
