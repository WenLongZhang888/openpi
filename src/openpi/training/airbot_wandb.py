"""Optional W&B metrics for staged AIRBOT training; checkpoints stay local."""

from pathlib import Path

import wandb


class WandbLogger:
    def __init__(self, run_dir, config, *, resume=False):
        self.run = None
        settings = config.get("wandb", {})
        if not settings.get("enabled", False):
            return
        run_dir = Path(run_dir)
        identity = run_dir / "wandb_id.txt"
        run_id = identity.read_text().strip() if identity.exists() else wandb.util.generate_id()
        identity.write_text(run_id + "\n")
        self.run = wandb.init(
            project=settings["project"], entity=settings.get("entity"),
            group=settings.get("group"), name=settings.get("name", run_dir.name),
            job_type=config["mode"], id=run_id, resume="allow" if resume else "never",
            mode=settings.get("mode", "online"), dir=str(run_dir), config=config,
            save_code=False,
        )
        # Custom optimizer step also handles replay of updates after a checkpoint.
        self.run.define_metric("step")
        self.run.define_metric("*", step_metric="step")
        (run_dir / "wandb_url.txt").write_text((self.run.url or "offline") + "\n")

    def log(self, metrics):
        if self.run is not None:
            self.run.log(metrics)

    def checkpoint(self, step, path):
        if self.run is not None:
            self.run.summary.update({"last_checkpoint_step": step, "last_checkpoint": str(path)})

    def complete(self, step):
        if self.run is not None:
            self.run.summary.update({"completed_steps": step, "training_complete": True})
