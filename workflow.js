export const meta = {
  name: "multi-uav-dsac-t-workflow",
  description:
    "Multi-UAV obstacle avoidance (DSAC-T + Box2D) — full lifecycle: env validation, network/algorithm verification, short training smoke test, demo visualization. 6 sequential phases mirroring CLAUDE.md phase plan.",
  phases: [
    { title: "Dependency & Environment Setup", detail: "pip install + import smoke test" },
    { title: "Environment Implementation", detail: "init, step, point-cloud, grid, tracking checks" },
    { title: "Networks Implementation", detail: "Actor + Critic forward pass dims" },
    { title: "DSAC-T Algorithm", detail: "Loss computation + 3 refinements" },
    { title: "Training Script", detail: "Short training (5000 steps), checkpoint save" },
    { title: "Demo / Visualization", detail: "Headless demo run" },
  ],
};

// ─────────────────────────────────────────────────────────────────────────
//  WORKFLOW
// ─────────────────────────────────────────────────────────────────────────
phase("Dependency & Environment Setup");

// ======================================================================
// Phase 0  –  Dependency & Environment Setup
// ======================================================================
await pipeline(
  [
    {
      label: "Pip install requirements",
      test: async () => {
        const { exitCode } = await agent(
          "Run: pip install -r requirements.txt. Report any errors."
        );
        if (exitCode !== 0) throw new Error("pip install failed");
      },
    },
    {
      label: "Import smoke test",
      test: async () => {
        const result = await agent(
          "Run: python -c \"import torch; import numpy; import matplotlib; import sklearn; import gym; print('All imports OK')\" and report the output."
        );
        if (!result || !result.includes("All imports OK")) throw new Error("Import smoke test failed");
      },
    },
    {
      label: "Create directories",
      test: async () => {
        await agent("Run: mkdir -p env checkpoints logs");
      },
    },
  ],
  // Stage 2: none needed — just validate
  (r) => r
);

// ======================================================================
// Phase 1  –  Environment  (fixedwing_env.py)
// ======================================================================
phase("Environment Implementation");

const envResults = await pipeline(
  [
    {
      label: "Env init & reset",
      test: async () => {
        const r = await agent(
          'Run: python -c "from env.fixedwing_env import MultiFixedWingEnv; env = MultiFixedWingEnv(num_uavs=5); obs = env.reset(); print(type(obs).__name__, len(obs))" and report the result.'
        );
        if (!r || r.includes("Error") || r.includes("Traceback")) throw new Error("env init failed: " + r);
        return r;
      },
    },
    {
      label: "100 random steps",
      test: async () => {
        const r = await agent(
          'Run: python -c "from env.fixedwing_env import MultiFixedWingEnv; import numpy as np; env = MultiFixedWingEnv(num_uavs=5); env.reset(); [env.step(env.action_space.sample()) for _ in range(100)]; print(\'100 steps OK\')"'
        );
        if (!r || !r.includes("100 steps OK")) throw new Error("100-step test failed");
        return r;
      },
    },
    {
      label: "Point cloud dims",
      test: async () => {
        const r = await agent(
          'Run: python -c "from env.fixedwing_env import MultiFixedWingEnv; import numpy as np; env = MultiFixedWingEnv(num_uavs=5); obs=env.reset(); [obs,_,_,_]=env.step(env.action_space.sample()); pc=obs[0][\'pointcloud\']; print(pc.shape, np.all(pc>=0))"'
        );
        if (!r) throw new Error("point cloud check failed");
        return r;
      },
    },
  ],
  (r) => r
);

// ======================================================================
// Phase 2  –  Networks  (networks.py)
// ======================================================================
phase("Networks Implementation");

await pipeline(
  [
    {
      label: "Actor forward pass",
      test: async () => {
        const r = await agent(
          'Run: python -c "import torch; from networks import Actor; a=Actor(160,160,60,6,25); g=torch.randn(4,1,160,160); p=torch.randn(4,60); s=torch.randn(4,6); d=torch.randn(4,25); mu,sig=a(g,p,s,d); ac,lp=a.sample(g,p,s,d); print(mu.shape, ac.shape, lp.shape)" and report the output.'
        );
        if (!r || r.includes("Error") || r.includes("Traceback")) throw new Error("Actor failed");
        return r;
      },
    },
    {
      label: "Critic forward pass",
      test: async () => {
        const r = await agent(
          'Run: python -c "import torch; from networks import DistributedCritic; c=DistributedCritic(160,160,60,6,25,32); g=torch.randn(4,1,160,160); p=torch.randn(4,60); s=torch.randn(4,6); d=torch.randn(4,25); ac=torch.randn(4,2); z=c(g,p,s,d,ac); print(z.shape)"'
        );
        if (!r || !r.includes("(4, 32)")) throw new Error("Critic output wrong shape: " + r);
        return r;
      },
    },
  ],
  (r) => r
);

// ======================================================================
// Phase 3  –  DSAC-T Algorithm  (dsac_t.py)
// ======================================================================
phase("DSAC-T Algorithm");

await pipeline(
  [
    {
      label: "Agent instantiation",
      test: async () => {
        const r = await agent(
          'Run: python -c "from dsac_t import DSACTAgent; a=DSACTAgent(160,160,60,6,25,3e-4,0.99,0.005,32,50); print(\'OK\')"'
        );
        if (!r || !r.includes("OK")) throw new Error("Agent init failed");
        return r;
      },
    },
    {
      label: "Synthetic update step",
      test: async () => {
        const r = await agent(
          'Run: python -c "import torch; from dsac_t import DSACTAgent, ReplayBuffer; buf=ReplayBuffer(10000); a=DSACTAgent(160,160,60,6,25,3e-4); for _ in range(256): buf.push({\'grid_map\':torch.randn(1,160,160),\'pointcloud\':torch.randn(60),\'self_state\':torch.randn(6),\'dynamic_obs\':torch.randn(25)},torch.randn(2),torch.tensor(-0.01),{\'grid_map\':torch.randn(1,160,160),\'pointcloud\':torch.randn(60),\'self_state\':torch.randn(6),\'dynamic_obs\':torch.randn(25)},torch.tensor(0.0)); b=buf.sample(256); cl,al,al2=a.update(b); print(f\'critic_loss={cl:.4f} actor_loss={al:.4f} alpha_loss={al2:.4f}\')"'
        );
        if (!r || r.includes("Error") || r.includes("Traceback")) throw new Error("Update failed");
        return r;
      },
    },
  ],
  (r) => r
);

// ======================================================================
// Phase 4  –  Training Script  (train.py)
// ======================================================================
phase("Training Script");

await pipeline(
  [
  {
    label: "Short training 5000 steps",
    test: async () => {
      const r = await agent(
        "Run: python train.py --total_steps 5000 --num_envs 4 --save_interval_min 1 --log_interval 100 and report the output (last 10 lines). Timeout 300s."
      );
      if (!r || r.includes("Traceback") || r.includes("Error")) throw new Error("Training crashed");
      return r;
    },
  },
  {
    label: "Checkpoint exists and loadable",
    test: async () => {
      const r = await agent(
        'Run: python -c "import torch,glob; fs=sorted(glob.glob(\'checkpoints/model_*.pth\')); print(f\'Found {len(fs)} checkpoints\'); c=torch.load(fs[-1],map_location=\'cpu\',weights_only=True); print(\'Keys:\',list(c.keys()))"'
      );
      if (!r || !r.includes("Found")) throw new Error("No checkpoint found");
      return r;
    },
  },
  ],
  (r) => r
);

// ======================================================================
// Phase 5  –  Demo / Visualization  (demo.py)
// ======================================================================
phase("Demo / Visualization");

await pipeline(
  [
  {
    label: "Headless demo 1 episode",
    test: async () => {
      const r = await agent(
        "Run: python demo.py --load_checkpoint checkpoints/model_*.pth --max_episodes 1 --headless --max_steps 200 and report the output (last 15 lines). Timeout 120s."
      );
      if (!r || r.includes("Traceback") || r.includes("Error")) throw new Error("Demo crashed");
      return r;
    },
  },
  ],
  (r) => r
);
