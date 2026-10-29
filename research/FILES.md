# File guide

The website in `docs/` is unchanged. This table describes every file in the cleaned research release; package initializers and optional shared model support are listed explicitly.

| File | Purpose |
| --- | --- |
| [.gitignore](../.gitignore) | Excludes datasets, checkpoints, caches, environments, outputs and installed LIBERO. |
| [LICENSE](../LICENSE) | Apache-2.0 license retained from the research source. |
| [README.md](../README.md) | Public research entry point and links to setup, experiments, baselines and method details. |
| [assets/sambar/broad/norm_stats.json](../assets/sambar/broad/norm_stats.json) | Unchanged 117-task state/action normalization statistics, relocated from the source asset directory. |
| [assets/sambar/narrow/norm_stats.json](../assets/sambar/narrow/norm_stats.json) | Unchanged 32-task state/action normalization statistics, relocated from the source asset directory. |
| [baselines/README.md](../baselines/README.md) | Baseline commands, retained defaults, sequential protocols and source limitations. |
| [baselines/__init__.py](../baselines/__init__.py) | Python package initializer. |
| [baselines/_impl/__init__.py](../baselines/_impl/__init__.py) | Python package initializer. |
| [baselines/_impl/cotrain.py](../baselines/_impl/cotrain.py) | Requested co-training implementation: mixes old/new batches and trains with supervised loss. |
| [baselines/_impl/ewc.py](../baselines/_impl/ewc.py) | EWC update with the source old-demo importance estimator, optional normalization and prior-stage task IDs. |
| [baselines/_impl/ewc_broad.py](../baselines/_impl/ewc_broad.py) | Broad curriculum driver reusing the EWC numerical implementation; retains original historical-data behavior. |
| [baselines/_impl/l2.py](../baselines/_impl/l2.py) | SFT and fixed-reference L2-SP update, narrow task selection and training loop; all parameters. |
| [baselines/_impl/l2_broad.py](../baselines/_impl/l2_broad.py) | Broad (117-task) curriculum driver reusing the L2/SFT numerical implementation. |
| [baselines/_impl/lora.py](../baselines/_impl/lora.py) | Narrow LoRA configuration, selected-task plan and standard supervised-training launch. |
| [baselines/_impl/lora_broad.py](../baselines/_impl/lora_broad.py) | Broad LoRA configuration and supervised-training launch. |
| [baselines/_impl/supervised.py](../baselines/_impl/supervised.py) | Original generic JAX supervised loop used by the LoRA baseline, including its optimizer/EMA behavior. |
| [baselines/libero_rl_env_worker.py](../baselines/libero_rl_env_worker.py) | Isolated LIBERO simulation process used by the GRPO trainer. |
| [baselines/merge_checkpoints_linear.py](../baselines/merge_checkpoints_linear.py) | RETAIN baseline: merges SFT and reference checkpoints with the original linear interpolation and copies assets. |
| [baselines/train.py](../baselines/train.py) | Shared CLI for SFT, all-parameter L2/EWC, LoRA and co-training; routes narrow/broad settings. |
| [baselines/train_cotrain_seq_libero.sh](../baselines/train_cotrain_seq_libero.sh) | Requested three-stage co-training launcher; tracks earlier tasks and chains checkpoints. |
| [baselines/train_grpo.py](../baselines/train_grpo.py) | Retained whole-model LoRA GRPO trainer: simulated rollouts, advantages, likelihoods and policy updates. |
| [baselines/train_grpo_lora_seq_libero.sh](../baselines/train_grpo_lora_seq_libero.sh) | Requested three-stage LoRA SFT/GRPO launcher, with optional evaluation. |
| [examples/libero/main.py](../examples/libero/main.py) | Retained LIBERO evaluation loop, simulator reset/rollout settings, task-language filtering and per-task results. |
| [examples/libero/requirements.txt](../examples/libero/requirements.txt) | Pinned simulator-environment dependencies, separate from the training environment. |
| [packages/openpi-client/pyproject.toml](../packages/openpi-client/pyproject.toml) | Package metadata and dependencies for the shared inference client. |
| [packages/openpi-client/src/openpi_client/__init__.py](../packages/openpi-client/src/openpi_client/__init__.py) | Python package initializer. |
| [packages/openpi-client/src/openpi_client/action_chunk_broker.py](../packages/openpi-client/src/openpi_client/action_chunk_broker.py) | Buffers policy action chunks and provides individual actions to a caller. |
| [packages/openpi-client/src/openpi_client/base_policy.py](../packages/openpi-client/src/openpi_client/base_policy.py) | Minimal inference policy interface shared by clients. |
| [packages/openpi-client/src/openpi_client/image_tools.py](../packages/openpi-client/src/openpi_client/image_tools.py) | Client-side image resizing and conversion. |
| [packages/openpi-client/src/openpi_client/msgpack_numpy.py](../packages/openpi-client/src/openpi_client/msgpack_numpy.py) | MessagePack serialization of NumPy arrays for policy RPC. |
| [packages/openpi-client/src/openpi_client/runtime/agent.py](../packages/openpi-client/src/openpi_client/runtime/agent.py) | Generic agent interface; no physical device implementation. |
| [packages/openpi-client/src/openpi_client/runtime/agents/policy_agent.py](../packages/openpi-client/src/openpi_client/runtime/agents/policy_agent.py) | Generic agent wrapping a policy; no physical device implementation. |
| [packages/openpi-client/src/openpi_client/runtime/environment.py](../packages/openpi-client/src/openpi_client/runtime/environment.py) | Generic environment interface used by the inherited client runtime. |
| [packages/openpi-client/src/openpi_client/runtime/runtime.py](../packages/openpi-client/src/openpi_client/runtime/runtime.py) | Generic environment-agent loop; no robot-specific control code. |
| [packages/openpi-client/src/openpi_client/runtime/subscriber.py](../packages/openpi-client/src/openpi_client/runtime/subscriber.py) | Generic runtime event subscriber interface. |
| [packages/openpi-client/src/openpi_client/websocket_client_policy.py](../packages/openpi-client/src/openpi_client/websocket_client_policy.py) | WebSocket client used by the LIBERO evaluator to request model actions. |
| [pyproject.toml](../pyproject.toml) | Research package metadata, pinned scientific dependencies, CUDA extra and workspace configuration. |
| [research/EXPERIMENTS.md](../research/EXPERIMENTS.md) | Narrow, broad and three-stage recipes, inherited defaults and task identities. |
| [research/FILES.md](../research/FILES.md) | This per-file guide to the cleaned research release. |
| [research/METHOD.md](../research/METHOD.md) | Exact preserved equations, approved importance lifecycle, checkpoint semantics and provenance. |
| [research/SETUP.md](../research/SETUP.md) | Training/simulator installation, model/data locations and evaluation commands. |
| [scripts/eval_pi05_libero40.sh](../scripts/eval_pi05_libero40.sh) | Compatibility entry point used by the retained co-training and GRPO sequence launchers. |
| [scripts/eval_summary_pi05_holdout.py](../scripts/eval_summary_pi05_holdout.py) | Reads raw per-task evaluation results and reports the selected pretraining/held-out split. |
| [scripts/evaluate.py](../scripts/evaluate.py) | Runs policy serving and LIBERO evaluation across seeds; groups retention/new-task scores by task language. |
| [scripts/serve_policy.py](../scripts/serve_policy.py) | Loads a configured checkpoint and serves its actions over WebSocket for LIBERO. |
| [scripts/setup_libero.sh](../scripts/setup_libero.sh) | Installs the pinned upstream LIBERO simulator into a separate environment; no simulator code is vendored. |
| [scripts/train.py](../scripts/train.py) | Unified command line for pretraining, importance export, single-stage SAMBAR, three-stage sequences and normalization statistics. |
| [scripts/verify_split_consistency.py](../scripts/verify_split_consistency.py) | Checks that evaluation task language matches the intended training split. |
| [src/openpi/__init__.py](../src/openpi/__init__.py) | Python package initializer. |
| [src/openpi/models/__init__.py](../src/openpi/models/__init__.py) | Python package initializer. |
| [src/openpi/models/gemma.py](../src/openpi/models/gemma.py) | Gemma language/action-expert transformer used by the JAX model. |
| [src/openpi/models/gemma_fast.py](../src/openpi/models/gemma_fast.py) | Inherited FAST-model Gemma support needed by the shared configuration imports; not the SAMBAR trainer. |
| [src/openpi/models/lora.py](../src/openpi/models/lora.py) | Low-rank adapter layers used by the LoRA baselines. |
| [src/openpi/models/model.py](../src/openpi/models/model.py) | Shared model/observation interfaces, preprocessing and parameter restoration. |
| [src/openpi/models/pi0.py](../src/openpi/models/pi0.py) | π0/π0.5 model, flow-matching loss, action sampling and retained GRPO noise/likelihood helpers. |
| [src/openpi/models/pi0_config.py](../src/openpi/models/pi0_config.py) | Builds π0.5 variants, inputs and LoRA freeze rules. |
| [src/openpi/models/pi0_fast.py](../src/openpi/models/pi0_fast.py) | Inherited FAST-model implementation supporting shared imports; SAMBAR configurations use π0.5. |
| [src/openpi/models/siglip.py](../src/openpi/models/siglip.py) | SigLIP vision encoder used by π0.5. |
| [src/openpi/models/tokenizer.py](../src/openpi/models/tokenizer.py) | Language/action tokenization and loading of tokenizer assets. |
| [src/openpi/models/utils/fsq_tokenizer.py](../src/openpi/models/utils/fsq_tokenizer.py) | Finite-scalar-quantization tokenizer support used by inherited tokenizer code. |
| [src/openpi/models/vit.py](../src/openpi/models/vit.py) | Vision-transformer support for the inherited model library. |
| [src/openpi/models_pytorch/gemma_pytorch.py](../src/openpi/models_pytorch/gemma_pytorch.py) | Inherited optional PyTorch gemma pytorch support for shared model/inference imports; SAMBAR training uses JAX. |
| [src/openpi/models_pytorch/pi0_pytorch.py](../src/openpi/models_pytorch/pi0_pytorch.py) | Inherited optional PyTorch pi0 pytorch support for shared model/inference imports; SAMBAR training uses JAX. |
| [src/openpi/models_pytorch/preprocessing_pytorch.py](../src/openpi/models_pytorch/preprocessing_pytorch.py) | Inherited optional PyTorch preprocessing pytorch support for shared model/inference imports; SAMBAR training uses JAX. |
| [src/openpi/models_pytorch/transformers_replace/models/gemma/configuration_gemma.py](../src/openpi/models_pytorch/transformers_replace/models/gemma/configuration_gemma.py) | Inherited optional PyTorch configuration gemma support for shared model/inference imports; SAMBAR training uses JAX. |
| [src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py](../src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py) | Inherited optional PyTorch modeling gemma support for shared model/inference imports; SAMBAR training uses JAX. |
| [src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py](../src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py) | Inherited optional PyTorch modeling paligemma support for shared model/inference imports; SAMBAR training uses JAX. |
| [src/openpi/models_pytorch/transformers_replace/models/siglip/check.py](../src/openpi/models_pytorch/transformers_replace/models/siglip/check.py) | Inherited optional PyTorch check support for shared model/inference imports; SAMBAR training uses JAX. |
| [src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py](../src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py) | Inherited optional PyTorch modeling siglip support for shared model/inference imports; SAMBAR training uses JAX. |
| [src/openpi/policies/libero_policy.py](../src/openpi/policies/libero_policy.py) | Converts LIBERO observations/actions to the model input/output conventions. |
| [src/openpi/policies/policy.py](../src/openpi/policies/policy.py) | Inference policy abstraction and sampling/transform application. |
| [src/openpi/policies/policy_config.py](../src/openpi/policies/policy_config.py) | Constructs a policy from a training configuration, checkpoint and normalization assets. |
| [src/openpi/py.typed](../src/openpi/py.typed) | Marks the model package as providing typing information. |
| [src/openpi/serving/websocket_policy_server.py](../src/openpi/serving/websocket_policy_server.py) | WebSocket observation/action service and health endpoint. |
| [src/openpi/shared/__init__.py](../src/openpi/shared/__init__.py) | Python package initializer. |
| [src/openpi/shared/array_typing.py](../src/openpi/shared/array_typing.py) | Array and pytree type annotations and runtime type/shape checks. |
| [src/openpi/shared/download.py](../src/openpi/shared/download.py) | Downloads model assets, handles cache paths and locks. |
| [src/openpi/shared/image_tools.py](../src/openpi/shared/image_tools.py) | Image resizing/padding utilities shared by preprocessing. |
| [src/openpi/shared/nnx_utils.py](../src/openpi/shared/nnx_utils.py) | NNX state mapping, regex filters and JIT helpers used by model/adapters. |
| [src/openpi/shared/normalize.py](../src/openpi/shared/normalize.py) | Running statistics and normalization JSON I/O. |
| [src/openpi/training/checkpoints.py](../src/openpi/training/checkpoints.py) | Saves/restores optimizer, model, normalization assets and optional SAMBAR method arrays. |
| [src/openpi/training/config.py](../src/openpi/training/config.py) | LIBERO-only model/training configurations and data transforms; hardware configs removed and asset paths relocated. |
| [src/openpi/training/curriculum.py](../src/openpi/training/curriculum.py) | 32-task and 27+90-task data selection, natural frame mixing, task-only dataset routing and normalization computation. |
| [src/openpi/training/data_loader.py](../src/openpi/training/data_loader.py) | LeRobot task filtering, data transforms, batching and sharded device transfer. |
| [src/openpi/training/importance.py](../src/openpi/training/importance.py) | Portable Orbax importance arrays plus task identities/counts; validates shape/structure and rejects repeated tasks. |
| [src/openpi/training/optimizer.py](../src/openpi/training/optimizer.py) | Inherited learning-rate schedules and optimizer construction. |
| [src/openpi/training/runner.py](../src/openpi/training/runner.py) | Shared experiment orchestration, task resolution, saved-importance loading, stage chaining and method-state persistence. |
| [src/openpi/training/sambar.py](../src/openpi/training/sambar.py) | Preserved weighted_taco_ar.py numerical operations: model initialization, reference, minibatch Fisher estimator, training and dual update. |
| [src/openpi/training/sambar_math.py](../src/openpi/training/sambar_math.py) | Selected epsilon-in-both-parts anchor, augmented penalty, dual norm and approved task-count history average. |
| [src/openpi/training/sharding.py](../src/openpi/training/sharding.py) | Device mesh, FSDP parameter sharding and mesh-context helpers. |
| [src/openpi/training/task_split.py](../src/openpi/training/task_split.py) | Seeded LIBERO-40 split and task-index-to-language mapping shared by training and evaluation. |
| [src/openpi/training/utils.py](../src/openpi/training/utils.py) | TrainState container and parameter-tree diagnostics; optional method-state field supports SAMBAR restore. |
| [src/openpi/training/weight_loaders.py](../src/openpi/training/weight_loaders.py) | Base/checkpoint weight loading and model-tree validation, including adapter-compatible initialization. |
| [src/openpi/transforms.py](../src/openpi/transforms.py) | Observation/action repacking, normalization, tokenization and transformation composition. |
| [tests/test_curricula.py](../tests/test_curricula.py) | Tests 32/117-task definitions, explicit current-only data routing, baseline CLI scope and language-based evaluation. |
| [tests/test_sambar.py](../tests/test_sambar.py) | CPU regression tests for estimator reduction order, epsilon anchor, cached history, serialization, stage execution and restore. |
| [uv.lock](../uv.lock) | Resolved dependency lock derived from the source lock; excluded hardware/RLDS dependency groups removed. |
