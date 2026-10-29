# Method and provenance

## Preserved calculations

The numerical source is `scripts/train_pi05_weighted_taco_ar.py` from `aayushis196/pi05_taco` at commit `8245c09fe15ed542a2aa4740076d8ca0b417bc9b`. Its all-parameter branch is extracted into `src/openpi/training/sambar.py` and `sambar_math.py`.

For minibatch b, let L_b be `mean(model.compute_loss(..., train=True))`. The estimate for parameter j is

\[
\Omega_j = \frac{1}{B}\sum_{b=1}^{B}\left(\frac{\partial L_b}{\partial\theta_j}\right)^2.
\]

The inner mean is **inside the gradient and the square**. This is the original minibatch-gradient-square estimator, often called a diagonal Fisher proxy in the source. It is not a mean of per-example squared gradients. The underlying π0.5 flow-matching loss and noise/time sampling are unchanged. The estimator samples 50 minibatches by default, with the original RNG folding and training-mode preprocessing.

The anchor is exactly the selected AR expression:

\[
z_j = \frac{\Omega_{old,j}\theta_{ref,j}+(\beta\Omega_{new,j}+\epsilon)\theta_j}
{\Omega_{old,j}+\beta\Omega_{new,j}+\epsilon}.
\]

The objective adds `sum(u * (theta - z)) + rho/2 * sum((theta-z)**2)`. The dual update is `u += rho * (theta-z)`. The source optimizer update, RNG folding, warmup gate, Fisher refresh timing, consensus/dual order and metric averaging are retained.

Inherited method defaults: rho=0.02, beta=1, epsilon=1e-8, Fisher batches=50, Fisher refresh=0, dual period=1, warmup=0. With refresh=0, the current-task importance used during training stays at its initial estimate. The separate estimate after stage completion is used for history in the next stage.

## Approved importance lifecycle change

1. After 32-task or 117-task LIBERO pretraining, estimate importance at the final model using that pretraining data, and save it.
2. At a new SAMBAR stage, load the previous model and saved historical importance. Compute initial new-task importance using only current-task data, as in the selected trainer.
3. Train with the preserved AR calculations.
4. Estimate the newly completed task's importance at its final weights using the same estimator. If history covers N tasks and the new stage covers K tasks, save `(N * old + K * current)/(N + K)` with N+K task identities.

This is a task-count-weighted average of saved stage estimates. The original pretraining estimate retains the original pooled frame sampler; the 117-task mixture uses natural frame-count proportions. The running average is not a new equal-per-task Fisher estimator, nor a Fisher recomputed over all historical demonstrations at the latest parameters. Estimates from different stages were evaluated at different checkpoints.

**Assumption:** original π0.5 pretraining importance is unavailable. A 32/117-task artifact measures the subsequent LIBERO pretraining only. Missing importance is an error, not silently replaced by zero or an invented matrix.

A complete final checkpoint has `params/`, `train_state/`, `assets/` and `importance/`. The importance folder contains Orbax arrays and `metadata.json` (task identities/count, curriculum, estimator and provenance). Checkpoints additionally store the SAMBAR reference, old/new importance, consensus and dual arrays. Restore retains those variables and optimizer state. The inherited data loader does not save its iterator position, so resume is not promised to reproduce an uninterrupted minibatch sequence bit for bit.

The default public dataset contains multiple tasks. Its inherited loader opens the dataset before selecting frames. To operate with **only current-task demonstrations on disk**, provide a valid current-task-only LeRobot dataset through `--finetune-repo-id` and `--finetune-root`, with `--finetune-tasks` specifying exact language strings. No historical demonstration loader is constructed by SAMBAR. This input override changes the data location, not the loss or estimator.

## Scope of the refactor

The SAMBAR training modes share one runner. Baselines share a dispatcher but retain their original training loops where consolidation would change behavior. L2/EWC use all parameters; LoRA trains its adapters. EWC retains its source historical-demo Fisher computation, and co-training deliberately replays old demonstrations. The SAMBAR cached-history change is not silently applied to either baseline.

Normalization files were relocated without changing their contents. Model, transforms, optimizer and serving utilities come from the same source commit; the Apache-2.0 license is retained. Unused real-robot policies, RLDS hardware loaders and hardware examples are excluded.

LIBERO revision `8f1084e3132a39270c3a13ebe37270a43ece2a01` matches the source repository's simulator/environment/assets files. The vendored differences were download/setup documentation and an extra package initializer, so those files do not require a local simulator fork. The repository installs the upstream simulator instead of shipping its assets or duplicate build tree.

## Validation

Run `uv run pytest`. Validation passed 12 CPU tests, 14 CLI help checks, Python syntax checks for all 78 retained Python files, and shell syntax checks for all four launchers. The package wheel builds, the training lockfile is current, and the simulator requirements resolve for Linux/Python 3.11. The small CPU tests check minibatch gradient cancellation, the selected epsilon behavior, 32/117-task running averages, importance shape validation and serialization, method/optimizer checkpoint recovery, task splits and evaluation aggregation.

During refactoring, extracted functions from the original source were compared directly against the new functions on a small NNX model. Fisher step, Fisher averaging, anchor, optimizer step and dual update produced exactly equal CPU arrays. The all-parameter L2 and EWC training steps also matched their original implementations exactly. This comparison does not replace full-model GPU training, multi-device memory testing or simulator evaluation.
