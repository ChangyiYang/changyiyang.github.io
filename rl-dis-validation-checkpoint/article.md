# 训推分离之后，validation 和 checkpoint 还会卡住训练吗？

verl、slime、Miles 源码对照 · 2026-10-29

**会。把 trainer 和 rollout 放到不同 GPU 上，并不会自动把 validation 和 checkpoint 移出训练的关键路径。** 当前三套框架的常规评测都要等结果返回，下一次参数更新才能继续。Miles 已经有独立的快照评测路径，可以让评测推理和训练重叠，但快照导出、积压反压和退出收尾仍有等待。checkpoint 则取决于后端：Megatron 可以异步写 shard，保存前的状态准备和需要确认完成的边界仍然要停。

这里把 dis 理解为 disaggregated，也就是训推分离。讨论的“阻塞主 run”指 trainer 无法进入下一次训练更新；Ray 的其他 actor、后台写盘进程、rollout 请求是否还在运行，是另一件事。

本文是调用链审计，没有跑 GPU benchmark。等待点可以从源码确认，停多少秒要在具体模型、并行拓扑和存储上测。下文的“保存完成”采用框架自身的完成协议，不额外声称文件系统已经提供断电持久性。

## 先看结论

| 路径 | Validation 会挡下一次训练更新吗？ | Checkpoint 落盘会挡吗？ |
|---|---|---|
| verl V1 `separate_async` | 会。主循环直接调用 `_validate()` | 默认会；Megatron `async_save` 可把部分 shard 写盘放到后台，调用本身仍有同步工作 |
| verl experimental fully async | 会。trainer `await` rollouter 的评测 RPC | 保存函数仍在训练步骤里；后端决定写盘能否异步，不能套用 V1 的全部完成协议 |
| slime `train.py`，含 fully async rollout | 会。显式 `ray.get(eval.remote(...))` | 默认同步；Megatron 开异步后可重叠部分写盘，straw / recovery 等边界会强制等完成 |
| Miles 默认共享 rollout 引擎评测 | 会。dispatcher 直接 `await eval()` | Megatron 可异步写 shard；默认关闭；HF 导出仍是额外工作 |
| Miles 独立 eval fleet / checkpoint eval backend | 不必等本次评测推理结束；快照导出、队列反压会等 | 评测快照和训练恢复 checkpoint 是两条链路；FSDP 保存仍同步 |

这个表概括的是下面逐条引用的入口和后端。自定义 eval 函数、保存 hook 或独立外部服务可以改变行为，不能只凭框架名给所有部署下结论。

## verl：当前默认 V1，评测仍在训练循环里

verl 变化比较快。此次检查的配置中，`trainer.use_v1: true` 已经是默认值，V1 支持 `sync`、`colocate_async`、`separate_async`。主入口按配置选择 trainer；旧 experimental fully async 不能代表整个当前实现。[配置与入口][v-config] [入口选择][v-main]

### V1 separate_async：异步采样没有带来异步评测

V1 的基类 `PPOTrainer.fit()` 把这几个操作顺序写在同一个循环里：

```text
step：采样、训练更新
  → 到保存点：_save_checkpoint()
  → on_step_end()
  → 到评测点：on_validate_begin() → _validate() → on_validate_end()
  → 记录指标、清理、global_steps += 1
```

`_validate()` 没有作为独立后台任务提交。它遍历验证集，把请求送到 agent loop，再从 replay buffer 的 `val` partition 收齐结果、统计分数。直到函数返回，循环才推进。`separate_async` 的 `on_validate_begin()` 还会在 hybrid engine 当前处于 trainer 模式时将其切回 rollout 模式。[V1 主循环][v-loop] [验证实现][v-val] [模式切换][v-switch]

因此，评测期间独立 rollout GPU 上可能还有其他请求在跑，但 trainer 不会继续做下一次 update。若启用了 hybrid GPU，还要加上模式切换的代价。纯分离配置是否切换，取决于当时的 engine mode。

### experimental fully async：await 远端评测，trainer 仍要等

旧路径也值得核对，因为现有脚本可能仍在用。`FullyAsyncTrainer.fit_step()` 依次执行 `await self._fit_validate()` 和 `self._fit_save_checkpoint()`。普通验证分支是：

```python
val_metrics = await self.rollouter.do_validate.remote()
```

rollouter 的 `do_validate()` 内部调用 `_validate()` 后才返回。这让 RPC 等待期间的事件循环有机会处理其他任务，却不会让当前训练协程自动跨过这个等待点。[训练步骤][v-fa-step] [验证分支][v-fa-val] [rollouter][v-fa-roll]

`use_trainer_do_validate` 更明确：先把 trainer GPU 切成 rollout replicas，跑验证，再切回训练。它借用训练资源加速验证，trainer 在这段时间里无法更新参数。[trainer-side validation][v-fa-hybrid]

### checkpoint：FSDP 同步，Megatron 只能异步掉其中一部分

V1 `_save_checkpoint()` 在主线程中调用 actor / critic 保存，接着保存 dataloader。异步模式还会保存 TransferQueue，让已取出但尚未训练的 prompt 能在重启后恢复：完成的 trajectory 恢复，未完成的请求重新发出。这里只备份模型权重是不够的。[V1 保存入口][v-save]

FSDP 的保存非常直接：各 rank 构造 CPU shard，用 `torch.save` 写模型、optimizer 和 extra state，前后都有 `torch.distributed.barrier()`。选了 HF 权重导出还会多做一次导出和等待。这个实现没有因为通用配置里存在 `async_save` 就变成异步。[FSDP writer][v-fsdp]

Megatron 的 `async_save` 默认是 `False`。开启后，manager 可以把 model、optimizer、extra 的分布式保存请求加入 AsyncCallsQueue，最后一个请求挂 finalize callback。状态字典构造、必要的通信、HF config 和通过 bridge 导出 HF 权重仍在调用链上；Megatron-FSDP 的专用 writer 也明确采用同步保存。[默认配置][v-async-default] [Megatron 保存][v-mcore] [异步调度][v-finalize-dispatch]

完成协议也要读清：manager 在所有 per-content 写入完成后写 `ckpt_contents.json`，然后处理 HDFS 上传、latest tracker 和 retention。V1 看到 actor 的 `async_save` 开启，就跳过自己立即写 latest tracker 的动作，把模型完成标记交给后端。HDFS copy 发生在 finalize 函数里，不能把“后台写 shard”推导成“上传也永远不占主流程”。manifest 表示本地内容写完，也不表示远端上传已经完成。[finalize][v-finalize] [V1 tracker 分支][v-save]

experimental fully async 的 `_save_checkpoint()` 是另一份实现：它保存 actor / critic，`ray.get` rollouter 的状态保存后直接写 latest tracker。这里没有 V1 那个 async-save 分支。若要在这个旧入口启用异步保存，需要继续核验所用 worker 的 finalize 驱动和恢复选择逻辑；单看这份函数，无法证明 latest tracker 与后台模型写完严格对齐。[旧路径保存][v-fa-save]

## slime：fully async 的是数据生产，主循环仍然等 eval

当前 slime 的 fully async 示例使用标准 `python3 train.py`，通过 `--rollout-function-path` 换成持续生产数据的实现。它没有另起一份“评测自动后台化”的 trainer。[fully async 使用说明][s-async-doc]

`train.py` 在训练前评测和周期评测两处都调用：

```python
ray.get(rollout_manager.eval.remote(rollout_id))
```

周期评测位于权重同步之后；`RolloutManager.eval()` 调用评测 rollout 函数，处理结果并记录指标后才返回。下一轮训练要等这条 RPC 完成。默认评测共享 rollout engines；持续生产 worker 的存在不改变 trainer 的等待关系。[训练主循环][s-loop] [eval 实现][s-eval]

共享引擎上是否仍有在途训练请求、如何分配并发，还取决于 rollout 实现。这里不把 trainer 等待夸大成“所有 GPU 都空闲”，也不把后台还在生成等同于“主 run 没有停”。

### 异步 checkpoint 会在恢复边界重新同步

保存调用先经过 `RayTrainGroup.save_model()`，它对所有训练 rank 的保存 RPC 做 `ray.get`。因此，即使开了异步写盘，driver 也要等各 rank 把保存调用执行到返回。[actor group][s-group]

Megatron actor 的 `save_model()` 有几个明确的等待点：保存前对上一份 async save 调 `maybe_finalize_async_save(blocking=True)`；提交当前保存；如果 `force_sync`，再以 blocking 模式等当前保存完成；随后可能额外导出 HF 权重。offload 配置还会加入 wake / sleep。[actor 保存][s-actor]

`force_sync` 由更上层的联合保存逻辑决定。当前代码在以下情况强制等待：

- straw checkpoint：满足代码里的 straw、actor 正在训练、非 debug 等条件；
- training recovery 开启；
- `release_train`，或最后一个 rollout。

模型、optimizer 和 rollout 数据进度必须形成同一个可恢复边界。straw 分支在模型和 rollout 保存成功后才发布 joint commit；recovery 通知也放在最后。否则训练状态还没写完，消费过的数据就可能被当作已安全保存而释放。[联合 checkpoint 协议][s-joint]

所以 slime 的普通周期保存，在不触发这些强制条件且 Megatron 开启 async save 时，可以让本次 shard 写盘与后续训练重叠。开启 straw / recovery 后，保存边界主动收回了这部分重叠。这里的停顿有明确的一致性目的。

## Miles：独立评测已经有了，但默认仍然阻塞

Miles 要按 eval backend 分开看。`EvalDispatcher.dispatch()` 的第一个分支就决定了主行为：没有启用 snapshot eval，直接 `await rollout_manager.eval(rollout_id)`，执行结束才返回。[dispatcher][m-dispatch]

fully async 的共享引擎评测还会清除 producer 的 resumed event，暂停提交新的生产请求；验证结束后恢复。已有在途请求可以继续完成，但这不消除训练 driver 对评测的等待。[共享引擎 eval][m-shared]

### 独立 fleet / 外部 checkpoint backend 可以后台评测

设置 `--eval-num-gpus N` 选择独立 fleet，或者把 `--eval-function-path` 指向 `CheckpointEvalFn`，会启用 snapshot eval。两者互斥，独立 fleet 的文档入口是 `train_async.py`。默认 `eval_num_gpus=0` 的共享引擎路径不会自动切过来。[backend 选择][m-select] [参数说明][m-cli]

独立路径中，dispatcher 获取某个 rollout 对应的 HF snapshot，再用 `asyncio.ensure_future(...)` 提交评测任务。训练循环只等 dispatch 完成，不必等这次评测推理完成。snapshot 路径还带完整性检查，结果按提交时的 rollout ID 记录，并提供 `eval/lag_steps`。[dispatcher][m-dispatch] [checkpoint eval][m-checkpoint-eval]

```text
trainer： update(k) → 导出 snapshot(k) ──→ update(k+1) → update(k+2)
eval：                               └→ 加载 snapshot(k) → 推理、打分 → 结果记在 k
                                    ↑ 后台评测从这里开始
```

这条路径仍有三处可能让 run 等待。

第一处是快照导出。使用 `--eval-hf-dir` 时，`dispatch()` 会 `await self._export(...)`，内部等待 actor 的 collective `export_hf`。把目录放在 `/dev/shm` 可以避开磁盘 I/O，仍要付权重转换、通信和内存拷贝的代价。若复用周期 `--save-hf`，就不再为评测单独导出，但原来的 HF 保存成本还在；文档要求这种复用模式的 `eval_interval` 是 `save_interval` 的整数倍。[导出调用][m-dispatch] [collective export][m-export] [快照来源约束][m-snapshot-doc]

第二处是积压。`--eval-max-in-flight` 默认 2，`--eval-overflow-policy` 默认 `backpressure`：到新评测点时，如果 pending 已满，就等最老的任务。选择 `skip` 可以跳过新点，不过最后一个 rollout 会传入 `force=True`，仍可能等待。增大 pending 上限只允许多积压快照；后端的 `_eval_lock` 依然串行处理 snapshot eval。[默认值][m-eval-config] [队列逻辑][m-dispatch] [后端锁][m-checkpoint-eval]

第三处是退出。driver 把 `eval_dispatcher.drain` 注册为 disposer，收尾时会等 pending 评测完成。因此“训练更新没有等待每次 eval”与“整个进程可以随时退出”也要分开。[async driver][m-driver] [drain][m-dispatch]

### 训练 checkpoint 的异步能力仍由后端决定

`train_async.py` 在保存点 `await model.save_model(...)`，随后保存 rollout 状态。外部 sentinel 触发和最后一个 rollout 会传 `force_sync=True`。这条 await 保证保存 RPC 返回；是否还存在后台写盘，由 actor 的实现决定。[async driver][m-driver]

Megatron actor 在提交新保存前等待前一份异步保存；当前保存若 `force_sync` 就再等一次。配置了 post-save hook 时，rank 0 也会先 finalize 再调用 hook。`--async-save` 默认关闭；开启后 native checkpoint 写入可以和后续 HF 导出重叠，但 HF 导出本身没有因此消失。[Megatron actor][m-actor] [保存参数][m-cli]

FSDP 的 torch-native actor 则直接拒绝 `backend.async_save`。其 writer 先 synchronize，再同步 `dcp.save` model、optimizer、scheduler，保存 RNG 和 metadata，最后 barrier。[torch-native actor][m-native] [FSDP writer][m-fsdp]

评测 HF snapshot 通常只有推理需要的权重；恢复训练还需要 optimizer、scheduler、RNG，以及相应的数据进度。它们可以共享一次权重导出的产物，但不能把 eval snapshot 当作完整的训练 checkpoint。

## 放到实际 run 里怎么判断

如果目标是长评测集不拖慢参数更新，当前 Miles 的独立 fleet / checkpoint backend 已经提供了明确的后台调度路径。需要同时考虑快照导出时间、存储容量和满队列策略。`skip` 牺牲评测点完整性换取较稳定的训练节奏；`backpressure` 保留评测点，但慢评测会反压训练。这是部署选择，不能只开一个“async”就忽略。

verl V1 和 slime 的上述入口中，validation 都还在主循环上。若想把完整评测外置，需要增加不可变权重快照、版本标记、结果归属和积压处理。直接删除 `ray.get` 或 `await`，可能让评测跑在持续更新的模型上，或者把资源冲突和异常处理留给后台。

checkpoint 的调优要先拆时间。建议分别记录保存调用返回耗时、后台 writer 完成耗时、远端上传耗时，以及最后一个可恢复版本落后当前训练多少步。独立 rollout 在保存期间还能继续产数据，也要看队列是否满、训练恢复后是否立刻有可用 batch。只看 GPU utilization，很容易把后台忙碌误读成训练没有停。

如果要求 straw / recovery 那样严格的数据与模型共同恢复边界，就要接受 commit 前的等待，或者另行设计能证明正确的异步提交协议。把等待删掉并不会让状态自动变得一致。

## 复查用的版本

以下均为调研时拉取的上游 `main`，链接固定到 commit。结论覆盖本文列出的入口与 backend，不代表所有插件或未来版本。

| 仓库 | Commit |
|---|---|
| volcengine/verl | `a08ea02a784d7d4982b1d81df36d188a476192bb` |
| THUDM/slime | `540ea919e45b8b929198abce008a103be9cfc779` |
| radixark/miles | `ae6bea66e8497391c73e9eb1bbe1a9eaeb9df1d4` |

主要证据是训练循环的等待关系、保存 writer 和完成标记；没有据此估算吞吐收益或暂停秒数。

[v-config]: https://github.com/volcengine/verl/blob/a08ea02a784d7d4982b1d81df36d188a476192bb/verl/trainer/config/ppo_trainer.yaml#L232-L261
[v-main]: https://github.com/volcengine/verl/blob/a08ea02a784d7d4982b1d81df36d188a476192bb/verl/trainer/main_ppo.py#L122-L194
[v-loop]: https://github.com/volcengine/verl/blob/a08ea02a784d7d4982b1d81df36d188a476192bb/verl/trainer/ppo/v1/trainer_base.py#L413-L532
[v-val]: https://github.com/volcengine/verl/blob/a08ea02a784d7d4982b1d81df36d188a476192bb/verl/trainer/ppo/v1/trainer_base.py#L1032-L1204
[v-switch]: https://github.com/volcengine/verl/blob/a08ea02a784d7d4982b1d81df36d188a476192bb/verl/trainer/ppo/v1/trainer_separate_async.py#L247-L250
[v-save]: https://github.com/volcengine/verl/blob/a08ea02a784d7d4982b1d81df36d188a476192bb/verl/trainer/ppo/v1/trainer_base.py#L949-L1030
[v-fa-step]: https://github.com/volcengine/verl/blob/a08ea02a784d7d4982b1d81df36d188a476192bb/verl/experimental/fully_async_policy/fully_async_trainer.py#L536-L600
[v-fa-val]: https://github.com/volcengine/verl/blob/a08ea02a784d7d4982b1d81df36d188a476192bb/verl/experimental/fully_async_policy/fully_async_trainer.py#L812-L830
[v-fa-roll]: https://github.com/volcengine/verl/blob/a08ea02a784d7d4982b1d81df36d188a476192bb/verl/experimental/fully_async_policy/fully_async_rollouter.py#L648-L653
[v-fa-hybrid]: https://github.com/volcengine/verl/blob/a08ea02a784d7d4982b1d81df36d188a476192bb/verl/experimental/fully_async_policy/fully_async_trainer.py#L832-L871
[v-fa-save]: https://github.com/volcengine/verl/blob/a08ea02a784d7d4982b1d81df36d188a476192bb/verl/experimental/fully_async_policy/fully_async_trainer.py#L917-L978
[v-fsdp]: https://github.com/volcengine/verl/blob/a08ea02a784d7d4982b1d81df36d188a476192bb/verl/utils/checkpoint/fsdp_checkpoint_manager.py#L360-L523
[v-async-default]: https://github.com/volcengine/verl/blob/a08ea02a784d7d4982b1d81df36d188a476192bb/verl/trainer/config/actor/actor.yaml#L134-L142
[v-mcore]: https://github.com/volcengine/verl/blob/a08ea02a784d7d4982b1d81df36d188a476192bb/verl/utils/checkpoint/megatron_checkpoint_manager.py#L1151-L1281
[v-finalize-dispatch]: https://github.com/volcengine/verl/blob/a08ea02a784d7d4982b1d81df36d188a476192bb/verl/utils/checkpoint/megatron_checkpoint_manager.py#L1128-L1149
[v-finalize]: https://github.com/volcengine/verl/blob/a08ea02a784d7d4982b1d81df36d188a476192bb/verl/utils/checkpoint/megatron_checkpoint_manager.py#L1081-L1126
[s-async-doc]: https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/examples/fully_async/README.md#L43-L80
[s-loop]: https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/train.py#L13-L99
[s-eval]: https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/ray/rollout.py#L392-L412
[s-group]: https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/ray/actor_group.py#L168-L176
[s-actor]: https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/backends/megatron_utils/actor.py#L601-L626
[s-joint]: https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/data/checkpoint.py#L39-L72
[m-dispatch]: https://github.com/radixark/miles/blob/ae6bea66e8497391c73e9eb1bbe1a9eaeb9df1d4/miles/ray/rollout/eval_dispatch.py#L11-L101
[m-shared]: https://github.com/radixark/miles/blob/ae6bea66e8497391c73e9eb1bbe1a9eaeb9df1d4/miles/rollout/fully_async_rollout.py#L123-L143
[m-select]: https://github.com/radixark/miles/blob/ae6bea66e8497391c73e9eb1bbe1a9eaeb9df1d4/miles/utils/arguments.py#L151-L165
[m-cli]: https://github.com/radixark/miles/blob/ae6bea66e8497391c73e9eb1bbe1a9eaeb9df1d4/docs/user-guide/cli-reference.md#L156-L214
[m-checkpoint-eval]: https://github.com/radixark/miles/blob/ae6bea66e8497391c73e9eb1bbe1a9eaeb9df1d4/miles/ray/rollout/rollout_executor.py#L336-L383
[m-export]: https://github.com/radixark/miles/blob/ae6bea66e8497391c73e9eb1bbe1a9eaeb9df1d4/miles/backends/megatron_utils/actor.py#L954-L968
[m-eval-config]: https://github.com/radixark/miles/blob/ae6bea66e8497391c73e9eb1bbe1a9eaeb9df1d4/miles/utils/args/configs/eval.py#L55-L76
[m-driver]: https://github.com/radixark/miles/blob/ae6bea66e8497391c73e9eb1bbe1a9eaeb9df1d4/train_async.py#L74-L171
[m-actor]: https://github.com/radixark/miles/blob/ae6bea66e8497391c73e9eb1bbe1a9eaeb9df1d4/miles/backends/megatron_utils/actor.py#L913-L952
[m-native]: https://github.com/radixark/miles/blob/ae6bea66e8497391c73e9eb1bbe1a9eaeb9df1d4/miles/backends/training_utils/torch_native/actor.py#L117-L121
[m-fsdp]: https://github.com/radixark/miles/blob/ae6bea66e8497391c73e9eb1bbe1a9eaeb9df1d4/miles/backends/fsdp_utils/checkpoint.py#L204-L268
[m-snapshot-doc]: https://github.com/radixark/miles/blob/ae6bea66e8497391c73e9eb1bbe1a9eaeb9df1d4/docs/user-guide/fully-async.md#L300-L329
