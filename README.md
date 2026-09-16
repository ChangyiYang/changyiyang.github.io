# changyiyang.github.io

Changyi Yang 的 decks & proposals 总站。

👉 <https://changyiyang.github.io/>

## 目录
- [`sglang-overlap/`](https://changyiyang.github.io/sglang-overlap/) — SGLang SBO / TBO 源码长文：6 张架构与时序图、ping-pong 调度、当前版本支持边界
- [`minicpm-routing-audit/`](https://changyiyang.github.io/minicpm-routing-audit/) — MiniCPM-o 4.5 实时升级路由器：27 轮 probe-only 实验、严格因果读出与失败边界（18 slides）
- [`weekly-progress-2026-08-31/`](https://changyiyang.github.io/weekly-progress-2026-08-31/) — concise English weekly engineering update: six measurable moves across B300, AMD, DSpark, data, parity, and E2E evaluation
- [`lfm2-exact-speed/`](https://changyiyang.github.io/lfm2-exact-speed/) — LFM2 24B MoE exact SGLang rollout 性能优化：术语、逐步改动、收益与未保留实验（41 slides）
- [`lfm2-logits-alignment/`](https://changyiyang.github.io/lfm2-logits-alignment/) — LFM2 24B MoE 训推 logits exact alignment：术语、完整定位与未采用方案（64 slides）
- [`dspark-draft-model/`](https://changyiyang.github.io/dspark-draft-model/) — DSpark / DFlash：面向 agentic 流量的 in-domain draft model（10 slides）
- [`interrupt-benchmark-survey/`](https://changyiyang.github.io/interrupt-benchmark-survey/) — 模型主动打断：Instruct-FD 与三个 Benchmark（18 slides）
- [`frontier-attention/`](https://changyiyang.github.io/frontier-attention/) — Kimi、Qwen、DeepSeek、GLM 四条新一代 attention 路线与推理侧取舍

## 新增一份 deck
把 HTML 放进一个新子目录即可，会自动出现在对应 URL：

```
changyiyang.github.io/
├── index.html            # 总目录页（新增 deck 时在这里加一张卡片）
└── <deck-name>/
    └── index.html        # → changyiyang.github.io/<deck-name>/
```
