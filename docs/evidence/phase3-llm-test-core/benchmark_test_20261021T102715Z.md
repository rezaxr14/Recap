# RECAP benchmark - suite `v2` split `test`

Generated: 2026-10-21T10:46:26.376273+00:00
Dataset hash: `sha256:fce743bc8176368fbd5731f5da1683690ad0e286bca59523b115a745aa6c1a21`
Agent: real LLM `gemma-4-e4b-uncensored-hauhaucs-aggressive` via `openai_compatible`
Prompt version: `llm-tool-agent-v1` | temperature: 0.0 | seed: 1 | tool budget: 6

> Deterministic mock environment with a real configured chat model. Model behavior is measured, not assumed; provider failures are recorded as failed runs.

## Summary

| baseline | success | 95% CI | recall@3 | MRR | nDCG@3 | tool calls | harmful transfer | incompatible selection | hard-neg selected | hard-neg outrank | false abstention |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 1.000 | [1.000, 1.000] | 1.000 | 0.750 | 0.815 | 1.86 | 0.000 | 1.000 | 1.000 | 0.500 | 0.000 |
| hybrid_rrf_version_filter | 1.000 | [1.000, 1.000] | 1.000 | 1.000 | 1.000 | 1.50 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| no_memory | 1.000 | [1.000, 1.000] | 0.000 | 0.000 | 0.000 | 1.50 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| outcome_aware | 1.000 | [1.000, 1.000] | 1.000 | 0.750 | 0.815 | 1.50 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

`incompatible selection` is an independent safety audit: selected memories are re-evaluated after the fact against the task's schema and preconditions. It never influences selection, so it cannot flatter any baseline.

## Per-family metrics

| family | baseline | success | recall@3 | MRR | harmful transfer | incompatible selection | hard-neg outrank | false abstention |
|---|---|---|---|---|---|---|---|---|
| date_format | bm25 | 1.000 | 1.000 | 0.750 | 0.000 | 1.000 | 0.500 | 0.000 |
| date_format | hybrid_rrf_version_filter | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| date_format | no_memory | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| date_format | outcome_aware | 1.000 | 1.000 | 0.750 | 0.000 | 0.000 | 0.000 | 0.000 |
| enum_change | bm25 | 1.000 | 1.000 | 0.750 | 0.000 | 1.000 | 0.500 | 0.000 |
| enum_change | hybrid_rrf_version_filter | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| enum_change | no_memory | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| enum_change | outcome_aware | 1.000 | 1.000 | 0.750 | 0.000 | 0.000 | 0.000 | 0.000 |
| required_parameter | bm25 | 1.000 | 1.000 | 0.750 | 0.000 | 1.000 | 0.500 | 0.000 |
| required_parameter | hybrid_rrf_version_filter | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| required_parameter | no_memory | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| required_parameter | outcome_aware | 1.000 | 1.000 | 0.750 | 0.000 | 0.000 | 0.000 | 0.000 |

## Per-task results

| task | baseline | run | success | status | error |
|---|---|---|---|---|---|
| required_parameter-v1-01 | no_memory | `run_ef2cf24a22844526a711f69356f90824` | True | not_retrieved |  |
| required_parameter-v1-02 | no_memory | `run_84607b61f9fa4ee498c0d6ab074c5085` | True | not_retrieved |  |
| required_parameter-v1-03 | no_memory | `run_6dd3ace568bf41dfb2c18011c1e1f4fb` | True | not_retrieved |  |
| required_parameter-v1-04 | no_memory | `run_b6f420d7b5e4439ca10214ea3f3b9e27` | True | not_retrieved |  |
| required_parameter-v1-05 | no_memory | `run_4000ff5e06614043a1a2dbf73eb33db4` | True | not_retrieved |  |
| required_parameter-v1-06 | no_memory | `run_ad181987d81d48d7b0bb0c3820c6c3f9` | True | not_retrieved |  |
| required_parameter-v2-01 | no_memory | `run_2e93d66eb32841faa1f741ed5331e071` | True | not_retrieved |  |
| required_parameter-v2-02 | no_memory | `run_10da85fb484b479c88833158be3d09cd` | True | not_retrieved |  |
| required_parameter-v2-03 | no_memory | `run_e49cfe2d1fcd4a1a987179d4f0ed15e8` | True | not_retrieved |  |
| required_parameter-v2-04 | no_memory | `run_7196b193a94441ca9d42b8002ec8bc7b` | True | not_retrieved |  |
| required_parameter-v2-05 | no_memory | `run_605a1354dd054231899ef00891c10927` | True | not_retrieved |  |
| required_parameter-v2-06 | no_memory | `run_ddfba6f3102949c8b527ad3379174b23` | True | not_retrieved |  |
| enum_change-v1-01 | no_memory | `run_611864e3fb8e495487c66a8b0bf5cf82` | True | not_retrieved |  |
| enum_change-v1-02 | no_memory | `run_1a6b2547aa334bc38a511b75cf3b8805` | True | not_retrieved |  |
| enum_change-v1-03 | no_memory | `run_4d7b10732538445e886a58504905e027` | True | not_retrieved |  |
| enum_change-v1-04 | no_memory | `run_0a16b93c423d473fbd22e84dde8f0efe` | True | not_retrieved |  |
| enum_change-v1-05 | no_memory | `run_f3790312ef684d64b97c70fa3f457745` | True | not_retrieved |  |
| enum_change-v1-06 | no_memory | `run_dd746f65486f47b0968b3cdb2a04bb26` | True | not_retrieved |  |
| enum_change-v2-01 | no_memory | `run_afb152b5b5594269983c0c17d4fc5cc2` | True | not_retrieved |  |
| enum_change-v2-02 | no_memory | `run_e7848bd959014d19820691b8b0bc5190` | True | not_retrieved |  |
| enum_change-v2-03 | no_memory | `run_5d60f3da0fbe43628d641aefe3f2cb45` | True | not_retrieved |  |
| enum_change-v2-04 | no_memory | `run_1ed653752768476db215f8b801642262` | True | not_retrieved |  |
| enum_change-v2-05 | no_memory | `run_5df95742bd254195a83918a7d7eff6fb` | True | not_retrieved |  |
| enum_change-v2-06 | no_memory | `run_a3df0a27963c49b9abcc9743f89f286c` | True | not_retrieved |  |
| date_format-v1-01 | no_memory | `run_363e06a84e0449769488d9100a5d6edf` | True | not_retrieved |  |
| date_format-v1-02 | no_memory | `run_038a72a90dd04d28ae75ee1dc98228db` | True | not_retrieved |  |
| date_format-v1-03 | no_memory | `run_853be22859894a5c8a9be9d1808737a1` | True | not_retrieved |  |
| date_format-v1-04 | no_memory | `run_b5b8580a9c2c4adfb06467f4249e08bd` | True | not_retrieved |  |
| date_format-v1-05 | no_memory | `run_ae4be4a4793b4f3085230f001e7143bd` | True | not_retrieved |  |
| date_format-v1-06 | no_memory | `run_5e95eb0bfd374971a458763a06380022` | True | not_retrieved |  |
| date_format-v2-01 | no_memory | `run_1dc706b18ed740b28f443b88fb5dbcd2` | True | not_retrieved |  |
| date_format-v2-02 | no_memory | `run_23373469497042e185a6cd8b312137c0` | True | not_retrieved |  |
| date_format-v2-03 | no_memory | `run_e74893db6ead40ccaf31c90098491bfc` | True | not_retrieved |  |
| date_format-v2-04 | no_memory | `run_e5b04a1c11134dae9995d32c2f087419` | True | not_retrieved |  |
| date_format-v2-05 | no_memory | `run_53d9f48d76f6485e86c39fa6f62cecfc` | True | not_retrieved |  |
| date_format-v2-06 | no_memory | `run_94a34dec9b454c0c99d9eebd83df499f` | True | not_retrieved |  |
| required_parameter-v1-01 | bm25 | `run_1b984ec8c4284ea89cd301753f124d28` | True | selected |  |
| required_parameter-v1-02 | bm25 | `run_a1ff86dd17394e28b1d94ffa931df944` | True | selected |  |
| required_parameter-v1-03 | bm25 | `run_59954f0556bb4ae7a88263b91e14f402` | True | selected |  |
| required_parameter-v1-04 | bm25 | `run_dfcbea596245494a996ef61b36833b5c` | True | selected |  |
| required_parameter-v1-05 | bm25 | `run_9acb9bb6413b4ab7b43a2717eff29f53` | True | selected |  |
| required_parameter-v1-06 | bm25 | `run_fc1774b9690442bfa2fcddfa8305451d` | True | selected |  |
| required_parameter-v2-01 | bm25 | `run_fcdbf2c6f6ac4f3c9e8a4ad9ad25f670` | True | selected |  |
| required_parameter-v2-02 | bm25 | `run_328f4b9e71bd408181279dae54ab5e6e` | True | selected |  |
| required_parameter-v2-03 | bm25 | `run_303e47a6c9274a6dbd8a74e46dc711a0` | True | selected |  |
| required_parameter-v2-04 | bm25 | `run_5d5bfb747124451caf5d37e69b00b8fe` | True | selected |  |
| required_parameter-v2-05 | bm25 | `run_2adef261705c43bca091e643098b92a4` | True | selected |  |
| required_parameter-v2-06 | bm25 | `run_9dd0e2071d764900a4fd5577c41bb886` | True | selected |  |
| enum_change-v1-01 | bm25 | `run_96917c3321ae4d129c3e8e62365e952f` | True | selected |  |
| enum_change-v1-02 | bm25 | `run_81e894e2d0f84f82a6299ef3b76c3323` | True | selected |  |
| enum_change-v1-03 | bm25 | `run_0c3a420c1cf14a07a9b4958db989d975` | True | selected |  |
| enum_change-v1-04 | bm25 | `run_e393cfcfa5a34c8482452e3b9ba9c7b0` | True | selected |  |
| enum_change-v1-05 | bm25 | `run_6105212a016d490c9999c8b59102af35` | True | selected |  |
| enum_change-v1-06 | bm25 | `run_df984b9378974422bd5ad4388bffd0ed` | True | selected |  |
| enum_change-v2-01 | bm25 | `run_76c10f001c9e479bb496e208a63973b1` | True | selected |  |
| enum_change-v2-02 | bm25 | `run_3d420be84e5c4b269dead4c150115ad6` | True | selected |  |
| enum_change-v2-03 | bm25 | `run_f90bb2cc4300440184c51317156982af` | True | selected |  |
| enum_change-v2-04 | bm25 | `run_c5560843e02649a69412de4770c9a391` | True | selected |  |
| enum_change-v2-05 | bm25 | `run_12303844803a446f8d7874ccdfaea9c6` | True | selected |  |
| enum_change-v2-06 | bm25 | `run_55f7cde469034dfea252fc8ca3aa880f` | True | selected |  |
| date_format-v1-01 | bm25 | `run_ac812c13311b4edcb327e73c77d96e12` | True | selected |  |
| date_format-v1-02 | bm25 | `run_7a23e14077a14c30b66c52379d4b6e4e` | True | selected |  |
| date_format-v1-03 | bm25 | `run_b51eb9326e234e5a9d03e51709456f10` | True | selected |  |
| date_format-v1-04 | bm25 | `run_fea9f63e95fb450c99a5535ca1204123` | True | selected |  |
| date_format-v1-05 | bm25 | `run_7ce20934a91748a7b27489db5b60ae4c` | True | selected |  |
| date_format-v1-06 | bm25 | `run_4619d67c93e84104b64a9033979e401c` | True | selected |  |
| date_format-v2-01 | bm25 | `run_8ca07d31b6ef4e2fa61ddb852fa24989` | True | selected |  |
| date_format-v2-02 | bm25 | `run_08bcb5fd9cda4f88aace45b801d6d43c` | True | selected |  |
| date_format-v2-03 | bm25 | `run_d2cf7e0a136748a58bb6e3f86be9b4d2` | True | selected |  |
| date_format-v2-04 | bm25 | `run_56a543e4aa00468a8e2241a47535083a` | True | selected |  |
| date_format-v2-05 | bm25 | `run_f4212b741264485eaa895c9d5364fdfa` | True | selected |  |
| date_format-v2-06 | bm25 | `run_1a44d6814c204f85aed6a002e8f89111` | True | selected |  |
| required_parameter-v1-01 | hybrid_rrf_version_filter | `run_a851d81d43f34a209a3da95e001b5ac8` | True | selected |  |
| required_parameter-v1-02 | hybrid_rrf_version_filter | `run_e45008be18c94f1fbedf004cf2da06a0` | True | selected |  |
| required_parameter-v1-03 | hybrid_rrf_version_filter | `run_9db422e0b3f1488d8ba35f051108dd65` | True | selected |  |
| required_parameter-v1-04 | hybrid_rrf_version_filter | `run_58c91e2d24704c478b26914e8a6f39bc` | True | selected |  |
| required_parameter-v1-05 | hybrid_rrf_version_filter | `run_254fdd46c06044629bd5e95a6e7a750b` | True | selected |  |
| required_parameter-v1-06 | hybrid_rrf_version_filter | `run_ae18b053f4bc46af8b41e6c456327251` | True | selected |  |
| required_parameter-v2-01 | hybrid_rrf_version_filter | `run_90953826380745ba9138316623c208ef` | True | selected |  |
| required_parameter-v2-02 | hybrid_rrf_version_filter | `run_a9435f6be06e4a14b97a0c10f37cd3cb` | True | selected |  |
| required_parameter-v2-03 | hybrid_rrf_version_filter | `run_c1649f748d2742f8a8fb0acec59a9961` | True | selected |  |
| required_parameter-v2-04 | hybrid_rrf_version_filter | `run_25bc931805a448eaa81bc62331e1a391` | True | selected |  |
| required_parameter-v2-05 | hybrid_rrf_version_filter | `run_634e0a4608f742d284ac5c9ea37b4f8b` | True | selected |  |
| required_parameter-v2-06 | hybrid_rrf_version_filter | `run_6042c255b6354bfc99f7b61afb2834ce` | True | selected |  |
| enum_change-v1-01 | hybrid_rrf_version_filter | `run_627251b31c9f442c945e8f7d5c401ff7` | True | selected |  |
| enum_change-v1-02 | hybrid_rrf_version_filter | `run_0e450e7a5a904158af21b599fd8359bc` | True | selected |  |
| enum_change-v1-03 | hybrid_rrf_version_filter | `run_1eb5e82dc96942a6b2248f76ffbaa488` | True | selected |  |
| enum_change-v1-04 | hybrid_rrf_version_filter | `run_4e0606d655cc492b9a62ad4f2be82b17` | True | selected |  |
| enum_change-v1-05 | hybrid_rrf_version_filter | `run_14a3dabef0fe4424a964dcd9cf7361bb` | True | selected |  |
| enum_change-v1-06 | hybrid_rrf_version_filter | `run_ec0dbcc5fa234ada839e55894cb2cd17` | True | selected |  |
| enum_change-v2-01 | hybrid_rrf_version_filter | `run_d8ec81c47bc043a19fb7a1fa62b34cc7` | True | selected |  |
| enum_change-v2-02 | hybrid_rrf_version_filter | `run_2ff159ba20304619984e8ad3ec4ff323` | True | selected |  |
| enum_change-v2-03 | hybrid_rrf_version_filter | `run_95ffa10b8fdc4df28a1323ae19b27d05` | True | selected |  |
| enum_change-v2-04 | hybrid_rrf_version_filter | `run_f4cf439e9c8c41e48b1ad29b46208f0a` | True | selected |  |
| enum_change-v2-05 | hybrid_rrf_version_filter | `run_204a68dd75c4401291181c496cb3a1e3` | True | selected |  |
| enum_change-v2-06 | hybrid_rrf_version_filter | `run_e37bc43a21374a678d43a1d6b6f567ae` | True | selected |  |
| date_format-v1-01 | hybrid_rrf_version_filter | `run_6e0258e627b649bfb1e4687b75a76315` | True | selected |  |
| date_format-v1-02 | hybrid_rrf_version_filter | `run_469d6c8917f842e986756c74dc81f0eb` | True | selected |  |
| date_format-v1-03 | hybrid_rrf_version_filter | `run_fd07dd1189d646cf8e76f6013b16b38b` | True | selected |  |
| date_format-v1-04 | hybrid_rrf_version_filter | `run_527cc82d51834685a606a406b0e83262` | True | selected |  |
| date_format-v1-05 | hybrid_rrf_version_filter | `run_f1550914522c49258bbb66d3adfa420d` | True | selected |  |
| date_format-v1-06 | hybrid_rrf_version_filter | `run_c94f59b0d4d34d638449a4cb00abbf49` | True | selected |  |
| date_format-v2-01 | hybrid_rrf_version_filter | `run_6c6c7c48dc954be0881acf5ec5af68a7` | True | selected |  |
| date_format-v2-02 | hybrid_rrf_version_filter | `run_0efea3e3a22c475cb2cea69a34cb1a5e` | True | selected |  |
| date_format-v2-03 | hybrid_rrf_version_filter | `run_8ff683493de04c5b8727a1b61517f629` | True | selected |  |
| date_format-v2-04 | hybrid_rrf_version_filter | `run_429917e107b04a6293d06876327ef1b9` | True | selected |  |
| date_format-v2-05 | hybrid_rrf_version_filter | `run_e5828ac0a0c44503aaf39db756167173` | True | selected |  |
| date_format-v2-06 | hybrid_rrf_version_filter | `run_eeeed4bb7d764edabde40e39683fe90f` | True | selected |  |
| required_parameter-v1-01 | outcome_aware | `run_8186d8c6e95b44a1a23dd478867e168f` | True | selected |  |
| required_parameter-v1-02 | outcome_aware | `run_aff963890751492a821a62c77a9df285` | True | selected |  |
| required_parameter-v1-03 | outcome_aware | `run_86a1777feed1487e968307c5ed03b762` | True | selected |  |
| required_parameter-v1-04 | outcome_aware | `run_9919dab292064c7c92cc065f22354238` | True | selected |  |
| required_parameter-v1-05 | outcome_aware | `run_3dc1b24441d04ed4a4c0c6ee0cb55a83` | True | selected |  |
| required_parameter-v1-06 | outcome_aware | `run_b55928a1c5964055b9a059d53fc7dfc6` | True | selected |  |
| required_parameter-v2-01 | outcome_aware | `run_46c87344db364cfba0171ee58acae0d5` | True | selected |  |
| required_parameter-v2-02 | outcome_aware | `run_b4d15d432ab846ac88fd3ab840aa2d14` | True | selected |  |
| required_parameter-v2-03 | outcome_aware | `run_8842f0c84fa54f1d8b2e6509659c62a9` | True | selected |  |
| required_parameter-v2-04 | outcome_aware | `run_f8bd2b8280b9495698a6c868c86d1727` | True | selected |  |
| required_parameter-v2-05 | outcome_aware | `run_09a50903aba94d2aa4a1b1564f26e1f2` | True | selected |  |
| required_parameter-v2-06 | outcome_aware | `run_9dff85215ecf449db47c06a17fe99f0a` | True | selected |  |
| enum_change-v1-01 | outcome_aware | `run_9d1b0d924ea8484fa00a350f036c79f3` | True | selected |  |
| enum_change-v1-02 | outcome_aware | `run_6750b370b8e04f469be48793946e2a4a` | True | selected |  |
| enum_change-v1-03 | outcome_aware | `run_64ac89db05484736b0c45d0e564c9337` | True | selected |  |
| enum_change-v1-04 | outcome_aware | `run_e2a8bafac889492dba0239615e547d10` | True | selected |  |
| enum_change-v1-05 | outcome_aware | `run_b3bb65e6afd445788faec173e794050d` | True | selected |  |
| enum_change-v1-06 | outcome_aware | `run_6b2858e559094cd59a90bb7f208967a0` | True | selected |  |
| enum_change-v2-01 | outcome_aware | `run_219876c3f34f43f8bcf877d0240d0292` | True | selected |  |
| enum_change-v2-02 | outcome_aware | `run_eb9849801d994b09966e8322801dab25` | True | selected |  |
| enum_change-v2-03 | outcome_aware | `run_9baf57244164456b9b0dec88472357b0` | True | selected |  |
| enum_change-v2-04 | outcome_aware | `run_8785752a6d5343ac9c26fef58207cf07` | True | selected |  |
| enum_change-v2-05 | outcome_aware | `run_b2ac7c4452c3487f8cd3e00a78dc23fe` | True | selected |  |
| enum_change-v2-06 | outcome_aware | `run_da365b6b33ef484fb52c577bd108b226` | True | selected |  |
| date_format-v1-01 | outcome_aware | `run_a5f7532da6bf4ec9bfb3bb6655d7cf4e` | True | selected |  |
| date_format-v1-02 | outcome_aware | `run_26ac7bb7d4814b8394f57dc6b804a71b` | True | selected |  |
| date_format-v1-03 | outcome_aware | `run_f62cbc53af3f4851b345ff0d454c1e87` | True | selected |  |
| date_format-v1-04 | outcome_aware | `run_101993b357f3441fb2063b3c2447f0ff` | True | selected |  |
| date_format-v1-05 | outcome_aware | `run_10a3139d004249279438bfc7a7da91f6` | True | selected |  |
| date_format-v1-06 | outcome_aware | `run_eba16ca4b5a945d58c9fdf4877ec04e9` | True | selected |  |
| date_format-v2-01 | outcome_aware | `run_283db8ac869e4189b8872eba091ff78c` | True | selected |  |
| date_format-v2-02 | outcome_aware | `run_710867f485d24d20893c47903ef34a6d` | True | selected |  |
| date_format-v2-03 | outcome_aware | `run_399a0a3f067948f8848505ca2151e521` | True | selected |  |
| date_format-v2-04 | outcome_aware | `run_46d441fa9a1d45e7a112f7c0ced3fc52` | True | selected |  |
| date_format-v2-05 | outcome_aware | `run_104f19f528074855a31bd1cf2035875f` | True | selected |  |
| date_format-v2-06 | outcome_aware | `run_72a2a749c4fe4e2f93db4fbffbfdfe64` | True | selected |  |

## Model usage (provider-reported)

| baseline | mean input tokens | mean output tokens | mean duration ms | errors |
|---|---|---|---|---|
| bm25 | 2239.0 | 78.3 | 5770.0 | 0 |
| hybrid_rrf_version_filter | 1934.5 | 66.8 | 11633.9 | 0 |
| no_memory | 1003.0 | 68.5 | 5511.7 | 0 |
| outcome_aware | 1934.3 | 66.7 | 4873.8 | 0 |

## Retrieval latency (non-reproducible)

| baseline | measured tasks | total p50 ms | total p95 ms | rerank p50 ms | rerank p95 ms |
|---|---|---|---|---|---|
| bm25 | 36 | 117.197 | 134.426 | n/a | n/a |
| hybrid_rrf_version_filter | 36 | 1994.846 | 5034.841 | n/a | n/a |
| no_memory | 0 | n/a | n/a | n/a | n/a |
| outcome_aware | 36 | 161.398 | 217.436 | n/a | n/a |

## Retrieval configuration

- default mode: `outcome_aware`
- configured reranker: `lexical`
- active reranker: `lexical-overlap-v1`
- reranker model path: ``
- embedding model: `text-embedding-nomic-embed-text-v1.5`
- vector dimension: 768
- active index: `recap-test-nomic-memories-v1`

## Embedding usage

- provider: `openai_compatible`
- model: `text-embedding-nomic-embed-text-v1.5`
- dimension: 768
- cache hits / misses: 67 / 54 (hit rate 0.554)
- provider batches / texts: 36 / 54
- provider latency: 81938.830 ms
- prompt tokens: 0

## Reproduction

```bash
recap eval run --suite v2 --split test --agent llm --seed 1
```