# TODO

- [ ] Should the multimodal trainer stuff be included in trainer in verifiers repo? anyway we aren't using any trainer code from there
- [ ] Resolve deepspeed dependency properly - currently added directly to prime-rl to avoid verifiers[rl] conflict. The issue is that verifiers/rl/trainer/trainer.py imports deepspeed at the top, which gets triggered when importing multimodal adapter, but we don't actually use that trainer code in prime-rl. Need to either: make the import lazy, move multimodal trainer out of verifiers/rl/trainer, or properly use verifiers[rl] extras.
- [ ] Properly integrate multimodal_adapter field into OrchestratorConfig - currently using extra="allow" as a temporary workaround to parse multimodal_adapter from TOML files. Should be added as a proper field in the config schema instead of relying on Pydantic's extra field handling.
- [ ] Properly handle FSDP for VLMs - currently FSDP setup is skipped for VLMs because they have different model structure (no model.model.layers). Need to either: detect VLM structure and adapt FSDP setup, or implement proper FSDP support for VLMs with their nested architecture (e.g., model.model.language_model.layers for Qwen2VL).
- [ ] Remove debug logging of rewards/advantages in trainer (train.py) and orchestrator (orchestrator.py) - these were added temporarily for debugging and should be cleaned up.
