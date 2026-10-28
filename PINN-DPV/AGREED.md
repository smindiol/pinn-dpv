# Agreed

## Ladder
- [ ] every run persists a checkpoint (parameters, optimizer state, weighting state, PRNG key, iteration, voltage index) through a callback every N iterations and at the end of each voltage, and resuming from the latest checkpoint reproduces the uninterrupted run `test_resume_equals_uninterrupted`
- [ ] run results (per-voltage metrics, timings, logs) are pushed to the orphan branch results of smindiol/pinn-dpv, the latest checkpoint of each run is uploaded as a release asset of the same repository, Colab also mirrors both to Drive, and the GitHub token is read only from Kaggle or Colab secrets `test_sync_writes_results_branch_and_asset`
- [ ] the current-voltage J(V) on validation and test devices and the field errors are evaluated at the end of each voltage instead of storing every voltage's weights; every voltage's weights are kept only in E0 `test_online_evaluation_matches_offline`
- [ ] the PCE is computed by a reimplementation of dPV calcPmax that reproduces the PCE stored in the dPV dataset from its own J-V curves `test_pce_matches_dpv_dataset`
- [ ] the generation term G(x) is recomputed in JAX at any collocation point and reproduces the G stored by dPV on its mesh, including the LDS transformation of E4 `test_generation_matches_dpv`
- [ ] every computation runs in JAX float64 `test_float64_enabled`
- [ ] the dPV HDF5 files reach the workers as a private Kaggle dataset on one account shared with the other three, and the operator copies them to Drive for Colab
- [ ] every remote step is rehearsed on a Kaggle GPU at small scale before it is sent to the Colab A100, and nothing is launched without the operator's authorization

<!-- position revision=research-concept-modelado-computacional-directo-e-inverso-de-celdas-solares-d-r09.md sha256=9ae12827f51472d4c9cba5c9051574e7a7b4ed80cfb35ba698c7cf9bdd98780b derivedAt=2026-10-27T15:14:15Z session=screen-e0-v05 target=kaggle-gpu -->
- [x] 1. E0 v05 screening group loss-init-a rehearsed on a Kaggle T4 at the pinned commit `@rehearsal:level screen-loss-init-a`
- [x] 2. E0 v05 screening group loss-init-b rehearsed on a Kaggle T4 at the pinned commit `@rehearsal:level screen-loss-init-b`
- [x] 3. E0 v05 screening group arch-fixed-a rehearsed on a Kaggle T4 at the pinned commit `@rehearsal:level screen-arch-fixed-a`
- [x] 4. E0 v05 screening group arch-fixed-b rehearsed on a Kaggle T4 at the pinned commit `@rehearsal:level screen-arch-fixed-b`
- [x] 5. E0 v05 screening group arch-brdr-a rehearsed on a Kaggle T4 at the pinned commit `@rehearsal:level screen-arch-brdr-a`
- [x] 6. E0 v05 screening group arch-brdr-b rehearsed on a Kaggle T4 at the pinned commit `@rehearsal:level screen-arch-brdr-b`
- [x] 7. E0 v05 screening group size-a rehearsed on a Kaggle T4 at the pinned commit `@rehearsal:level screen-size-a`
- [x] 8. E0 v05 screening group size-b rehearsed on a Kaggle T4 at the pinned commit `@rehearsal:level screen-size-b`
- [x] 9. E0 v05 screening group voltage-interfaces-a rehearsed on a Kaggle T4 at the pinned commit `@rehearsal:level screen-voltage-interfaces-a`
- [x] 10. E0 v05 screening group voltage-interfaces-b rehearsed on a Kaggle T4 at the pinned commit `@rehearsal:level screen-voltage-interfaces-b`
- [x] 11. E0 v05 screening group start rehearsed on a Kaggle T4 at the pinned commit `@rehearsal:level screen-start`
<!-- /position -->

