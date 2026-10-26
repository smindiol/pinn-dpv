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
