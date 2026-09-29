# M4.0: validating the Lagrangian reward mode on the M3 setup

**Status: pre-registered, results pending.** The criteria and the decision rule
below were written before the first run finished, so that the decision on the
reward mode for M4 does not adapt to what the runs happen to show.

## Question

`docs/results/m3.md`, consequence 2, left open whether M4 keeps the fixed reward
weights or switches to the constrained formulation of section 6.4. The question
is decided here, on the one setup whose frontier has already been measured: PV
curtailment only, `1-LV-rural1--2-sw`, `moderate_growth`.

## What was changed to make the mode usable

The mode existed in M3 only as half a mechanism: `RewardComposer` accepted
multipliers, but nothing updated them. Two further defects surfaced when the
code was read against the claim that `d = 0.05` is the standard's criterion:

- **The K95 cost was not a rate.** It was normalised by `n_buses ×
  budget_windows`, so its weekly sum was the budget share *averaged over buses*
  and its limit would have been 1, not 0.05 -- and an average lets twelve
  compliant buses dilute one that fails. The voltage costs are now the
  violating windows at the worst bus per window closed. Their mean is a share
  of windows outside the band, which is what EN 50160 bounds; summing the worst
  bus over time is conservative.
- **The dual update did not exist.** `LagrangianCallback` now performs dual
  ascent once per rollout on an EMA-smoothed cost estimate, with bounded
  multipliers, and writes the multiplier history to `lagrangian.json` in the
  run directory.

`fixed_weights` mode is unchanged, bit for bit, so the M3 runs remain a valid
reference.

## Setup

Identical to M3 except for the reward mode:

| | |
|---|---|
| grid, scenario | `1-LV-rural1--2-sw`, `moderate_growth` |
| training | PPO, 600,000 steps, 8 workers, base seeds 1 to 5 (the M3 seeds) |
| PPO settings | Stable-Baselines3 defaults, as in M3: `n_steps` 2,048 per worker (16,384 per rollout), `batch_size` 64, network 64×64. `configs/agent/ppo.yaml` is **not** read by `train.py`; see the correction note in `m3.md` |
| reward | `lagrangian`: objective terms with the M3 weights; constraints priced by multipliers |
| limits | K95 rate 0.05, K100 rate 0, thermal overload 0 |
| dual settings | `DEFAULT_DUALS` in `experiment/callbacks.py` (step size 4 for the voltage rates, 80 for thermal overload; bounds 50 and 200); EMA decay 0.8; one update per rollout, about 37 per run |
| evaluation | test weeks, `EpisodeMode.EVALUATE`, baselines merged from the M3 results |

The dual settings are starting values chosen from the cost scales of the M3 test
weeks, not tuned. They replace the penalty weights as hyperparameters, with one
difference that is the point of the exercise: in theory the weights decide the
solution, whereas the dual settings decide only how fast it is reached.

## What to expect

On this setup the two formulations should land in almost the same place. The
thermal limit is zero, and a hinge penalty with a zero limit is an exact
penalty: above some finite weight, the penalised optimum *is* the constrained
optimum. M3's weight of 10 already drove overload to near zero, so M3 sits at
the zero-overload end of the frontier, and a Lagrangian run with `d = 0` aims
at the same point. K95 does not bind in this scenario (every cap at 0.40 or
tighter passes), so its multiplier should stay near zero.

The validation therefore tests **the mechanism, not an improvement**: whether
dual ascent reaches the constrained optimum at least as well as a hand-set
weight, without oscillating and without degenerating into a fixed weight at its
bound. The payoff of the formulation is expected in M4, where several objectives
compete and the constraint prices can no longer be read off one corner.

## Criteria

All on the test weeks, over five seeds, reported as in M3 (IQM with bootstrap
confidence interval).

| | criterion | reference |
|---|---|---|
| **V1** mechanism | no multiplier ends at its bound; the thermal multiplier changes by less than 10 % over the last 20 % of rollouts | `lagrangian.json` per run |
| **V2** constraints | `pass_rate` IQM = 1.0; overload IQM ≤ 2.0 | M3 policies: 1.0 and 0.00 to 1.98 |
| **V3** objective | curtailment IQM ≤ 45.0 MWh; no more than one of five policies dominated by a measured cap | M3: IQM 39.89, CI upper bound 45.01; one of five dominated |

## Decision rule

- **V1 to V3 hold** → M4 uses the Lagrangian mode. Section 13 D5 is updated
  accordingly, and the fixed-weight mode stays available as the comparison arm
  for M7.
- **V1 fails, multiplier at its bound** → the formulation degenerated into a
  fixed weight. Raise the bound once and rerun one seed; if it recurs, M4 stays
  on fixed weights and the Lagrangian mode returns in M7 as planned.
- **V1 fails, oscillation** → reduce the dual step size once and rerun one seed;
  the same fallback applies.
- **V2 or V3 fails with V1 holding** → the mechanism works but reaches a worse
  point than the hand-set weight. That is a result in its own right and is
  reported; M4 stays on fixed weights.

At most one correction round per failure mode, so that the rule cannot be
satisfied by tuning until it passes.

## Results

Pending.
