# Battery delivery option pilot

A customer buys a call on up to 1 MWh of electricity. The battery’s operating history determines how much of that energy it could deliver at 4 MW over 15 minutes. The study uses the repository’s shallow randomized-signature reservoir to estimate deliverable volume and translate that estimate into an option quote and a one-trade forward hedge.

The [results](REPORT.md) include the comparison on 30,000 simulated test histories and the independent confirmation on another 24,000. The [cell model](BATTERY_MODEL.md) documents the operating scenario and the parameters adapted from NASA’s published electrochemical model.

## Reproduce

Install the Python dependencies in the repository’s `requirements.txt`. From the repository root run

```sh
python tests/test_battery_delivery.py
python studies/battery_delivery/reproduce.py
```

The first command checks the physical simulator and the option pricing and hedging formulas against independent numerical solutions. The second runs every split, supplemental comparison and sensitivity check. All data are simulated locally. Generated predictions and per-path paired losses stay under the study’s `results/` directory. The full run regenerates the [pooled and per-split metrics](results/metrics.csv).

For an individual split, enter `studies/battery_delivery` and run, for example, `python run_pilot.py --split 1`. The primary [protocol](protocol.json) and [confirmation protocol](confirmation_protocol.json) give the split sizes, seeds and comparison rules. The normal power-price model assumes no transaction costs, no basis risk and no premium for physical volume risk.
