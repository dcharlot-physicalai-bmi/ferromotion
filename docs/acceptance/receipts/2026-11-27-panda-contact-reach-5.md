# Panda reach contact, revision 5, 2026-11-27

This file supersedes `docs/acceptance/receipts/2026-11-27-panda-contact-reach-4.md`, sha256 `d0f800200fbaa9bdc2794771f8be6754a2788b75f5603379f8f079203d3853f0`. That file is not modified. Receipt -3 (`40a4edcb84d2876baa96bd399fe045c0fdecb9a5c77ead7100142bb7d203a237`) is not modified. The episode hashes in those files are unchanged.

## Norm formula

Receipt -3's plant and zero-torque norm cells are the Euclidean norm of the seven `qfrc_constraint` components, computed as the square root of a left-associated sum of separately rounded squares. Starting from `0.0`:

```text
acc = 0.0
for v in qfrc_constraint[0..7]:
    acc = acc + (v * v)
norm = sqrt(acc)
```

`v * v` rounds, then `acc + product` rounds. That is `q.iter().map(|v| v * v).sum::<f64>().sqrt()`. It is not an FMA chain. Accumulating with `v.mul_add(v, acc)` and then taking `sqrt` reproduces 27 of these 32 cells and is not the formula that printed the table.

`panda_reach_contact_moves_the_arm` asserts `f64::to_bits` of this formula on both runs. The bits below are those cells. Their shortest scientific print is the decimal in receipt -3 (tick 0 is `4047905f90c0e588` = `4.7127916425884052e1`; tick 1 plant is `40467a19bb0d9839` = `4.4953910237920759e1`; tick 15 plant is `4042169ed41afab7` = `3.6176722062276376e1`).

| tick | plant norm bits | zero-torque norm bits |
|---|---|---|
| 0 | `4047905f90c0e588` | `4047905f90c0e588` |
| 1 | `40467a19bb0d9839` | `404680e8ddd20dcc` |
| 2 | `40459fc7c3d8b765` | `4045a0564d801716` |
| 3 | `4044e40457282872` | `4044e71a02ade292` |
| 4 | `4044442c822801cc` | `40444ed4cabfe926` |
| 5 | `4043cb4e26e3808f` | `4043d22d03230218` |
| 6 | `40435e7ef23bac4c` | `40436ca70c637d36` |
| 7 | `4043110ce00b20cc` | `40431a817d83c114` |
| 8 | `4042d9056261c086` | `4042d896004bdd2f` |
| 9 | `4042a12e18d44686` | `4042a43ef3fbcab5` |
| 10 | `404277961a0210a9` | `40427b41827ffb04` |
| 11 | `40424f51ee227f50` | `40425bbb9eb259fd` |
| 12 | `4042387b1194c2da` | `4042441551b5da8d` |
| 13 | `4042260174491678` | `404232f4c250922f` |
| 14 | `40421eb4156f0618` | `4042273470dc5498` |
| 15 | `4042169ed41afab7` | `40421fdb3664e183` |
