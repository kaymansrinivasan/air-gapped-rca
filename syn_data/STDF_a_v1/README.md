# Product_A STDF V4 files — SYNTHETIC

These files encode the existing generated Product_A logs. No physical DUT was measured, and they contain no investigation, confirmed cause, corrective action, or retest.

- `lot/L01.stdf` through `lot/L05.stdf`: one lot per file, seven wafers each.
- `wafer/L01_W01.stdf` through `wafer/L05_W07.stdf`: one wafer per file, 112 DUTs each.
- Unzip the bundle, then open an individual `.stdf` file in an STDF V4 viewer. Some viewers accept only one STDF per ZIP.
- One wafer file is easiest for a single wafer map; a lot file lets you select among its seven wafers in viewers that support multiple WIR/WRR pairs.

The streams use little-endian STDF V4 with FAR/MIR, WCR, PMR, WIR/WRR, PIR/FTR/PTR/PRR, PCR/HBR/SBR/TSR, and MRR records. PRR coordinates and bin IDs come from the generated metadata. Physical wafer diameter, die size, and orientation are unknown and marked as such in WCR. IDD values and limits are in mA. Failing pin bit positions use the pin numbers in `config.json`. PRR TEST_T is 0 because real elapsed test duration was not measured.

`manifest.json` lists file hashes and counts. `export_stdf.py` in the source dataset reproduces the files from `device_metadata.jsonl` and `config.json`.
