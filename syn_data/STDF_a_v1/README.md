# Product_A STDF V4 files — SYNTHETIC

These files encode the existing generated Product_A logs. No physical DUT was measured, and they contain no investigation, confirmed cause, corrective action, or retest.

- `lot/L01.stdf` through `lot/L05.stdf`: one lot per file, seven wafers each.
- `wafer/L01_W01.stdf` through `wafer/L05_W07.stdf`: one wafer per file, 112 DUTs each.
- Open an individual `.stdf` file directly in an STDF V4 viewer.
- One wafer file is easiest for a single wafer map; a lot file lets you select among its seven wafers in viewers that support multiple WIR/WRR pairs.

The streams use little-endian STDF V4 with FAR/MIR, WCR, PMR, WIR/WRR, PIR/FTR/PTR/PRR, PCR/HBR/SBR/TSR, and MRR records. PRR coordinates and bin IDs come from the generated metadata. Physical wafer diameter, die size, and orientation are unknown and marked as such in WCR. IDD values and limits are in mA. Failing pin bit positions use the pin numbers in `config.json`. PRR TEST_T is 0 because real elapsed test duration was not measured.

`manifest.json` lists file hashes and counts. The matching generated logs, metadata and configuration are in `../product_a_v1/`.
