# Four generated DUT examples

Every result below was generated. Each excerpt comes from the written wafer log; the complete identity comes from external metadata.

## PASS: Product_A-L01-W01-D001

Source: `logs/L01/W01.txt`

```text
Device: 001      Station: 1   Site: 0    Date: Thu Jan  1 08:00:10 2026
   100 Open/Short-          Failing Pins:  0
   210 IDD_Static         curr         35.00 ma <    44.23 ma     <  55.00 ma
   606 SCAN Test            Failing Pins:  0
       Pattern: scan
Bin:  1     Wafer Coordinates: ( 2 , 14)
```

## CONTINUITY_FAIL: Product_A-L01-W03-D005

Source: `logs/L01/W03.txt`

```text
Device: 005      Station: 1   Site: 0    Date: Thu Jan  1 08:13:54 2026
   100 Open/Short-          Halt Vector:      0   Halt Cycle:      0   Failing Pins:  3 (F)
          Failed Pins:
                 XCKP : 45     DTO2P : 82       CCK : 79
Bin:  4     Wafer Coordinates: ( 7 , 15)
```

## IDD_STATIC_FAIL: Product_A-L01-W01-D100

Source: `logs/L01/W01.txt`

```text
Device: 100      Station: 1   Site: 0    Date: Thu Jan  1 08:05:07 2026
   100 Open/Short-          Failing Pins:  0
   210 IDD_Static         curr         35.00 ma <    65.58 ma (F) <  55.00 ma
Bin:  3     Wafer Coordinates: ( -1 , 23)
```

## SCAN_FAIL: Product_A-L01-W02-D001

Source: `logs/L01/W02.txt`

```text
Device: 001      Station: 1   Site: 0    Date: Thu Jan  1 08:06:56 2026
   100 Open/Short-          Failing Pins:  0
   210 IDD_Static         curr         35.00 ma <    47.21 ma     <  55.00 ma
   606 SCAN Test            Halt Vector:   4030   Halt Cycle:   4030   Failing Pins:  2 (F)
       Pattern: scan
          Failed Pins:
                TSTIN : 75     TSTEN : 37
Bin:  2     Wafer Coordinates: ( 2 , 14)
```
