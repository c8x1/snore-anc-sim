# Snore ANC Hardware Prototype Setup

## Bill of Materials

| Role | Component | Part Number | Approx. Price |
|------|-----------|-------------|---------------|
| DSP | STM32F407 Discovery | STM32F407G-DISC1 | ¥80-120 |
| Reference Mic | I²S MEMS Mic | SPH0645LM4H | ¥15-25 |
| Error Mic ×2 | I²S MEMS Mic | SPH0645LM4H | ¥15×2 |
| Speaker ×2 | 25mm Full-Range | CUI CDM-25008 | ¥8×2 |
| Power | USB 5V / 3.7V LiPo + LDO | — | ¥10 |
| **Total** | | | **¥150-300** |

## Wiring Diagram

```
SPH0645 (ref) ──I²S──→ STM32F407 ──I²S──→ SPK_L (25mm)
SPH0645 (eL) ──I²S──→  Discovery  ──I²S──→ SPK_R (25mm)
SPH0645 (eR) ──I²S──→              ──UART──→ PC (NR monitor)
                        │
                        └── USB power
```

## I²S Configuration

- Master clock: STM32F407 I2S PLL (16kHz sample rate)
- Data format: I2S Philips, 16-bit
- DMA: Double-buffer mode (half/full transfer interrupts)

## Physical Setup

```
打鼾者(手机播放) → 0.8-1.2m → 妻子枕头
                              ├── Mic_ref (参考麦，枕面朝打鼾者)
                              ├── SPK_L, SPK_R (左/右扬声器，枕内)
                              └── Mic_eL, Mic_eR (左/右误差麦，耳位)
                                    │ I²S
                              STM32F4 Discovery
                                    │ UART
                              PC (实时 NR 监控)
```

## Test Procedure

### Phase A: Single-Channel (1 SPK + 1 Error Mic)

1. Flash firmware with `fxlms_core.c`
2. Play 150Hz tone from phone at 1m distance
3. Monitor UART output for real-time NR
4. Target: NR > 20dB

### Phase B: Subband FxLMS

1. Switch firmware to subband variant
2. Same test as Phase A
3. Compare convergence speed: target ≥ 2x improvement

### Phase C: Multi-Channel (2 SPK + 2 Error Mic)

1. Connect second speaker and error mic
2. Position error mics at left/right ear positions
3. Run multi-channel firmware
4. Target: NR > 10dB in both zones

## Embedded Firmware Structure

```
firmware/
├── Core/
│   ├── Src/
│   │   ├── main.c          — I2S DMA setup + main loop
│   │   ├── fxlms_core.c    — ANC algorithm (C port from Python)
│   │   ├── secondary_path.c — Offline S(z) calibration
│   │   └── uart_logger.c   — NR value output
│   └── Inc/
│       ├── fxlms_core.h
│       ├── secondary_path.h
│       └── uart_logger.h
└── Drivers/  (STM32 HAL)
```

## Key Porting Notes

Python → C translation guide:
- `np.dot(a, b)` → `arm_dot_prod_f32(a, b, len, &result)`
- `np.roll(buf, 1)` → circular buffer with pointer
- Float32 throughout (Cortex-M4F has hardware FPU)
- Use CMSIS-DSP for FFT (subband decomposition)
