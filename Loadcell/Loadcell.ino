// PlateGo load cell: stability detection and removal-based reset
// Library: "HX711" by Bogdan Necula (bogde), install via Library Manager
// Keep scale_logic.h in the same sketch folder as this file.
//
// Serial output (115200 baud):
//   Machine-readable events, one per line, no prefix:
//       WEIGHT_STABLE:150.5
//       SCALE_EMPTY
//   Diagnostics, always start with "# " so a parser can skip them.
// Serial input: 'r' prints the raw reading for checking CAL_FACTOR.

#include <Arduino.h>
#include <HX711.h>
#include "scale_logic.h"

// HX711 pin configuration (unchanged)
const int HX711_DT  = 18;  // DT  -> GPIO 18
const int HX711_SCK = 4;   // SCK -> GPIO 4

// Calibration factor, raw units per gram. Unverified: put a known mass on, send 'r',
// then CAL_FACTOR = delta / known_grams.
const float CAL_FACTOR = 238.0;

// Startup zero (unchanged behavior: 10 s warm-up, then average 20 readings)
const uint32_t WARMUP_MS   = 10000;
const int      TARE_SAMPLES = 20;

// Diagnostic print interval
const uint32_t DIAG_INTERVAL_MS = 250;

HX711 scale;
ScaleLogic fsm;             // thresholds and the median filter live in scale_logic.h (PlateConfig)
ZeroTracker zero;

enum BootPhase { BOOT_WARMUP, BOOT_TARING, BOOT_RUN };
BootPhase phase = BOOT_WARMUP;

long offsetRaw = 0;         // startup zero; after boot only ZeroTracker moves it, and only while EMPTY
long tareSum = 0;
int  tareCount = 0;
long lastRaw = 0;

uint32_t lastDiag = 0;
ScaleState lastState = EMPTY;

void setup() {
  Serial.begin(115200);
  scale.begin(HX711_DT, HX711_SCK);

  Serial.println("# PlateGo load cell");
  Serial.println("# Warming up, keep the platform empty");
}

void loop() {
  uint32_t now = millis();

  while (Serial.available()) {
    if (Serial.read() == 'r') {
      Serial.printf("# raw=%ld offset=%ld delta=%ld grams=%.2f cal=%.1f\n",
                    lastRaw, offsetRaw, offsetRaw - lastRaw, (offsetRaw - lastRaw) / CAL_FACTOR, CAL_FACTOR);
    }
  }

  // Non-blocking warm-up
  if (phase == BOOT_WARMUP) {
    if (now >= WARMUP_MS) phase = BOOT_TARING;
    return;
  }

  // Only read when a new sample is ready, so the loop never blocks
  if (!scale.is_ready()) return;
  long raw = lastRaw = scale.read();

  // Non-blocking startup zero
  if (phase == BOOT_TARING) {
    tareSum += raw;
    tareCount++;
    if (tareCount >= TARE_SAMPLES) {
      offsetRaw = tareSum / tareCount;
      phase = BOOT_RUN;
      Serial.println("# Tared. Ready");
    }
    return;
  }

  // Signed on purpose: fabs() turned negative drift into fake weight.
  // Sign assumes raw drops under load (original wiring); flip if grams go negative with a plate on.
  float grams = (offsetRaw - raw) / CAL_FACTOR;

  // State machine (filters internally)
  ScaleEvent evt = fsm.update(grams, now);
  float w = fsm.filtered();

  // Machine-readable events (no prefix, one per line)
  if (evt == EV_WEIGHT_STABLE) {
    Serial.printf("WEIGHT_STABLE:%.1f\n", fsm.stableGrams());
  } else if (evt == EV_SCALE_EMPTY) {
    Serial.println("SCALE_EMPTY");
  }

  float corr = zero.update(w, fsm.state() == EMPTY, now);
  if (corr != 0) {
    offsetRaw -= lroundf(corr * CAL_FACTOR);
    if (fabsf(corr) >= 0.5f) Serial.printf("# rezero %.2f g\n", corr);  // tiny noise corrections stay silent
  }

  // Diagnostics (prefixed with "# ")
  if (fsm.state() != lastState) {
    Serial.printf("# state %s -> %s\n", stateName(lastState), stateName(fsm.state()));
    lastState = fsm.state();
  }

  if (now - lastDiag >= DIAG_INTERVAL_MS) {
    lastDiag = now;
    Serial.printf("# state=%s weight=%.1f g", stateName(fsm.state()), w);
    if (fsm.state() == STABILIZING) {
      Serial.printf(" stable=%.1f/%.1f s",
                    fsm.stabilityElapsedMs(now) / 1000.0f,
                    fsm.config().stable_ms / 1000.0f);
    }
    if (fsm.hasStableWeight()) {
      Serial.printf(" last_stable=%.1f g", fsm.stableGrams());
    }
    Serial.println();
  }
}
