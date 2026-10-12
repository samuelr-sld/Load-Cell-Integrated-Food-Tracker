// Plate scale filter + state machine. Plain C++11, no Arduino deps, so it runs in host tests.
#pragma once
#include <stdint.h>
#include <math.h>

struct PlateConfig {
  float empty_g = 10.0f;           // at or below this the platform counts as empty
  float detect_g = 30.0f;          // above this a plate counts as present
  float band_g = 3.0f;             // stable means within +/- band_g of the window mean
  uint32_t stable_ms = 3500;       // time within band before WEIGHT_STABLE
  uint32_t empty_ms = 1000;        // time at or below empty_g before SCALE_EMPTY
  float zero_steady_g = 1.0f;      // zero tracking: reading must stay within +/- this of its mean
  uint32_t zero_ms = 2000;         // zero tracking: for this long, while the FSM is EMPTY
};

enum ScaleState { EMPTY, WEIGHT_DETECTED, STABILIZING, WEIGHT_STABLE, WAITING_FOR_REMOVAL };
enum ScaleEvent { EV_NONE, EV_WEIGHT_STABLE, EV_SCALE_EMPTY };

inline const char* stateName(ScaleState s) {
  switch (s) {
    case EMPTY: return "EMPTY";
    case WEIGHT_DETECTED: return "WEIGHT_DETECTED";
    case STABILIZING: return "STABILIZING";
    case WEIGHT_STABLE: return "WEIGHT_STABLE";
    default: return "WAITING_FOR_REMOVAL";
  }
}

class ScaleLogic {
 public:
  explicit ScaleLogic(const PlateConfig& cfg = PlateConfig()) : cfg_(cfg) {}

  // Feed one reading in grams (signed, already tared) at time now_ms. Returns an event at most once per transition.
  ScaleEvent update(float grams, uint32_t now_ms) {
    const float w = filtered_ = median5(grams);

    // Empty timer runs in every state; unsigned subtraction keeps millis() wraparound safe.
    if (w <= cfg_.empty_g) {
      if (!low_) { low_ = true; low_since_ = now_ms; }
    } else {
      low_ = false;
    }
    const bool emptyLongEnough = low_ && (uint32_t)(now_ms - low_since_) >= cfg_.empty_ms;

    switch (state_) {
      case EMPTY:
        if (w > cfg_.empty_g) state_ = WEIGHT_DETECTED;
        break;

      case WEIGHT_DETECTED:
        if (emptyLongEnough) state_ = EMPTY;  // brief touch, nothing was reported, so no event
        else if (w > cfg_.detect_g) startWindow(w, now_ms);
        break;

      case STABILIZING:
        if (w <= cfg_.detect_g) { state_ = WEIGHT_DETECTED; break; }
        if (fabsf(w - mean()) > cfg_.band_g) { startWindow(w, now_ms); break; }
        sum_ += w; count_++;
        if ((uint32_t)(now_ms - window_start_) >= cfg_.stable_ms) {
          stable_g_ = mean();
          state_ = WEIGHT_STABLE;
          return EV_WEIGHT_STABLE;
        }
        break;

      case WEIGHT_STABLE:  // one-tick state so callers can see it; adding food later does not re-report
        state_ = WAITING_FOR_REMOVAL;
        // fall through
      case WAITING_FOR_REMOVAL:
        if (emptyLongEnough) { state_ = EMPTY; return EV_SCALE_EMPTY; }
        break;
    }
    return EV_NONE;
  }

  ScaleState state() const { return state_; }
  float filtered() const { return filtered_; }
  float stableGrams() const { return stable_g_; }
  bool hasStableWeight() const { return stable_g_ > 0; }
  uint32_t stabilityElapsedMs(uint32_t now_ms) const { return state_ == STABILIZING ? now_ms - window_start_ : 0; }
  const PlateConfig& config() const { return cfg_; }

 private:
  // Median of the last 5 samples: drops single HX711 spikes but lags a step by only 2 samples (an EMA lagged ~1.3 s).
  float median5(float g) {
    if (n_ == 0) for (int i = 0; i < 5; i++) ring_[i] = g;
    ring_[n_++ % 5] = g;
    float s[5];
    for (int i = 0; i < 5; i++) s[i] = ring_[i];
    for (int i = 1; i < 5; i++)
      for (int j = i; j > 0 && s[j - 1] > s[j]; j--) { float t = s[j]; s[j] = s[j - 1]; s[j - 1] = t; }
    return s[2];
  }

  void startWindow(float w, uint32_t now_ms) {
    state_ = STABILIZING;
    window_start_ = now_ms;
    sum_ = w; count_ = 1;
  }
  float mean() const { return sum_ / count_; }

  PlateConfig cfg_;
  ScaleState state_ = EMPTY;
  float ring_[5];
  uint32_t n_ = 0;
  float filtered_ = 0;
  bool low_ = false;
  uint32_t low_since_ = 0;
  uint32_t window_start_ = 0;
  float sum_ = 0;
  int count_ = 0;
  float stable_g_ = 0;
};

// Re-zeroes drift and the post-removal residual. Only acts while the FSM says EMPTY (so at most
// empty_g, never food that was reported) and the reading is steady, so a plate being set down is not zeroed.
class ZeroTracker {
 public:
  explicit ZeroTracker(const PlateConfig& cfg = PlateConfig()) : cfg_(cfg) {}

  // Returns grams to subtract from the zero (0 = no correction this tick).
  float update(float w, bool empty, uint32_t now_ms) {
    if (!empty || fabsf(w) > cfg_.empty_g) { active_ = false; return 0; }
    if (!active_ || fabsf(w - sum_ / count_) > cfg_.zero_steady_g) {
      active_ = true; since_ = now_ms; sum_ = w; count_ = 1;
      return 0;
    }
    sum_ += w; count_++;
    if ((uint32_t)(now_ms - since_) < cfg_.zero_ms) return 0;
    active_ = false;
    return sum_ / count_;
  }

 private:
  PlateConfig cfg_;
  bool active_ = false;
  uint32_t since_ = 0;
  float sum_ = 0;
  int count_ = 0;
};
