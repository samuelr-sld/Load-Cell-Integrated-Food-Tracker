// Host tests: g++ -std=c++11 -Wall -Wextra -o test_scale_logic test_scale_logic.cpp && ./test_scale_logic
// Arduino compiles every .cpp in the sketch folder, so keep this out of the firmware build.
#ifndef ARDUINO
#include <stdio.h>
#include <vector>
#include "scale_logic.h"

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { printf("FAIL line %d: %s\n", __LINE__, #cond); failures++; } } while (0)

struct Run {
  ScaleLogic s;
  uint32_t t = 0;
  std::vector<ScaleEvent> events;
  std::vector<float> stable;
  // Feed a constant reading for ms milliseconds at 10 Hz (HX711 default rate).
  void feed(float g, uint32_t ms) {
    for (uint32_t n = 0; n < ms / 100; n++, t += 100) {
      ScaleEvent e = s.update(g, t);
      if (e != EV_NONE) events.push_back(e);
      if (e == EV_WEIGHT_STABLE) stable.push_back(s.stableGrams());
    }
  }
};

static void test_place_and_remove() {
  Run r;
  r.feed(0, 2000);
  CHECK(r.events.empty());
  r.feed(250, 5000);
  CHECK(r.events.size() == 1 && r.events[0] == EV_WEIGHT_STABLE);
  CHECK(r.stable.size() == 1 && r.stable[0] > 247 && r.stable[0] < 253);
  CHECK(r.s.state() == WAITING_FOR_REMOVAL);
  r.feed(250, 10000);
  CHECK(r.events.size() == 1);  // reported once only
  r.feed(7.65f, 1500);          // known empty residual still counts as empty; event within 1 s + filter lag
  CHECK(r.events.size() == 2 && r.events[1] == EV_SCALE_EMPTY);
  CHECK(r.s.state() == EMPTY);
}

static void test_not_stable_before_3500ms() {
  Run r;
  r.feed(250, 3000);
  CHECK(r.events.empty());
}

static void test_wobble_restarts_window() {
  Run r;
  for (int i = 0; i < 20; i++) r.feed(i % 2 ? 200.0f : 220.0f, 500);  // +/-10 g swings for 10 s
  CHECK(r.events.empty());
}

static void test_small_jitter_still_stable() {
  Run r;
  for (int i = 0; i < 50; i++) r.feed(i % 2 ? 149.0f : 151.0f, 100);
  CHECK(r.events.size() == 1 && r.events[0] == EV_WEIGHT_STABLE);
}

static void test_single_spike_ignored() {
  Run r;
  r.feed(0, 1000);
  r.feed(500, 100);  // one bad HX711 sample
  r.feed(0, 1000);
  CHECK(r.s.state() == EMPTY);
  r.feed(300, 4000);
  r.feed(300, 2000);
  r.feed(-400, 100);  // spike while plate is on must not end the plate
  r.feed(300, 2000);
  CHECK(r.events.size() == 1);
}

static void test_light_item_ignored() {
  Run r;
  r.feed(25, 10000);  // between empty and detect thresholds
  r.feed(0, 2000);
  CHECK(r.events.empty());
  CHECK(r.s.state() == EMPTY);
}

static void test_short_dip_does_not_empty() {
  Run r;
  r.feed(300, 5000);
  r.feed(0, 500);  // lifted briefly
  r.feed(300, 2000);
  CHECK(r.events.size() == 1);
  CHECK(r.s.state() == WAITING_FOR_REMOVAL);
}

static void test_negative_reading_counts_as_empty() {
  Run r;
  r.feed(300, 5000);
  r.feed(-15, 2000);  // signed weight: drift below zero must not look like a plate
  CHECK(r.events.size() == 2 && r.events[1] == EV_SCALE_EMPTY);
}

static void test_millis_wraparound() {
  Run r;
  r.t = 0xFFFFFFFFu - 1000;
  r.feed(250, 5000);
  CHECK(r.events.size() == 1 && r.events[0] == EV_WEIGHT_STABLE);
}

static void test_zero_tracker() {
  ZeroTracker z;
  float got = 0;
  for (uint32_t t = 0; t <= 2000; t += 100) got += z.update(7.65f, true, t);
  CHECK(got > 7.6f && got < 7.7f);  // residual removed after 2 s

  ZeroTracker busy;
  got = 0;
  for (uint32_t t = 0; t <= 5000; t += 100) got += busy.update(5, false, t);
  CHECK(got == 0);  // never while a plate is present

  ZeroTracker shaky;
  got = 0;
  for (uint32_t t = 0; t <= 5000; t += 100) got += shaky.update((t / 100) % 2 ? 2.0f : 6.0f, true, t);
  CHECK(got == 0);  // never while the reading moves
}

int main() {
  test_zero_tracker();
  test_place_and_remove();
  test_not_stable_before_3500ms();
  test_wobble_restarts_window();
  test_small_jitter_still_stable();
  test_single_spike_ignored();
  test_light_item_ignored();
  test_short_dip_does_not_empty();
  test_negative_reading_counts_as_empty();
  test_millis_wraparound();
  if (failures) { printf("%d failure(s)\n", failures); return 1; }
  printf("All tests passed\n");
  return 0;
}
#endif  // ARDUINO
