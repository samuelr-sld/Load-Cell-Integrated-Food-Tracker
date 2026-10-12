import React, { useMemo, useState } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, ScrollView, SafeAreaView } from 'react-native';
import { NUTRIENTS, addDays } from '../domain/meal';
import { summarize, comparePeriods, chartData, detectPatterns, periodRange, previousRange } from '../domain/analytics';
import { Bars, MACRO_COLOR } from './Charts';

// Every number on this screen comes from domain/analytics.js over the SQLite diary; '—' = unknown.
const LABEL = { calories: 'Calories', protein: 'Protein', carbs: 'Carbs', fat: 'Fat' };
const UNIT = { calories: 'kcal', protein: 'g', carbs: 'g', fat: 'g' };
const fmt = (v) => (v == null ? '—' : v);
const cap = (c) => c[0].toUpperCase() + c.slice(1);
const day = (k, opts) => new Date(`${k}T00:00:00Z`).toLocaleDateString('en-US', { timeZone: 'UTC', ...opts });
const periodLabel = (period, r, today) => (
  period === 'day' ? (r.from === today ? 'Today' : day(r.from, { weekday: 'short', month: 'short', day: 'numeric' }))
    : period === 'week' ? `${day(r.from, { month: 'short', day: 'numeric' })} – ${day(r.to, { month: 'short', day: 'numeric' })}`
      : day(r.from, { month: 'long', year: 'numeric' })
);
// "≥" marks a lower bound: some meals lacked this nutrient.
const val = (v, lower, unit) => (v == null ? '—' : `${lower ? '≥ ' : ''}${v} ${unit}`);

const Chip = ({ on, label, onPress }) => (
  <TouchableOpacity style={[s.chip, on && s.chipOn]} onPress={onPress} accessibilityRole="button" accessibilityState={{ selected: on }}>
    <Text style={[s.chipText, on && s.chipTextOn]}>{label}</Text>
  </TouchableOpacity>
);

const Card = ({ title, children }) => (
  <View style={s.card}>
    <Text style={s.cardTitle}>{title}</Text>
    {children}
  </View>
);

export default function NutritionDashboard({ meals, prefs, today, onEditPreferences, onClose, children }) {
  const [period, setPeriod] = useState('week');
  const [anchor, setAnchor] = useState(today);
  const [metric, setMetric] = useState('calories');
  const range = periodRange(period, anchor);
  const atLatest = range.to >= today;

  const { sum, cmp, chart, patterns } = useMemo(() => {
    const opts = { today, averageBasis: prefs.averageBasis };
    const current = summarize(meals, { ...range, ...opts, goals: prefs.goals });
    return {
      sum: current,
      cmp: comparePeriods(current, summarize(meals, { ...previousRange(period, range), ...opts })),
      chart: period === 'day' ? null : chartData(meals, { type: 'bar', metric, ...range, ...opts, goals: prefs.goals }),
      patterns: detectPatterns(meals, { ...range, preferences: prefs }),
    };
  }, [meals, prefs, today, period, range.from, metric]);

  const multiDay = period !== 'day';
  const basis = prefs.averageBasis === 'calendar_days' ? 'calendar day' : 'logged day';
  const c = sum.completeness;
  const ofType = (t) => patterns.filter((p) => p.type === t);

  return (
    <SafeAreaView style={s.container}>
      <ScrollView contentContainerStyle={{ padding: 18 }}>
        <View style={s.banner}>
          <Text style={s.bannerTitle}>NUTRITION INSIGHTS</Text>
          <TouchableOpacity onPress={onClose} style={s.closeBtn} accessibilityLabel="Close">
            <Text style={s.closeText}>✕</Text>
          </TouchableOpacity>
        </View>

        <View style={s.row}>
          {['day', 'week', 'month'].map((p) => <Chip key={p} on={period === p} label={cap(p)} onPress={() => setPeriod(p)} />)}
        </View>
        <View style={[s.row, s.nav]}>
          <TouchableOpacity onPress={() => setAnchor(addDays(range.from, -1))} accessibilityLabel={`Previous ${period}`} style={s.navBtn}>
            <Text style={s.navText}>‹</Text>
          </TouchableOpacity>
          <TouchableOpacity onPress={() => setAnchor(today)} accessibilityLabel="Jump to today" style={{ flex: 1 }}>
            <Text style={s.navLabel}>{periodLabel(period, range, today)}</Text>
          </TouchableOpacity>
          <TouchableOpacity onPress={() => setAnchor(addDays(range.to, 1))} disabled={atLatest} accessibilityLabel={`Next ${period}`} style={[s.navBtn, atLatest && { opacity: 0.3 }]}>
            <Text style={s.navText}>›</Text>
          </TouchableOpacity>
        </View>

        <Card title={`TOTALS · ${sum.mealCount} MEAL(S) ON ${sum.loggedDays} DAY(S)`}>
          <View style={s.tRow}>
            <Text style={[s.tCell, s.tHead, { flex: 1.2 }]} />
            <Text style={[s.tCell, s.tHead]}>Total</Text>
            {multiDay && <Text style={[s.tCell, s.tHead]}>Avg / {basis}</Text>}
            <Text style={[s.tCell, s.tHead]}>Goal</Text>
          </View>
          {NUTRIENTS.map((n) => {
            const lower = sum.nutrients[n].missingMeals > 0;
            const g = sum.goals?.[n];
            return (
              <View key={n} style={s.tRow}>
                <Text style={[s.tCell, s.tName, { flex: 1.2 }]}>{LABEL[n]}</Text>
                <Text style={s.tCell}>{val(sum.nutrients[n].total, lower, UNIT[n])}</Text>
                {multiDay && <Text style={s.tCell}>{val(sum.averages.perDay[n], lower, UNIT[n])}</Text>}
                <Text style={s.tCell}>{g ? `${g.pctOfGoal == null ? '—' : `${g.lowerBound ? '≥ ' : ''}${g.pctOfGoal}%`} of ${g.goal}` : '—'}</Text>
              </View>
            );
          })}
          {Object.values(sum.nutrients).some((x) => x.missingMeals) && <Text style={s.note}>≥ = at least: some meals have unknown values (not counted as zero).</Text>}
          {multiDay && Object.values(prefs.goals).some((g) => g != null) && <Text style={s.note}>Goals compare against the average per {basis}.</Text>}
        </Card>

        <Card title={`VS PREVIOUS ${period.toUpperCase()} (AVG PER ${basis.toUpperCase()})`}>
          {NUTRIENTS.map((n) => {
            const x = cmp.nutrients[n];
            const text = x.comparable
              ? `${x.change > 0 ? '▲' : x.change < 0 ? '▼' : '='} ${Math.abs(x.change)} ${UNIT[n]}${x.changePct == null ? '' : ` (${x.changePct > 0 ? '+' : ''}${x.changePct}%)`}`
              : x.current == null || x.previous == null ? 'no data in one period' : 'not comparable: some values unknown';
            return (
              <View key={n} style={s.tRow}>
                <Text style={[s.tCell, s.tName]}>{LABEL[n]}</Text>
                <Text style={[s.tCell, { flex: 2.4 }]}>{text}</Text>
              </View>
            );
          })}
        </Card>

        {chart && (
          <Card title="DAILY TREND">
            <View style={s.row}>
              {NUTRIENTS.map((n) => <Chip key={n} on={metric === n} label={LABEL[n]} onPress={() => setMetric(n)} />)}
            </View>
            <Bars chart={chart} />
          </Card>
        )}

        <Card title="MACRONUTRIENT DISTRIBUTION (ENERGY)">
          {sum.macros.pct ? (
            <>
              <View style={s.macroBar}>
                {Object.entries(sum.macros.pct).map(([k, p]) => <View key={k} style={{ flex: Math.max(p, 1), backgroundColor: MACRO_COLOR[k] }} />)}
              </View>
              <Text style={s.line}>
                {Object.entries(sum.macros.pct).map(([k, p]) => `${LABEL[k]} ${p}% (${sum.macros.grams[k]} g)`).join(' · ')}
              </Text>
              {sum.macros.excludedMeals > 0 && <Text style={s.note}>Based on {sum.macros.basedOnMeals} of {sum.mealCount} meals; the rest lack carbs, protein or fat.</Text>}
            </>
          ) : <Text style={s.note}>Not enough meals with known carbs, protein and fat.</Text>}
        </Card>

        {sum.byMealType.length > 0 && (
          <Card title="BY MEAL TYPE">
            {sum.byMealType.map((b) => (
              <View key={b.category} style={s.tRow}>
                <Text style={[s.tCell, s.tName]}>{cap(b.category)}</Text>
                <Text style={s.tCell}>{b.mealCount} meal(s)</Text>
                <Text style={s.tCell}>{val(b.nutrients.calories.total, b.nutrients.calories.missingMeals > 0, 'kcal')}</Text>
              </View>
            ))}
          </Card>
        )}

        {multiDay && patterns.length > 0 && (
          <Card title="PATTERNS IN YOUR LOG">
            {ofType('exclusion_matches').map((p) => (
              <Text key={p.term} style={[s.line, { color: '#b45309' }]}>⚠ "{p.term}" appears in {p.mealCount} meal name(s): {p.evidence.matchedNames.join(', ')}</Text>
            ))}
            {ofType('frequent_foods').slice(0, 5).map((p) => (
              <Text key={p.food} style={s.line}>• {p.food}: {p.mealCount} meals on {p.dayCount} day(s)</Text>
            ))}
            {ofType('meal_type_logging').map((p) => (
              <Text key={p.category} style={s.line}>• {cap(p.category)} logged on {p.daysLogged} of {p.loggedDays} logged day(s)</Text>
            ))}
            {ofType('goal_adherence').map((p) => (
              <Text key={p.nutrient} style={s.line}>
                • {LABEL[p.nutrient]} above your goal on {p.daysAbove} of {p.daysAbove + p.daysAtOrBelow} day(s)
                {p.daysExcludedIncomplete ? ` (${p.daysExcludedIncomplete} day(s) with unknown values not judged)` : ''}
              </Text>
            ))}
            <Text style={s.note}>Based only on what was logged: food names, meal types and daily totals.</Text>
          </Card>
        )}

        <Card title="DATA COMPLETENESS">
          <Text style={s.line}>{sum.mealCount ? `${c.completeMeals} of ${sum.mealCount} meals have complete nutrition.` : 'No meals logged in this period.'}</Text>
          {NUTRIENTS.filter((n) => sum.nutrients[n].missingMeals).map((n) => (
            <Text key={n} style={s.line}>• {LABEL[n]} unknown in {sum.nutrients[n].missingMeals} meal(s)</Text>
          ))}
          {multiDay && c.unloggedDays.length > 0 && <Text style={s.line}>• {c.unloggedDays.length} day(s) with nothing logged</Text>}
          {sum.evidence.estimatedMealIds.length > 0 && <Text style={s.line}>• {sum.evidence.estimatedMealIds.length} scanned meal(s): values are scan estimates</Text>}
          {sum.evidence.userEditedScanMealIds.length > 0 && <Text style={s.line}>• {sum.evidence.userEditedScanMealIds.length} scanned meal(s) edited by you</Text>}
          {sum.evidence.sampleMealIds.length > 0 && <Text style={[s.line, { color: '#b45309' }]}>• {sum.evidence.sampleMealIds.length} SAMPLE demo meal(s) included</Text>}
          {c.undatedMealsExcluded > 0 && <Text style={s.line}>• {c.undatedMealsExcluded} undated meal(s) not included in any period</Text>}
          <Text style={s.note}>Read-only cloud history is not included; totals use meals saved on this device.</Text>
        </Card>

        {children}

        <TouchableOpacity style={[s.btn, { backgroundColor: '#1e3a8a' }]} onPress={onEditPreferences}>
          <Text style={s.btnText}>Goals & preferences</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[s.btn, { backgroundColor: '#111827', marginBottom: 30 }]} onPress={onClose}>
          <Text style={s.btnText}>Close Insights</Text>
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
}

const s = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#ffffff' },
  banner: { backgroundColor: '#0c3875', borderRadius: 8, paddingVertical: 12, paddingHorizontal: 14, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 },
  bannerTitle: { color: '#ffffff', fontWeight: '900', fontSize: 16, letterSpacing: 0.5 },
  closeBtn: { width: 28, height: 28, borderRadius: 14, backgroundColor: '#e2e8f0', alignItems: 'center', justifyContent: 'center' },
  closeText: { fontSize: 13, fontWeight: 'bold', color: '#475569' },
  row: { flexDirection: 'row', alignItems: 'center', gap: 6, flexWrap: 'wrap' },
  chip: { paddingVertical: 6, paddingHorizontal: 12, borderRadius: 16, borderWidth: 1, borderColor: '#cbd5e1' },
  chipOn: { backgroundColor: '#10b981', borderColor: '#10b981' },
  chipText: { fontSize: 13, color: '#334155', fontWeight: '600' },
  chipTextOn: { color: '#ffffff' },
  nav: { marginVertical: 8 },
  navBtn: { paddingHorizontal: 14, paddingVertical: 4 },
  navText: { fontSize: 24, fontWeight: '700', color: '#1e3a8a' },
  navLabel: { textAlign: 'center', fontSize: 15, fontWeight: '700', color: '#1e293b' },
  card: { backgroundColor: '#ffffff', borderRadius: 12, padding: 12, borderWidth: 1, borderColor: '#cbd5e1', marginVertical: 6 },
  cardTitle: { fontSize: 10, fontWeight: '800', color: '#1e293b', marginBottom: 8, letterSpacing: 0.4 },
  tRow: { flexDirection: 'row', paddingVertical: 3 },
  tCell: { flex: 1, fontSize: 12, color: '#0f172a', fontWeight: '600' },
  tHead: { fontSize: 10, color: '#64748b', fontWeight: '700' },
  tName: { color: '#475569', fontWeight: '500' },
  line: { fontSize: 12, color: '#1e293b', marginVertical: 2 },
  note: { fontSize: 10, color: '#64748b', marginTop: 6, fontStyle: 'italic' },
  macroBar: { flexDirection: 'row', height: 10, borderRadius: 5, overflow: 'hidden', marginVertical: 8 },
  btn: { paddingVertical: 13, borderRadius: 12, alignItems: 'center', marginTop: 8 },
  btnText: { color: '#ffffff', fontSize: 15, fontWeight: 'bold' },
});
