import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import Svg, { Circle, Path } from 'react-native-svg';

// Renders chartData() results from domain/analytics.js as-is: no value here is computed or invented.
// A bucket with nothing logged (value null) shows '·', never a zero-height bar.
const LABEL = { calories: 'Calories', protein: 'Protein', carbs: 'Carbs', fat: 'Fat' };
export const MACRO_COLOR = { carbs: '#2563eb', protein: '#16a34a', fat: '#d97706' };
const day = (k, opts) => new Date(`${k}T00:00:00Z`).toLocaleDateString('en-US', { timeZone: 'UTC', ...opts });

const tick = (chart, p, i) => {
  if (chart.groupBy === 'month') return day(p.key, { month: 'short' });
  if (chart.groupBy === 'week') return day(p.key, { month: 'numeric', day: 'numeric' });
  if (chart.points.length <= 10) return day(p.key, { weekday: 'narrow' });
  return i % 5 === 0 ? Number(p.key.slice(8)) : ' ';
};

// shortcut: line requests are drawn as bars (same points, same engine); add a line renderer if trends need it.
export const Bars = ({ chart, title }) => {
  const H = 80;
  const max = Math.max(1, chart.goal ?? 0, ...chart.points.map((p) => p.value ?? 0));
  return (
    <View>
      {title && <Text style={s.title}>{title}</Text>}
      <View style={[s.bars, { height: H + 16 }]}>
        {chart.goal != null && <View style={[s.goalLine, { bottom: 16 + (chart.goal / max) * H }]} />}
        {chart.points.map((p, i) => (
          <View key={p.key} style={s.barCol} accessibilityLabel={`${p.key}: ${p.value == null ? 'nothing logged' : `${p.lowerBound ? 'at least ' : ''}${p.value} ${chart.unit}`}`}>
            {p.value == null
              ? <Text style={s.barNone}>·</Text>
              : <View style={[s.bar, { height: Math.max((p.value / max) * H, 2), opacity: p.lowerBound ? 0.45 : 1 }]} />}
            <Text style={s.barLabel}>{tick(chart, p, i)}</Text>
          </View>
        ))}
      </View>
      <Text style={s.note}>
        {chart.valueKind === 'total' ? 'Daily totals' : `Average per day in each ${chart.groupBy}`} · '·' = nothing logged
        {chart.goal != null ? ` · line = goal ${chart.goal} ${chart.unit}` : ''}
        {chart.points.some((p) => p.lowerBound) ? ' · faded = some values unknown' : ''}
      </Text>
    </View>
  );
};

const R = 44;
const arc = (a0, a1) => {
  const pt = (a) => `${R + R * Math.cos(a)},${R + R * Math.sin(a)}`;
  return `M${R},${R} L${pt(a0)} A${R},${R} 0 ${a1 - a0 > Math.PI ? 1 : 0} 1 ${pt(a1)} Z`;
};

export const Pie = ({ chart, title }) => {
  if (!chart.slices) return <Text style={s.note}>Not enough meals with known carbs, protein and fat for an energy split.</Text>;
  const shown = chart.slices.filter((x) => x.pct > 0);
  let a = -Math.PI / 2;
  return (
    <View>
      {title && <Text style={s.title}>{title}</Text>}
      <View style={s.pieRow}>
        <Svg width={R * 2} height={R * 2} accessibilityLabel={`Energy split: ${chart.slices.map((x) => `${LABEL[x.key]} ${x.pct}%`).join(', ')}`}>
          {shown.length === 1
            ? <Circle cx={R} cy={R} r={R} fill={MACRO_COLOR[shown[0].key]} />
            : shown.map((x) => {
              const a0 = a;
              a += (x.pct / 100) * 2 * Math.PI;
              return <Path key={x.key} d={arc(a0, a)} fill={MACRO_COLOR[x.key]} />;
            })}
        </Svg>
        <View style={{ marginLeft: 14, flex: 1 }}>
          {chart.slices.map((x) => (
            <Text key={x.key} style={s.legend}>
              <Text style={{ color: MACRO_COLOR[x.key] }}>■ </Text>{LABEL[x.key]} {x.pct}% ({x.grams} g)
            </Text>
          ))}
        </View>
      </View>
      <Text style={s.note}>
        Share of energy (4/4/9 kcal per g){chart.excludedMeals > 0 ? ` · ${chart.excludedMeals} meal(s) with unknown macros left out` : ''}
      </Text>
    </View>
  );
};

const range = (c) => (c.from === c.to ? c.from : `${day(c.from, { month: 'short', day: 'numeric' })} – ${day(c.to, { month: 'short', day: 'numeric' })}`);

export const ChartView = ({ chart }) => (chart.type === 'pie'
  ? <Pie chart={chart} title={`Macronutrient energy split · ${range(chart)}`} />
  : <Bars chart={chart} title={`${LABEL[chart.metric]} (${chart.unit}) · ${range(chart)}`} />);

const s = StyleSheet.create({
  title: { fontSize: 11, fontWeight: '800', color: '#1e293b', letterSpacing: 0.3 },
  note: { fontSize: 10, color: '#64748b', marginTop: 6, fontStyle: 'italic' },
  bars: { flexDirection: 'row', alignItems: 'flex-end', marginTop: 10 },
  barCol: { flex: 1, alignItems: 'center', justifyContent: 'flex-end' },
  bar: { width: '60%', maxWidth: 14, backgroundColor: '#10b981', borderRadius: 3 },
  barNone: { fontSize: 12, color: '#94a3b8' },
  barLabel: { fontSize: 9, color: '#64748b', marginTop: 4, fontWeight: '600', height: 12 },
  goalLine: { position: 'absolute', left: 0, right: 0, height: 1, backgroundColor: '#dc2626' },
  pieRow: { flexDirection: 'row', alignItems: 'center', marginTop: 10 },
  legend: { fontSize: 12, color: '#1e293b', marginVertical: 2, fontWeight: '600' },
});
