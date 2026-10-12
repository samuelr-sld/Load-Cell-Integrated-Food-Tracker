import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { scaleIndicator } from '../domain/homeView';
import { MACRO_COLOR } from './Charts';
import { Glass, ProgressBar, Ring, Tag, fmt } from './glass';
import { color, space, tone, type } from './theme';

const MACRO_LABEL = { protein: 'Protein', carbs: 'Carbs', fat: 'Fat' };
const lead = (r) => (r.lowerBound ? '≥ ' : '');

// Spoken summary so VoiceOver reads the card once instead of ring, number and bars separately.
const spoken = ({ calories: c, macros }) => [
  c.total == null ? 'Today: no calories logged yet' : `Today: ${c.lowerBound ? 'at least ' : ''}${c.total} kilocalories${c.goal ? ` of ${c.goal}` : ''}`,
  ...macros.map((m) => `${MACRO_LABEL[m.key]} ${m.total == null ? 'unknown' : `${m.total} grams`}${m.goal ? ` of ${m.goal}` : ''}`),
].join('. ');

// `overview` = domain/homeView.todayOverview(); `onSetGoals` is shown only while no goal exists.
export const TodayCard = ({ overview, onSetGoals }) => {
  const { calories: c, macros } = overview;
  const hasGoals = c.goal != null || macros.some((m) => m.goal != null);
  return (
    <Glass variant="hero" accessible accessibilityLabel={spoken(overview)}>
      <View style={s.heroRow}>
        <Ring fraction={c.fraction} over={c.over}>
          <Text style={[type.display, s.big]} numberOfLines={1} adjustsFontSizeToFit>{lead(c)}{fmt(c.total)}</Text>
          <Text style={[type.caption, { color: color.inkMuted }]}>{c.goal ? `of ${c.goal} kcal` : 'kcal'}</Text>
        </Ring>
        <View style={s.macros}>
          {macros.map((m) => (
            <View key={m.key} style={s.macro}>
              <View style={s.macroTop}>
                <View style={s.macroName}>
                  <View style={[s.swatch, { backgroundColor: MACRO_COLOR[m.key] }]} />
                  <Text style={[type.footnote, { color: color.inkMuted }]}>{MACRO_LABEL[m.key]}</Text>
                </View>
                <Text style={[type.number, { color: color.ink }]}>{lead(m)}{fmt(m.total)}{m.total != null && ' g'}</Text>
              </View>
              {m.goal != null && <ProgressBar fraction={m.fraction} tint={MACRO_COLOR[m.key]} over={m.over} />}
            </View>
          ))}
        </View>
      </View>
      {overview.mealCount === 0 && <Text style={[type.footnote, s.hint]}>Nothing logged today yet.</Text>}
      {!hasGoals && onSetGoals && (
        <Pressable onPress={onSetGoals} accessibilityRole="button" style={s.goalsLink}>
          <Text style={[type.footnote, { color: color.emerald, fontWeight: '700' }]}>Set daily goals to see your progress</Text>
        </Pressable>
      )}
    </Glass>
  );
};

// Truthful link state of the automatic scale scan (see homeView.scaleIndicator).
export const ScaleCard = ({ link }) => {
  const i = scaleIndicator(link);
  return (
    <Glass accessible accessibilityLabel={`${i.title}. ${i.body}`} style={{ marginTop: space.md }}>
      <View style={s.scaleRow}>
        <View style={[s.scaleDot, { backgroundColor: tone[i.tone].bg }]}><View style={[s.scaleCore, { backgroundColor: tone[i.tone].fg }]} /></View>
        <View style={{ flex: 1 }}>
          <Text style={[type.headline, { color: color.ink }]}>{i.title}</Text>
          <Text style={[type.footnote, { color: color.inkMuted, marginTop: 2 }]}>{i.body}</Text>
        </View>
      </View>
    </Glass>
  );
};

// `meal` is the display shape built in App.toListItem plus { tag, sync, time }. Estimates show "~" and a tag; unknown shows a dash.
export const MealCard = ({ meal, onPress }) => {
  const estimate = meal.tag?.label === 'Estimated';
  const macros = [['protein', meal.totalProtein], ['carbs', meal.totalCarbs], ['fat', meal.totalFat]];
  const kcal = `${estimate && meal.totalCalories != null ? '~' : ''}${fmt(meal.totalCalories)}`;
  const label = [
    meal.title, meal.time, `${kcal} kilocalories`,
    ...macros.map(([k, v]) => `${MACRO_LABEL[k]} ${v == null ? 'unknown' : `${v} grams`}`),
    meal.tag?.label, meal.sync === 'conflict' ? 'sync conflict' : meal.sync === 'pending' ? 'not synced yet' : null,
  ].filter(Boolean).join(', ');
  return (
    <Pressable onPress={onPress} accessibilityRole="button" accessibilityLabel={label} accessibilityHint="Opens meal details" style={({ pressed }) => [{ marginBottom: space.md, opacity: pressed ? 0.85 : 1 }]}>
      <Glass>
        <View style={s.mealTop}>
          <View style={{ flex: 1 }}>
            <View style={s.titleRow}>
              <Text style={[type.headline, { color: color.ink, flexShrink: 1 }]} numberOfLines={1}>{meal.title}</Text>
              {meal.tag && <Tag label={meal.tag.label} tone={meal.tag.tone} />}
            </View>
            {!!meal.foods && <Text style={[type.footnote, { color: color.inkMuted, marginTop: 2 }]} numberOfLines={1}>{meal.foods}</Text>}
            <Text style={[type.caption, { color: color.inkFaint, marginTop: 2 }]}>
              {meal.time}
              {meal.sync === 'conflict' ? '  ·  Needs your choice' : meal.sync === 'pending' ? '  ·  Not synced yet' : ''}
            </Text>
          </View>
          <View style={s.kcalBox}>
            <Text style={[type.number, { color: color.ink, fontSize: 22, lineHeight: 26 }]} numberOfLines={1} adjustsFontSizeToFit>{kcal}</Text>
            <Text style={[type.caption, { color: color.inkMuted }]}>kcal</Text>
          </View>
        </View>
        <View style={s.mealMacros}>
          {macros.map(([k, v]) => (
            <View key={k} style={s.mealMacro}>
              <View style={[s.swatch, { backgroundColor: MACRO_COLOR[k] }]} />
              <Text style={[type.footnote, { color: color.inkMuted }]} numberOfLines={1}>{MACRO_LABEL[k]} <Text style={{ color: color.ink, fontWeight: '700' }}>{fmt(v)}{v != null && ' g'}</Text></Text>
            </View>
          ))}
        </View>
      </Glass>
    </Pressable>
  );
};

const s = StyleSheet.create({
  heroRow: { flexDirection: 'row', alignItems: 'center', gap: space.lg },
  big: { fontSize: 32, lineHeight: 36, color: color.ink, maxWidth: 100, textAlign: 'center' },
  macros: { flex: 1, gap: space.md },
  macro: { gap: 6 },
  macroTop: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'baseline' },
  macroName: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  swatch: { width: 8, height: 8, borderRadius: 4 },
  hint: { color: color.inkMuted, marginTop: space.md, textAlign: 'center' },
  goalsLink: { minHeight: 44, justifyContent: 'center', alignItems: 'center', marginTop: space.xs },
  scaleRow: { flexDirection: 'row', alignItems: 'center', gap: space.md },
  scaleDot: { width: 36, height: 36, borderRadius: 18, alignItems: 'center', justifyContent: 'center' },
  scaleCore: { width: 12, height: 12, borderRadius: 6 },
  mealTop: { flexDirection: 'row', alignItems: 'flex-start', gap: space.md },
  titleRow: { flexDirection: 'row', alignItems: 'center', gap: space.sm },
  kcalBox: { alignItems: 'flex-end', minWidth: 64 },
  mealMacros: { flexDirection: 'row', justifyContent: 'space-between', marginTop: space.md, paddingTop: space.md, borderTopWidth: StyleSheet.hairlineWidth * 2, borderTopColor: 'rgba(6,95,70,0.12)' },
  mealMacro: { flexDirection: 'row', alignItems: 'center', gap: 6, flexShrink: 1 },
});
