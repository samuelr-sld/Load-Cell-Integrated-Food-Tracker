import React, { useEffect, useMemo, useRef, useState } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import {
  StyleSheet, Text, View, SectionList, ActivityIndicator,
  SafeAreaView, StatusBar, TouchableOpacity, ScrollView, Modal, AppState
} from 'react-native';
import { collection, onSnapshot, query } from 'firebase/firestore';
import { onAuthStateChanged, signInAnonymously } from 'firebase/auth';
import { db, auth } from './firebaseConfig';
import { summarize, detectPatterns, periodRange } from './domain/analytics';
import { localDayKey } from './domain/meal';
import { DEFAULT_PREFERENCES } from './domain/preferences';
import { openMealRepository } from './data/db';
import { startAutoSync } from './data/sync';
import { createScanClient, scanBaseUrl, startScanPolling } from './data/scanClient';
import MealForm from './ui/MealForm';
import MealDetails from './ui/MealDetails';
import NutritionDashboard from './ui/NutritionDashboard';
import PreferencesForm from './ui/PreferencesForm';
import AssistantScreen from './ui/AssistantScreen';
import { SafeAreaProvider, useSafeAreaInsets } from 'react-native-safe-area-context';
import { Backdrop, Banner, Button, Dock, DockButton, EmptyState, LoadingCard, StatusPill, cap } from './ui/glass';
import { MealCard, ScaleCard, TodayCard } from './ui/HomeParts';
import { color, space, type } from './ui/theme';
import { mealTag, syncPill, todayOverview } from './domain/homeView';

import {
  generateDietaryAdvice,
  generateSingleMealAdvice,
  generateWeeklyInsightsAdvice
} from './geminiService';

export default function App() {
  return <SafeAreaProvider><Home /></SafeAreaProvider>;
}

function Home() {
  const insets = useSafeAreaInsets();
  const [scaleLink, setScaleLink] = useState('checking'); // 'unpaired' | 'offline' | 'online': last scan poll result, display only
  const [meals, setMeals] = useState([]); // read-only legacy Firestore docs
  const [repo, setRepo] = useState(null);
  const [localMeals, setLocalMeals] = useState([]); // canonical meals from SQLite (the editable diary)
  const [undatedMeals, setUndatedMeals] = useState([]);
  const [conflicts, setConflicts] = useState([]);
  const [prefs, setPrefs] = useState({ ...DEFAULT_PREFERENCES, rev: 0 });
  const [prefsOpen, setPrefsOpen] = useState(false);
  const [assistantOpen, setAssistantOpen] = useState(false);
  const [unsyncedIds, setUnsyncedIds] = useState(new Set());
  const [localLoaded, setLocalLoaded] = useState(false);
  const [boundUid, setBoundUid] = useState(null); // uid the repository is scoped to
  const [syncStatus, setSyncStatus] = useState(null);
  const [today, setToday] = useState(() => localDayKey(Date.now()));
  const [formMeal, setFormMeal] = useState(undefined); // undefined = closed, null = new meal, meal = edit
  const [selectedLocalId, setSelectedLocalId] = useState(null);
  const [scan, setScan] = useState(null); // { scan, client }: a load-cell scan from the laptop awaiting review
  const modalBusy = useRef(false);
  const [dbError, setDbError] = useState(null);
  const [loading, setLoading] = useState(true); // legacy cloud history
  const [uid, setUid] = useState(null);
  const [authUnavailable, setAuthUnavailable] = useState(false);
  const [authAttempt, setAuthAttempt] = useState(0);
  const [cloudError, setCloudError] = useState(false);

  // Overall & Single AI states
  const [aiAdvice, setAiAdvice] = useState('');
  const [generatingAi, setGeneratingAi] = useState(false);
  const [selectedMeal, setSelectedMeal] = useState(null);
  const [singleModalVisible, setSingleModalVisible] = useState(false);
  const [singleMealAdvice, setSingleMealAdvice] = useState('');
  const [generatingSingleAi, setGeneratingSingleAi] = useState(false);

  // Weekly Insights State
  const [weeklyModalVisible, setWeeklyModalVisible] = useState(false);
  const [weeklyAdvice, setWeeklyAdvice] = useState('');
  const [loadingWeeklyAdvice, setLoadingWeeklyAdvice] = useState(false);

  const getJsDate = (timestamp) => {
    if (!timestamp) return new Date();
    if (timestamp.toDate) return timestamp.toDate();
    const parsed = new Date(timestamp);
    return isNaN(parsed.getTime()) ? new Date() : parsed;
  };

  const formatTimeOnly = (timestamp) => {
    const date = getJsDate(timestamp);
    return date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true });
  };

  const groupMealsByDate = (mealList) => {
    const currentYear = new Date().getFullYear();
    const sorted = [...mealList].sort((a, b) => {
      return getJsDate(b.timestamp || b.createdAt).getTime() - getJsDate(a.timestamp || a.createdAt).getTime();
    });

    const groups = {};
    sorted.forEach((meal) => {
      const date = getJsDate(meal.timestamp || meal.createdAt);
      const dateLabel = date.getFullYear() === currentYear
        ? date.toLocaleDateString('en-US', { month: 'long', day: 'numeric' })
        : date.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' });

      if (!groups[dateLabel]) groups[dateLabel] = [];
      groups[dateLabel].push(meal);
    });

    return Object.keys(groups).map((dateTitle) => ({
      title: dateTitle,
      data: groups[dateTitle],
    }));
  };

  // Gemini's weekly input: this calendar week from the deterministic engine (SQLite diary only).
  const computeWeeklyInsights = (mealList, todayKey, averageBasis) => {
    const week = periodRange('week', todayKey);
    const w = summarize(mealList, { ...week, today: todayKey, averageBasis });
    if (!w.mealCount) return null;
    return {
      avgCalories: w.averages.perDay.calories,
      avgProtein: w.averages.perDay.protein,
      avgCarbs: w.averages.perDay.carbs,
      avgFat: w.averages.perDay.fat,
      topFoods: detectPatterns(mealList, { ...week, types: ['frequent_foods'] }).slice(0, 3).map((p) => p.food),
    };
  };

  // Unknown nutrition renders as a dash, never as 0.
  const fmt = (v) => (v == null ? '—' : v);

  // Auth: wait for persisted session restore; sign in anonymously only if none exists.
  // No automatic retry: failure shows a retry button (authAttempt re-runs this effect).
  useEffect(() => {
    let cancelled = false;
    setAuthUnavailable(false);

    const unsubscribeAuth = onAuthStateChanged(
      auth,
      (user) => {
        if (cancelled) return;
        if (user) {
          setUid(user.uid);
          return;
        }
        setUid(null);
        signInAnonymously(auth).catch((error) => {
          if (cancelled) return;
          console.error('Anonymous sign-in failed:', error.code);
          setAuthUnavailable(true);
          setLoading(false);
        });
      },
      (error) => {
        if (cancelled) return;
        console.error('Auth state error:', error.code);
        setAuthUnavailable(true);
        setLoading(false);
      }
    );

    return () => {
      cancelled = true;
      unsubscribeAuth();
    };
  }, [authAttempt]);

  useEffect(() => {
    if (!uid) return;
    const q = query(collection(db, 'meal'));
    // Offline Firestore never errors, it just never answers; surface that instead of spinning forever.
    const timer = setTimeout(() => { setCloudError(true); setLoading(false); }, 10000);

    // Metadata changes on: offline Firestore answers from its (possibly empty) cache, which is not "connected".
    const unsubscribe = onSnapshot(
      q,
      { includeMetadataChanges: true },
      (snapshot) => {
        const mealList = [];
        snapshot.forEach((doc) => {
          mealList.push({ id: doc.id, ...doc.data() });
        });

        setMeals(mealList);
        if (!snapshot.metadata.fromCache) {
          clearTimeout(timer);
          setCloudError(false);
          setLoading(false);
        }
      },
      (error) => {
        clearTimeout(timer);
        console.error('Firestore error:', error);
        setCloudError(true);
        setLoading(false);
      }
    );

    return () => { clearTimeout(timer); unsubscribe(); };
  }, [uid]);

  // Local SQLite repository: canonical meals, available offline and before sign-in.
  useEffect(() => {
    let cancelled = false;
    openMealRepository()
      .then((r) => !cancelled && setRepo(r))
      .catch((error) => {
        console.error('Local database error:', error.message);
        if (!cancelled) setDbError(error.message);
      });
    return () => { cancelled = true; };
  }, []);

  // Scope local data to the current identity; adopts pending offline meals on first sign-in.
  useEffect(() => {
    if (!repo) return;
    let cancelled = false;
    setBoundUid(null);
    repo.setIdentity(uid)
      .then(() => !cancelled && setBoundUid(uid))
      .catch((error) => {
        console.error('Local identity error:', error.message);
        if (!cancelled) setDbError(error.message);
      });
    return () => { cancelled = true; };
  }, [repo, uid]);

  useEffect(() => {
    if (!repo) return;
    let cancelled = false;
    const load = () => Promise.all([repo.listMeals(), repo.listUndated(), repo.listConflicts(), repo.listOutbox(), repo.getPreferences()])
      .then(([list, undated, open, outbox, p]) => {
        if (cancelled) return;
        setPrefs(p);
        setLocalMeals(list);
        setUndatedMeals(undated);
        setConflicts(open);
        setUnsyncedIds(new Set(outbox.map((o) => o.meal_id)));
        setLocalLoaded(true);
      })
      .catch((error) => !cancelled && setDbError(error.message));
    const unsubscribe = repo.onChange(load);
    load();
    return () => { cancelled = true; unsubscribe(); };
  }, [repo]);

  // Cloud sync runs only once the repository is scoped to the signed-in uid; stopped on sign-out/switch.
  useEffect(() => {
    if (!repo || !boundUid) return;
    const sync = startAutoSync({ repo, firestore: db, uid: boundUid, onStatus: setSyncStatus });
    return () => { sync.stop(); setSyncStatus(null); };
  }, [repo, boundUid]);

  // Load-cell scans from the laptop: poll the pending-scan API (same Wi-Fi address and token as the AI assistant settings).
  // A scan opens the review form only when no other screen is open; saving still goes through the SQLite repository.
  modalBusy.current = formMeal !== undefined || !!selectedLocalId || !!scan || assistantOpen || prefsOpen;
  useEffect(() => {
    if (!repo) return;
    return startScanPolling({
      canShow: () => !modalBusy.current,
      onScan: (found, client) => setScan({ scan: found, client }),
      getClient: async () => {
        const cfg = JSON.parse((await AsyncStorage.getItem('platego.assistant.settings')) ?? 'null');
        const baseUrl = scanBaseUrl(cfg?.baseUrl);
        if (!(baseUrl && cfg?.token)) { setScaleLink('unpaired'); return null; }
        const client = createScanClient({ baseUrl, token: cfg.token });
        // Only observes the poll result for the Home indicator; polling and acknowledge behave as before.
        return { ...client, latest: () => client.latest().then((r) => { setScaleLink('online'); return r; }, (e) => { setScaleLink('offline'); throw e; }) };
      },
    });
  }, [repo]);

  // Calendar-day rollover: re-key "today" at local midnight and whenever the app returns to the foreground.
  useEffect(() => {
    let timer;
    const refresh = () => {
      setToday(localDayKey(Date.now()));
      const next = new Date();
      next.setHours(24, 0, 1, 0);
      clearTimeout(timer);
      timer = setTimeout(refresh, next.getTime() - Date.now());
    };
    refresh();
    const sub = AppState.addEventListener('change', (state) => state === 'active' && refresh());
    return () => { clearTimeout(timer); sub.remove(); };
  }, []);

  useEffect(() => setWeeklyAdvice(''), [today]); // cached advice described the previous 7-day window

  // Canonical meals rendered in the existing card shape (display only; legacy docs stay separate).
  const toListItem = (m) => ({
    id: `local-${m.id}`,
    localId: m.id,
    category: m.category,
    foods: m.items.map((i) => i.name).join(', '),
    mealType: `${m.category[0].toUpperCase()}${m.category.slice(1)} (${m.items.map((i) => i.name).join(', ')})`,
    timestamp: m.loggedAt,
    totalCalories: m.totals.calories,
    totalProtein: m.totals.protein,
    totalCarbs: m.totals.carbs,
    totalFat: m.totals.fat,
    totalWeight_g: m.items.reduce((sum, i) => sum + i.grams, 0),
  });

  const conflictByMeal = useMemo(() => new Map(conflicts.map((c) => [c.mealId, c])), [conflicts]);

  // Diary: canonical days (with daily totals), then undated, then the separate read-only legacy history.
  const sections = useMemo(() => {
    const byDay = new Map();
    localMeals.forEach((m) => byDay.set(m.dayKey, [...(byDay.get(m.dayKey) ?? []), m]));
    const days = [...byDay.keys()].sort().reverse().map((dayKey) => {
      const d = new Date(`${dayKey}T00:00:00`);
      const label = dayKey === today ? 'Today'
        : d.toLocaleDateString('en-US', { month: 'long', day: 'numeric', ...(d.getFullYear() !== new Date().getFullYear() && { year: 'numeric' }) });
      return { title: label, summary: summarize(byDay.get(dayKey), { from: dayKey, to: dayKey, today }), data: byDay.get(dayKey).map(toListItem) };
    });
    return [
      ...days,
      ...(undatedMeals.length ? [{ title: 'Undated meals', data: undatedMeals.map(toListItem) }] : []),
      ...groupMealsByDate(meals).map((s) => ({ ...s, title: `${s.title} · Cloud history (read-only)` })),
    ];
  }, [localMeals, undatedMeals, meals, today]);

  const weeklyStats = useMemo(() => computeWeeklyInsights(localMeals, today, prefs.averageBasis), [localMeals, today, prefs.averageBasis]);
  const allLocal = useMemo(() => [...localMeals, ...undatedMeals], [localMeals, undatedMeals]);
  const selectedLocal = allLocal.find((m) => m.id === selectedLocalId) ?? null;

  const isSample = (m) => m.source.kind === 'sample';

  // Demo-only: clearly labeled on-device samples (SQLite only; source kind 'sample' never enters the outbox).
  const loadSampleMeals = async () => {
    const H = 3600000;
    const food = (name, grams, calories, protein, carbs, fat) => ({ name: `SAMPLE ${name}`, grams, per100g: { calories, protein, carbs, fat } });
    const samples = [
      ['breakfast', 0, 20, food('oatmeal', 250, 70, 2.5, 12, 1.5)],
      ['lunch', 0, 6, food('chicken rice', 350, 160, 9, 20, 4)],
      ['dinner', 1, 18, food('grilled fish & veg', 300, 110, 14, 6, 3)],
    ];
    try {
      for (const [category, daysAgo, hoursAgo, item] of samples) {
        await repo.createMeal({ loggedAt: Date.now() - (daysAgo * 24 + hoursAgo) * H, category, items: [item], source: { kind: 'sample' } });
      }
    } catch (error) {
      setDbError(error.message);
    }
  };

  const clearSampleMeals = async () => {
    try {
      for (const m of allLocal.filter(isSample)) await repo.deleteMeal(m.id, m.revision.rev);
    } catch (error) {
      setDbError(error.message);
    }
  };

  const openWeeklyInsights = async () => {
    setWeeklyModalVisible(true);
    if (!weeklyAdvice && weeklyStats) {
      setLoadingWeeklyAdvice(true);
      const advice = await generateWeeklyInsightsAdvice(weeklyStats);
      setWeeklyAdvice(advice);
      setLoadingWeeklyAdvice(false);
    }
  };

  const handleGenerateGeneralAdvice = async () => {
    setGeneratingAi(true);
    const advice = await generateDietaryAdvice([...localMeals.map(toListItem), ...meals]);
    setAiAdvice(advice);
    setGeneratingAi(false);
  };

  // List items: canonical meals open the editable details; legacy docs open the read-only view.
  const openMealDetails = (item) => {
    setSelectedMeal(item);
    setSingleMealAdvice('');
    if (item.localId) setSelectedLocalId(item.localId);
    else setSingleModalVisible(true);
  };

  const handleGenerateMealAdvice = async () => {
    if (!selectedMeal) return;
    setGeneratingSingleAi(true);
    const advice = await generateSingleMealAdvice(selectedMeal);
    setSingleMealAdvice(advice);
    setGeneratingSingleAi(false);
  };

  // Shared by both detail views; the AI receives the display shape (unknown values stay unknown).
  const singleAiBox = (
    <View style={styles.singleAiBox}>
      <View style={styles.aiHeader}>
        <Text style={styles.singleAiTitle}>✨ Meal Optimization AI</Text>
        <TouchableOpacity
          style={[styles.aiButton, { backgroundColor: '#7c3aed' }, generatingSingleAi && { opacity: 0.7 }]}
          onPress={handleGenerateMealAdvice}
          disabled={generatingSingleAi}
        >
          {generatingSingleAi ? (
            <ActivityIndicator size="small" color="#ffffff" />
          ) : (
            <Text style={styles.aiButtonText}>Improve Meal</Text>
          )}
        </TouchableOpacity>
      </View>

      {singleMealAdvice ? (
        <Text style={styles.singleAiAdviceText}>{singleMealAdvice}</Text>
      ) : (
        <Text style={styles.aiPlaceholder}>
          Tap "Improve Meal" to get AI recommendations on balancing nutrients, adjusting portions, or adding vegetables to this specific meal.
        </Text>
      )}
    </View>
  );

  const pill = syncPill({ uid, conflicts: conflicts.length, unsynced: unsyncedIds.size, status: syncStatus });
  const localById = useMemo(() => new Map(allLocal.map((m) => [m.id, m])), [allLocal]);
  // Today's totals come straight from the analytics engine over the SQLite diary (unknown stays null).
  const overview = useMemo(
    () => todayOverview(summarize(allLocal, { from: today, to: today, today, averageBasis: prefs.averageBasis, goals: prefs.goals })),
    [allLocal, today, prefs]
  );
  const dateLabel = new Date(`${today}T00:00:00`).toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' });

  const openFirstConflict = () => {
    const m = localById.get(conflicts[0]?.mealId);
    if (m) openMealDetails(toListItem(m));
  };

  // Display model for one list item: canonical meals carry their estimate/corrected tag, legacy cloud docs are view-only.
  const cardOf = (item) => ({
    ...item,
    title: item.localId ? cap(item.category) : (item.mealType || 'Meal log'),
    foods: item.localId ? item.foods : null,
    tag: item.localId ? mealTag(localById.get(item.localId)) : { label: 'View only', tone: 'neutral' },
    sync: item.localId ? (conflictByMeal.has(item.localId) ? 'conflict' : unsyncedIds.has(item.localId) ? 'pending' : null) : null,
    time: item.localId && item.timestamp == null ? 'Undated' : formatTimeOnly(item.timestamp || item.createdAt),
  });

  return (
    <Backdrop>
      <StatusBar barStyle="dark-content" />

      <SectionList
        sections={dbError || !localLoaded ? [] : sections}
        keyExtractor={(item) => item.id}
        stickySectionHeadersEnabled={false}
        contentInsetAdjustmentBehavior="never"
        contentContainerStyle={{ paddingTop: insets.top + space.md, paddingHorizontal: space.lg, paddingBottom: insets.bottom + 128 }}
        ListHeaderComponent={(
          <View>
            <View style={styles.header}>
              <View style={{ flex: 1 }}>
                <Text style={[type.title, { color: color.ink }]} accessibilityRole="header">Today</Text>
                <Text style={[type.footnote, { color: color.inkMuted }]}>{dateLabel}</Text>
              </View>
              {pill && <StatusPill tone={pill.tone} label={pill.label} onPress={conflicts.length ? openFirstConflict : undefined} />}
            </View>

            {authUnavailable && (
              <Banner
                title="Sign-in unavailable" body="Your meals are still saved on this phone."
                action={{ label: 'Retry', onPress: () => { setLoading(true); setAuthAttempt((n) => n + 1); } }}
              />
            )}
            {cloudError && <Banner title="Cloud history unavailable" body="You’re offline or don’t have access. Showing meals saved on this phone." />}

            {dbError ? null : !localLoaded ? <LoadingCard /> : (
              <TodayCard overview={overview} onSetGoals={() => { setPrefsOpen(true); setWeeklyModalVisible(true); }} />
            )}
            {repo && <ScaleCard link={scaleLink} />}
          </View>
        )}
        ListEmptyComponent={dbError ? (
          <Banner tone="danger" title="Meals can’t be opened" body="Local meal storage failed to open. Restart the app to try again." />
        ) : !localLoaded ? null : (
          <EmptyState
            title="No meals yet"
            body="Place a plate on your scale, or log a meal by hand."
            action={repo ? { label: 'Log a meal', onPress: () => setFormMeal(null) } : undefined}
            secondary={repo ? { label: 'Load sample data (this phone only)', onPress: loadSampleMeals } : undefined}
          />
        )}
        ListFooterComponent={allLocal.some(isSample) ? <Button variant="danger" label="Clear sample demo data" onPress={clearSampleMeals} /> : null}
        renderSectionHeader={({ section: { title, summary } }) => (
          <View style={styles.sectionHeader}>
            <Text style={[type.headline, { color: color.ink }]} accessibilityRole="header">{title}</Text>
            {summary && (
              <Text style={[type.caption, { color: color.inkMuted, marginTop: 2 }]}>
                {summary.completeness.incompleteMeals.length ? 'At least ' : ''}{fmt(summary.nutrients.calories.total)} kcal · P {fmt(summary.nutrients.protein.total)} g · C {fmt(summary.nutrients.carbs.total)} g · F {fmt(summary.nutrients.fat.total)} g
                {summary.completeness.incompleteMeals.length ? ' · some values unknown' : ''}
              </Text>
            )}
          </View>
        )}
        renderItem={({ item }) => <MealCard meal={cardOf(item)} onPress={() => openMealDetails(item)} />}
      />

      <Dock bottom={insets.bottom - space.sm}>
        {repo && <DockButton label="Assistant" icon="chat" onPress={() => setAssistantOpen(true)} />}
        {repo && <DockButton primary label="Log meal" icon="plus" onPress={() => setFormMeal(null)} />}
        <DockButton label="Insights" icon="chart" onPress={openWeeklyInsights} />
      </Dock>

      {/* MODAL 1: NUTRITION INSIGHTS — dashboard and goals/preferences share one modal (iOS cannot stack two) */}
      <Modal
        visible={weeklyModalVisible}
        animationType="slide"
        presentationStyle="pageSheet"
        onRequestClose={() => (prefsOpen ? setPrefsOpen(false) : setWeeklyModalVisible(false))}
      >
        {prefsOpen && repo ? (
          <PreferencesForm key={prefs.rev} repo={repo} prefs={prefs} onClose={() => setPrefsOpen(false)} />
        ) : (
          <NutritionDashboard
            meals={allLocal}
            prefs={prefs}
            today={today}
            onEditPreferences={() => setPrefsOpen(true)}
            onClose={() => setWeeklyModalVisible(false)}
          >
            {/* Multi-day Gemini analysis (moved here from Home; same handler and output). */}
            <View style={styles.aiContainer}>
              <View style={styles.aiHeader}>
                <Text style={styles.aiTitle}>✨ Overall Diet Analysis</Text>
                <TouchableOpacity style={[styles.aiButton, generatingAi && { opacity: 0.7 }]} onPress={handleGenerateGeneralAdvice} disabled={generatingAi}>
                  {generatingAi ? <ActivityIndicator size="small" color="#ffffff" /> : <Text style={styles.aiButtonText}>Analyze All</Text>}
                </TouchableOpacity>
              </View>
              {aiAdvice ? (
                <ScrollView style={styles.adviceScroll}><Text style={styles.aiAdviceText}>{aiAdvice}</Text></ScrollView>
              ) : (
                <Text style={styles.aiPlaceholder}>Tap "Analyze All" to let Gemini evaluate your multi-day nutritional patterns.</Text>
              )}
            </View>

            {/* Gemini recommendation (kept for now): input is this calendar week from the deterministic engine. */}
            <View style={styles.recommendationCard}>
              <View style={styles.recomHeader}>
                <Text style={styles.recomIcon}>💡</Text>
                <Text style={styles.recomTitle}>RECOMMENDATION · THIS WEEK (GEMINI)</Text>
              </View>
              {loadingWeeklyAdvice ? (
                <ActivityIndicator size="small" color="#047857" style={{ marginVertical: 6 }} />
              ) : (
                <Text style={styles.recomText}>
                  {weeklyStats
                    ? weeklyAdvice || "Maintain your protein intake and try to include more vegetables and fiber-rich foods."
                    : 'Log meals this week to get a recommendation.'}
                </Text>
              )}
            </View>
          </NutritionDashboard>
        )}
      </Modal>

      {/* MODAL 2: SINGLE MEAL DETAIL */}
      <Modal
        visible={singleModalVisible}
        animationType="slide"
        presentationStyle="pageSheet"
        onRequestClose={() => setSingleModalVisible(false)}
      >
        <SafeAreaView style={styles.modalContainer}>
          {selectedMeal && (
            <ScrollView contentContainerStyle={{ padding: 20 }}>
              <View style={styles.modalHeader}>
                <Text style={styles.modalTitle}>{selectedMeal.mealType || 'Meal Details'}</Text>
                <TouchableOpacity onPress={() => setSingleModalVisible(false)} style={styles.closeRoundBtn}>
                  <Text style={styles.closeRoundBtnText}>✕</Text>
                </TouchableOpacity>
              </View>

              <Text style={styles.modalSubTime}>
                Logged at: {getJsDate(selectedMeal.timestamp || selectedMeal.createdAt).toLocaleString()}
                {'\n'}Cloud history · read-only (not part of the editable diary)
              </Text>

              <Text style={styles.modalSectionHeading}>Nutrition & Mass Breakdown</Text>
              <View style={styles.metricsGrid}>
                <View style={[styles.metricBox, { backgroundColor: '#ecfdf5', borderColor: '#a7f3d0' }]}>
                  <Text style={styles.metricLabel}>Total Energy</Text>
                  <Text style={[styles.metricValue, { color: '#059669' }]}>{fmt(selectedMeal.totalCalories)} kcal</Text>
                </View>
                <View style={[styles.metricBox, { backgroundColor: '#f0f9ff', borderColor: '#bae6fd' }]}>
                  <Text style={styles.metricLabel}>Portion Mass</Text>
                  <Text style={[styles.metricValue, { color: '#0284c7' }]}>{selectedMeal.totalWeight_g ?? 0} g</Text>
                </View>
              </View>

              <View style={styles.macroDetailedContainer}>
                <View style={styles.macroDetailRow}>
                  <Text style={styles.macroDetailName}>🥩 Dietary Protein</Text>
                  <Text style={styles.macroDetailVal}>{fmt(selectedMeal.totalProtein)} g</Text>
                </View>
                <View style={styles.macroDetailRow}>
                  <Text style={styles.macroDetailName}>🍞 Carbohydrates</Text>
                  <Text style={styles.macroDetailVal}>{fmt(selectedMeal.totalCarbs)} g</Text>
                </View>
                <View style={styles.macroDetailRow}>
                  <Text style={styles.macroDetailName}>🥑 Total Fats</Text>
                  <Text style={styles.macroDetailVal}>{fmt(selectedMeal.totalFat)} g</Text>
                </View>
              </View>

              {/* Single Meal AI Advisor */}
              {singleAiBox}

              <TouchableOpacity style={styles.doneBtn} onPress={() => setSingleModalVisible(false)}>
                <Text style={styles.doneBtnText}>Done</Text>
              </TouchableOpacity>
            </ScrollView>
          )}
        </SafeAreaView>
      </Modal>

      {/* MODAL 3: DIARY MEAL (SQLite) — details, edit, new meal and scan review share one modal (iOS cannot stack two) */}
      <Modal
        visible={!!selectedLocal || formMeal !== undefined || !!scan}
        animationType="slide"
        presentationStyle="pageSheet"
        onRequestClose={() => (formMeal !== undefined ? setFormMeal(undefined) : scan ? setScan(null) : setSelectedLocalId(null))}
      >
        {formMeal !== undefined && repo ? (
          <MealForm key={formMeal?.id ?? 'new'} repo={repo} meal={formMeal} onClose={() => setFormMeal(undefined)} />
        ) : scan && repo ? (
          <MealForm
            key={scan.scan.scan_id} repo={repo} meal={null} scan={scan.scan} onClose={() => setScan(null)}
            onScanDone={(outcome) => scan.client.acknowledge(scan.scan.scan_id, outcome).catch(() => {})}
          />
        ) : selectedLocal ? (
          <MealDetails
            repo={repo}
            meal={selectedLocal}
            conflict={conflictByMeal.get(selectedLocal.id) ?? null}
            pending={unsyncedIds.has(selectedLocal.id)}
            onEdit={() => setFormMeal(selectedLocal)}
            onClose={() => setSelectedLocalId(null)}
          >
            {singleAiBox}
          </MealDetails>
        ) : null}
      </Modal>

      {/* MODAL 4: NUTRITION ASSISTANT — keyed by identity so a sign-in switch reloads that user's conversation */}
      <Modal visible={assistantOpen && !!repo} animationType="slide" presentationStyle="pageSheet" onRequestClose={() => setAssistantOpen(false)}>
        {repo && <AssistantScreen key={boundUid ?? 'local'} repo={repo} today={today} onClose={() => setAssistantOpen(false)} />}
      </Modal>

    </Backdrop>
  );
}

const styles = StyleSheet.create({
  header: { flexDirection: 'row', alignItems: 'center', gap: 12, marginBottom: 16 },
  sectionHeader: { marginTop: 20, marginBottom: 12, paddingHorizontal: 4 },

  // AI Assistant Card
  aiContainer: { backgroundColor: '#ffffff', borderRadius: 14, padding: 16, elevation: 3, marginBottom: 12, borderWidth: 1, borderColor: '#e0e7ff' },
  aiHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 },
  aiTitle: { fontSize: 16, fontWeight: 'bold', color: '#4338ca' },
  aiButton: { backgroundColor: '#4f46e5', paddingVertical: 6, paddingHorizontal: 12, borderRadius: 8 },
  aiButtonText: { color: '#ffffff', fontWeight: '600', fontSize: 13 },
  aiPlaceholder: { fontSize: 13, color: '#6b7280', fontStyle: 'italic', lineHeight: 18 },
  adviceScroll: { maxHeight: 120, marginTop: 4 },
  aiAdviceText: { fontSize: 13, color: '#1f2937', lineHeight: 19 },

  // Modal Base
  modalContainer: { flex: 1, backgroundColor: '#ffffff' },
  modalHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  modalTitle: { fontSize: 20, fontWeight: '800', color: '#111827', flex: 1, marginRight: 10 },
  closeRoundBtn: { width: 28, height: 28, borderRadius: 14, backgroundColor: '#e2e8f0', alignItems: 'center', justifyContent: 'center' },
  closeRoundBtnText: { fontSize: 13, fontWeight: 'bold', color: '#475569' },
  modalSubTime: { fontSize: 13, color: '#6b7280', marginTop: 4, marginBottom: 16 },
  modalSectionHeading: { fontSize: 16, fontWeight: '700', color: '#374151', marginBottom: 10 },

  // Gemini recommendation card (inside the insights dashboard)
  recommendationCard: { 
    backgroundColor: '#ffffff', 
    borderRadius: 12, 
    padding: 14, 
    borderWidth: 1, 
    borderColor: '#cbd5e1', 
    marginVertical: 8 
  },
  recomHeader: { flexDirection: 'row', alignItems: 'center', marginBottom: 4 },
  recomIcon: { fontSize: 16, marginRight: 6 },
  recomTitle: { fontSize: 11, fontWeight: '800', color: '#1e293b', letterSpacing: 0.5 },
  recomText: { fontSize: 12, color: '#334155', lineHeight: 18 },

  // Metric Boxes for Single Meal
  metricsGrid: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 14 },
  metricBox: { flex: 0.48, padding: 14, borderRadius: 12, borderWidth: 1, alignItems: 'center' },
  metricLabel: { fontSize: 13, color: '#4b5563', fontWeight: '500', marginBottom: 4 },
  metricValue: { fontSize: 18, fontWeight: 'bold' },
  macroDetailedContainer: { backgroundColor: '#f9fafb', borderRadius: 12, padding: 14, marginBottom: 16 },
  macroDetailRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 6 },
  macroDetailName: { fontSize: 14, color: '#374151', fontWeight: '500' },
  macroDetailVal: { fontSize: 14, color: '#111827', fontWeight: '700' },

  singleAiBox: { backgroundColor: '#faf5ff', borderRadius: 14, padding: 16, borderWidth: 1, borderColor: '#e9d5ff', marginBottom: 20 },
  singleAiTitle: { fontSize: 15, fontWeight: 'bold', color: '#6b21a8' },
  singleAiAdviceText: { fontSize: 13, color: '#1f2937', lineHeight: 20, marginTop: 4 },

  doneBtn: { backgroundColor: '#111827', paddingVertical: 14, borderRadius: 12, alignItems: 'center', marginTop: 10, marginBottom: 30 },
  doneBtnText: { color: '#ffffff', fontSize: 15, fontWeight: 'bold' }
});