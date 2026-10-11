import React, { useEffect, useState } from 'react';
import { 
  StyleSheet, Text, View, SectionList, ActivityIndicator, 
  SafeAreaView, StatusBar, TouchableOpacity, ScrollView, Modal 
} from 'react-native';
import { collection, onSnapshot, query } from 'firebase/firestore';
import { signInAnonymously } from 'firebase/auth';
import { db, auth } from './firebaseConfig';

import { 
  generateDietaryAdvice, 
  generateSingleMealAdvice, 
  generateWeeklyInsightsAdvice 
} from './geminiService';

export default function App() {
  const [meals, setMeals] = useState([]);
  const [sections, setSections] = useState([]);
  const [loading, setLoading] = useState(true);

  // Overall & Single AI states
  const [aiAdvice, setAiAdvice] = useState('');
  const [generatingAi, setGeneratingAi] = useState(false);
  const [selectedMeal, setSelectedMeal] = useState(null);
  const [singleModalVisible, setSingleModalVisible] = useState(false);
  const [singleMealAdvice, setSingleMealAdvice] = useState('');
  const [generatingSingleAi, setGeneratingSingleAi] = useState(false);

  // Weekly Insights State
  const [weeklyModalVisible, setWeeklyModalVisible] = useState(false);
  const [weeklyStats, setWeeklyStats] = useState(null);
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

  // Helper to extract actual food dish items (ignoring Meal tags like Lunch/Dinner)
  const extractFoodItems = (mealTypeStr) => {
    if (!mealTypeStr) return ['Logged Dish'];
    
    // Check for dish names inside parentheses e.g. "Lunch (Chicken Adobo & White Rice)"
    const match = mealTypeStr.match(/\((.*?)\)/);
    const content = match ? match[1] : mealTypeStr;

    // Split by commas, &, +, and "and"
    return content
      .split(/[,&+]|\band\b/i)
      .map((item) => item.trim())
      .filter((item) => item.length > 0);
  };

  // Real-time recalculation of averages, distribution, and top foods
  const computeWeeklyInsights = (mealList) => {
    if (!mealList || mealList.length === 0) return null;

    let totalCals = 0;
    let totalProteinGrams = 0;
    let totalCarbsGrams = 0;
    let totalFatGrams = 0;

    const dayCalories = { Mon: 0, Tue: 0, Wed: 0, Thu: 0, Fri: 0, Sat: 0, Sun: 0 };
    const foodFrequencyMap = {};
    const uniqueActiveDays = new Set();

    mealList.forEach((m) => {
      const c = Number(m.totalCalories) || 0;
      const p = Number(m.totalProtein) || 0;
      const cb = Number(m.totalCarbs) || 0;
      const f = Number(m.totalFat) || 0;

      totalCals += c;
      totalProteinGrams += p;
      totalCarbsGrams += cb;
      totalFatGrams += f;

      const d = getJsDate(m.timestamp || m.createdAt);
      uniqueActiveDays.add(d.toDateString());

      const dayName = d.toLocaleDateString('en-US', { weekday: 'short' });
      if (dayCalories[dayName] !== undefined) {
        dayCalories[dayName] += c;
      }

      // Count individual foods
      const foodsInMeal = extractFoodItems(m.mealType);
      foodsInMeal.forEach((food) => {
        foodFrequencyMap[food] = (foodFrequencyMap[food] || 0) + 1;
      });
    });

    const activeDaysCount = Math.max(uniqueActiveDays.size, 1);
    const avgCalories = Math.round(totalCals / activeDaysCount);
    const avgProtein = Math.round(totalProteinGrams / activeDaysCount);
    const avgCarbs = Math.round(totalCarbsGrams / activeDaysCount);
    const avgFat = Math.round(totalFatGrams / activeDaysCount);

    // Accurate caloric macronutrient distribution (P: 4 kcal/g, C: 4 kcal/g, F: 9 kcal/g)
    const calFromP = totalProteinGrams * 4;
    const calFromC = totalCarbsGrams * 4;
    const calFromF = totalFatGrams * 9;
    const totalMacroCals = calFromP + calFromC + calFromF;

    let carbsPct = 0, proteinPct = 0, fatPct = 0;
    if (totalMacroCals > 0) {
      carbsPct = Math.round((calFromC / totalMacroCals) * 100);
      proteinPct = Math.round((calFromP / totalMacroCals) * 100);
      fatPct = Math.max(0, 100 - carbsPct - proteinPct);
    }

    const topFoods = Object.entries(foodFrequencyMap)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([foodName]) => foodName);

    return {
      avgCalories,
      avgProtein,
      avgCarbs,
      avgFat,
      carbsPct,
      proteinPct,
      fatPct,
      dayCalories,
      topFoods: topFoods.length > 0 ? topFoods : ['Logged Food Item']
    };
  };

  
  useEffect(() => {
    let unsubscribe = null;
    let cancelled = false;

    const initializeFirebase = async () => {
      try {
        // STEP 1: Authenticate anonymously before reading Firestore
        if (!auth.currentUser) {
          await signInAnonymously(auth);
        }

        // Prevent starting a listener after unmount
        if (cancelled) return;

        console.log(
          'Firebase authenticated successfully:',
          auth.currentUser?.uid
        );

        // STEP 2: Connect to the existing meal collection
        const q = query(collection(db, 'meal'));

        // STEP 3: Start the real-time Firestore listener
        unsubscribe = onSnapshot(
          q,
          (snapshot) => {
            if (cancelled) return;

            const mealList = [];

            snapshot.forEach((doc) => {
              mealList.push({
                id: doc.id,
                ...doc.data(),
              });
            });

            // Update existing application states
            setMeals(mealList);
            setSections(groupMealsByDate(mealList));
            setWeeklyStats(computeWeeklyInsights(mealList));
            setLoading(false);

            console.log(
              'Firestore connected. Meals loaded:',
              mealList.length
            );
          },
          (error) => {
            if (cancelled) return;

            console.error('Firestore error:', error);
            setLoading(false);
          }
        );
      } catch (error) {
        if (cancelled) return;

        console.error('Firebase authentication error:', error);
        setLoading(false);
      }
    };

    initializeFirebase();

    // STEP 4: Clean up Firestore listener on unmount
    return () => {
      cancelled = true;

      if (unsubscribe) {
        unsubscribe();
      }
    };
  }, []);


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
    const advice = await generateDietaryAdvice(meals);
    setAiAdvice(advice);
    setGeneratingAi(false);
  };

  const openMealDetails = (meal) => {
    setSelectedMeal(meal);
    setSingleMealAdvice('');
    setSingleModalVisible(true);
  };

  const handleGenerateMealAdvice = async () => {
    if (!selectedMeal) return;
    setGeneratingSingleAi(true);
    const advice = await generateSingleMealAdvice(selectedMeal);
    setSingleMealAdvice(advice);
    setGeneratingSingleAi(false);
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" />
      
      {/* Top Header */}
      <View style={styles.topBar}>
        <Text style={styles.headerTitle}>Dietary Monitoring</Text>
        <TouchableOpacity style={styles.weeklyNavButton} onPress={openWeeklyInsights}>
          <Text style={styles.weeklyNavButtonText}>📊 Weekly Insights</Text>
        </TouchableOpacity>
      </View>

      {/* Multi-Day General AI Box */}
      <View style={styles.aiContainer}>
        <View style={styles.aiHeader}>
          <Text style={styles.aiTitle}>✨ Overall Diet Analysis</Text>
          <TouchableOpacity 
            style={[styles.aiButton, generatingAi && { opacity: 0.7 }]} 
            onPress={handleGenerateGeneralAdvice}
            disabled={generatingAi}
          >
            {generatingAi ? (
              <ActivityIndicator size="small" color="#ffffff" />
            ) : (
              <Text style={styles.aiButtonText}>Analyze All</Text>
            )}
          </TouchableOpacity>
        </View>

        {aiAdvice ? (
          <ScrollView style={styles.adviceScroll}>
            <Text style={styles.aiAdviceText}>{aiAdvice}</Text>
          </ScrollView>
        ) : (
          <Text style={styles.aiPlaceholder}>
            Tap "Analyze All" to let Gemini evaluate your multi-day nutritional patterns.
          </Text>
        )}
      </View>

      {/* Grouped Clickable Meal List */}
      {loading ? (
        <ActivityIndicator size="large" color="#10b981" style={{ marginTop: 20 }} />
      ) : sections.length === 0 ? (
        <View style={styles.emptyContainer}>
          <Text style={styles.emptyText}>No meal records found.</Text>
        </View>
      ) : (
        <SectionList
          sections={sections}
          keyExtractor={(item) => item.id}
          contentContainerStyle={{ paddingBottom: 30 }}
          renderSectionHeader={({ section: { title } }) => (
            <View style={styles.sectionHeaderContainer}>
              <Text style={styles.sectionHeaderText}>{title}</Text>
              <View style={styles.sectionHeaderDivider} />
            </View>
          )}
          renderItem={({ item }) => (
            <TouchableOpacity 
              activeOpacity={0.7} 
              style={styles.card}
              onPress={() => openMealDetails(item)}
            >
              <View style={styles.cardHeader}>
                <View style={{ flex: 1 }}>
                  <Text style={styles.mealType}>{item.mealType || 'Meal Log'}</Text>
                  <Text style={styles.timestampText}>🕒 {formatTimeOnly(item.timestamp || item.createdAt)} • Tap for details</Text>
                </View>
                <Text style={styles.caloriesBadge}>{item.totalCalories ?? 0} kcal</Text>
              </View>

              <Text style={styles.weightText}>Measured Mass: {item.totalWeight_g ?? 0} g</Text>

              <View style={styles.macroRow}>
                <Text style={styles.macro}>🥩 Protein: {item.totalProtein ?? 0}g</Text>
                <Text style={styles.macro}>🍞 Carbs: {item.totalCarbs ?? 0}g</Text>
                <Text style={styles.macro}>🥑 Fat: {item.totalFat ?? 0}g</Text>
              </View>
            </TouchableOpacity>
          )}
        />
      )}

      {/* MODAL 1: WEEKLY DIETARY INSIGHTS */}
      <Modal
        visible={weeklyModalVisible}
        animationType="slide"
        presentationStyle="pageSheet"
        onRequestClose={() => setWeeklyModalVisible(false)}
      >
        <SafeAreaView style={styles.modalContainer}>
          <ScrollView contentContainerStyle={{ padding: 18 }}>
            
            {/* Header Banner */}
            <View style={styles.weeklyHeaderBanner}>
              <Text style={styles.weeklyBannerTitle}>WEEKLY DIETARY INSIGHTS</Text>
              <TouchableOpacity onPress={() => setWeeklyModalVisible(false)} style={styles.closeRoundBtn}>
                <Text style={styles.closeRoundBtnText}>✕</Text>
              </TouchableOpacity>
            </View>

            {weeklyStats ? (
              <>
                {/* 2-Column Row 1: Average Daily Intake & Calories Trend */}
                <View style={styles.gridRow}>
                  {/* Average Daily Intake */}
                  <View style={[styles.gridCard, { flex: 1, marginRight: 8 }]}>
                    <Text style={styles.gridCardTitle}>AVERAGE DAILY INTAKE</Text>
                    <View style={styles.intakeItem}>
                      <Text style={styles.intakeLabel}>🔥 Calories</Text>
                      <Text style={styles.intakeVal}>{weeklyStats.avgCalories} kcal</Text>
                    </View>
                    <View style={styles.intakeItem}>
                      <Text style={styles.intakeLabel}>🥩 Protein</Text>
                      <Text style={styles.intakeVal}>{weeklyStats.avgProtein} g</Text>
                    </View>
                    <View style={styles.intakeItem}>
                      <Text style={styles.intakeLabel}>🌾 Carbohydrates</Text>
                      <Text style={styles.intakeVal}>{weeklyStats.avgCarbs} g</Text>
                    </View>
                    <View style={styles.intakeItem}>
                      <Text style={styles.intakeLabel}>🫒 Fat</Text>
                      <Text style={styles.intakeVal}>{weeklyStats.avgFat} g</Text>
                    </View>
                  </View>

                  {/* Calories Trend */}
                  <View style={[styles.gridCard, { flex: 1, marginLeft: 8 }]}>
                    <Text style={styles.gridCardTitle}>CALORIES TREND</Text>
                    <View style={styles.trendBarsContainer}>
                      {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].map((day) => {
                        const val = weeklyStats.dayCalories[day] || 0;
                        const barHeight = Math.min(Math.max((val / 2500) * 60, 4), 60);
                        return (
                          <View key={day} style={styles.trendCol}>
                            <View style={[styles.trendBar, { height: barHeight }]} />
                            <Text style={styles.trendDayText}>{day[0]}</Text>
                          </View>
                        );
                      })}
                    </View>
                    <Text style={styles.trendAvgNote}>Daily Avg: {weeklyStats.avgCalories} kcal</Text>
                  </View>
                </View>

                {/* 2-Column Row 2: Macronutrient Distribution & Top Foods */}
                <View style={styles.gridRow}>
                  {/* Macronutrient Distribution */}
                  <View style={[styles.gridCard, { flex: 1, marginRight: 8 }]}>
                    <Text style={styles.gridCardTitle}>MACRONUTRIENT DISTRIBUTION</Text>
                    <View style={styles.macroBarWrapper}>
                      <View style={[styles.macroBarFill, { flex: Math.max(weeklyStats.carbsPct, 1), backgroundColor: '#2563eb' }]} />
                      <View style={[styles.macroBarFill, { flex: Math.max(weeklyStats.proteinPct, 1), backgroundColor: '#16a34a' }]} />
                      <View style={[styles.macroBarFill, { flex: Math.max(weeklyStats.fatPct, 1), backgroundColor: '#d97706' }]} />
                    </View>
                    <View style={styles.macroLegendRow}>
                      <View style={[styles.dot, { backgroundColor: '#2563eb' }]} />
                      <Text style={styles.legendText}>Carbs {weeklyStats.carbsPct}%</Text>
                    </View>
                    <View style={styles.macroLegendRow}>
                      <View style={[styles.dot, { backgroundColor: '#16a34a' }]} />
                      <Text style={styles.legendText}>Protein {weeklyStats.proteinPct}%</Text>
                    </View>
                    <View style={styles.macroLegendRow}>
                      <View style={[styles.dot, { backgroundColor: '#d97706' }]} />
                      <Text style={styles.legendText}>Fat {weeklyStats.fatPct}%</Text>
                    </View>
                  </View>

                  {/* Top Foods This Week */}
                  <View style={[styles.gridCard, { flex: 1, marginLeft: 8 }]}>
                    <Text style={styles.gridCardTitle}>TOP FOODS THIS WEEK</Text>
                    {weeklyStats.topFoods.map((food, idx) => (
                      <Text key={idx} style={styles.topFoodItem} numberOfLines={1}>
                        {idx + 1}. {food}
                      </Text>
                    ))}
                  </View>
                </View>

                {/* Recommendation Box */}
                <View style={styles.recommendationCard}>
                  <View style={styles.recomHeader}>
                    <Text style={styles.recomIcon}>💡</Text>
                    <Text style={styles.recomTitle}>RECOMMENDATION</Text>
                  </View>
                  {loadingWeeklyAdvice ? (
                    <ActivityIndicator size="small" color="#047857" style={{ marginVertical: 6 }} />
                  ) : (
                    <Text style={styles.recomText}>
                      {weeklyAdvice || "Maintain your protein intake and try to include more vegetables and fiber-rich foods."}
                    </Text>
                  )}
                </View>
              </>
            ) : (
              <Text style={{ textAlign: 'center', marginTop: 40, color: '#6b7280' }}>
                Log meals to populate your weekly analytics.
              </Text>
            )}

            <TouchableOpacity style={styles.doneBtn} onPress={() => setWeeklyModalVisible(false)}>
              <Text style={styles.doneBtnText}>Close Insights</Text>
            </TouchableOpacity>
          </ScrollView>
        </SafeAreaView>
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
              </Text>

              <Text style={styles.modalSectionHeading}>Nutrition & Mass Breakdown</Text>
              <View style={styles.metricsGrid}>
                <View style={[styles.metricBox, { backgroundColor: '#ecfdf5', borderColor: '#a7f3d0' }]}>
                  <Text style={styles.metricLabel}>Total Energy</Text>
                  <Text style={[styles.metricValue, { color: '#059669' }]}>{selectedMeal.totalCalories ?? 0} kcal</Text>
                </View>
                <View style={[styles.metricBox, { backgroundColor: '#f0f9ff', borderColor: '#bae6fd' }]}>
                  <Text style={styles.metricLabel}>Portion Mass</Text>
                  <Text style={[styles.metricValue, { color: '#0284c7' }]}>{selectedMeal.totalWeight_g ?? 0} g</Text>
                </View>
              </View>

              <View style={styles.macroDetailedContainer}>
                <View style={styles.macroDetailRow}>
                  <Text style={styles.macroDetailName}>🥩 Dietary Protein</Text>
                  <Text style={styles.macroDetailVal}>{selectedMeal.totalProtein ?? 0} g</Text>
                </View>
                <View style={styles.macroDetailRow}>
                  <Text style={styles.macroDetailName}>🍞 Carbohydrates</Text>
                  <Text style={styles.macroDetailVal}>{selectedMeal.totalCarbs ?? 0} g</Text>
                </View>
                <View style={styles.macroDetailRow}>
                  <Text style={styles.macroDetailName}>🥑 Total Fats</Text>
                  <Text style={styles.macroDetailVal}>{selectedMeal.totalFat ?? 0} g</Text>
                </View>
              </View>

              {/* Single Meal AI Advisor */}
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

              <TouchableOpacity style={styles.doneBtn} onPress={() => setSingleModalVisible(false)}>
                <Text style={styles.doneBtnText}>Done</Text>
              </TouchableOpacity>
            </ScrollView>
          )}
        </SafeAreaView>
      </Modal>

    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f3f4f6', paddingHorizontal: 16, paddingTop: 40 },
  topBar: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 },
  headerTitle: { fontSize: 20, fontWeight: 'bold', color: '#111827' },
  weeklyNavButton: { backgroundColor: '#1e3a8a', paddingVertical: 8, paddingHorizontal: 12, borderRadius: 10 },
  weeklyNavButtonText: { color: '#ffffff', fontWeight: 'bold', fontSize: 13 },

  // AI Assistant Card
  aiContainer: { backgroundColor: '#ffffff', borderRadius: 14, padding: 16, elevation: 3, marginBottom: 12, borderWidth: 1, borderColor: '#e0e7ff' },
  aiHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 },
  aiTitle: { fontSize: 16, fontWeight: 'bold', color: '#4338ca' },
  aiButton: { backgroundColor: '#4f46e5', paddingVertical: 6, paddingHorizontal: 12, borderRadius: 8 },
  aiButtonText: { color: '#ffffff', fontWeight: '600', fontSize: 13 },
  aiPlaceholder: { fontSize: 13, color: '#6b7280', fontStyle: 'italic', lineHeight: 18 },
  adviceScroll: { maxHeight: 120, marginTop: 4 },
  aiAdviceText: { fontSize: 13, color: '#1f2937', lineHeight: 19 },

  // Grouped Header
  sectionHeaderContainer: { flexDirection: 'row', alignItems: 'center', marginTop: 16, marginBottom: 10 },
  sectionHeaderText: { fontSize: 19, fontWeight: '800', color: '#1e293b', marginRight: 10 },
  sectionHeaderDivider: { flex: 1, height: 1.5, backgroundColor: '#cbd5e1' },

  // List Cards
  emptyContainer: { alignItems: 'center', marginTop: 30, paddingHorizontal: 20 },
  emptyText: { fontSize: 16, fontWeight: '600', color: '#374151' },
  card: { backgroundColor: '#ffffff', padding: 14, borderRadius: 12, marginBottom: 10, elevation: 2 },
  cardHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 6 },
  mealType: { fontSize: 16, fontWeight: '700', color: '#1f2937' },
  timestampText: { fontSize: 12, color: '#64748b', marginTop: 3, fontWeight: '500' },
  caloriesBadge: { fontSize: 15, fontWeight: 'bold', color: '#059669' },
  weightText: { fontSize: 13, color: '#6b7280', marginVertical: 6 },
  macroRow: { flexDirection: 'row', justifyContent: 'space-between', borderTopWidth: 1, borderTopColor: '#f3f4f6', paddingTop: 8 },
  macro: { fontSize: 13, fontWeight: '500', color: '#374151' },

  // Modal Base
  modalContainer: { flex: 1, backgroundColor: '#ffffff' },
  modalHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  modalTitle: { fontSize: 20, fontWeight: '800', color: '#111827', flex: 1, marginRight: 10 },
  closeRoundBtn: { width: 28, height: 28, borderRadius: 14, backgroundColor: '#e2e8f0', alignItems: 'center', justifyContent: 'center' },
  closeRoundBtnText: { fontSize: 13, fontWeight: 'bold', color: '#475569' },
  modalSubTime: { fontSize: 13, color: '#6b7280', marginTop: 4, marginBottom: 16 },
  modalSectionHeading: { fontSize: 16, fontWeight: '700', color: '#374151', marginBottom: 10 },

  // WEEKLY INSIGHTS DESIGN STYLES
  weeklyHeaderBanner: { 
    backgroundColor: '#0c3875', 
    borderRadius: 8, 
    paddingVertical: 12, 
    paddingHorizontal: 14, 
    flexDirection: 'row', 
    alignItems: 'center', 
    justifyContent: 'space-between',
    marginBottom: 14 
  },
  weeklyBannerTitle: { color: '#ffffff', fontWeight: '900', fontSize: 16, letterSpacing: 0.5 },

  gridRow: { flexDirection: 'row', marginVertical: 6 },
  gridCard: { 
    backgroundColor: '#ffffff', 
    borderRadius: 12, 
    padding: 12, 
    borderWidth: 1, 
    borderColor: '#cbd5e1', 
    minHeight: 140 
  },
  gridCardTitle: { fontSize: 10, fontWeight: '800', color: '#1e293b', marginBottom: 8, letterSpacing: 0.4 },
  
  intakeItem: { flexDirection: 'row', justifyContent: 'space-between', marginVertical: 2 },
  intakeLabel: { fontSize: 11, color: '#475569', fontWeight: '500' },
  intakeVal: { fontSize: 11, fontWeight: '700', color: '#0f172a' },

  trendBarsContainer: { flexDirection: 'row', alignItems: 'flex-end', justifyContent: 'space-between', height: 68, marginTop: 4 },
  trendCol: { alignItems: 'center' },
  trendBar: { width: 10, backgroundColor: '#10b981', borderRadius: 3 },
  trendDayText: { fontSize: 9, color: '#64748b', marginTop: 4, fontWeight: '600' },
  trendAvgNote: { fontSize: 10, color: '#64748b', textAlign: 'center', marginTop: 6, fontStyle: 'italic' },

  macroBarWrapper: { flexDirection: 'row', height: 10, borderRadius: 5, overflow: 'hidden', marginVertical: 8 },
  macroBarFill: { height: '100%' },
  macroLegendRow: { flexDirection: 'row', alignItems: 'center', marginVertical: 2 },
  dot: { width: 8, height: 8, borderRadius: 4, marginRight: 6 },
  legendText: { fontSize: 11, color: '#334155', fontWeight: '600' },

  topFoodItem: { fontSize: 12, color: '#1e293b', fontWeight: '600', marginVertical: 4 },

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