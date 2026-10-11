// geminiService.js
const GEMINI_API_KEY = ""; // Keep your active Gemini API key inside quotes

// 1. Overall Multi-Day / General Diet Analysis
export async function generateDietaryAdvice(meals) {
  if (!meals || meals.length === 0) {
    return "Please log at least one meal to receive AI dietary recommendations.";
  }

  const mealSummary = meals.map((m, idx) => {
    const timeStr = m.timestamp?.toDate 
      ? m.timestamp.toDate().toLocaleString() 
      : (m.timestamp || 'Logged recently');
    return `- Entry ${idx + 1} [${timeStr}] (${m.mealType || 'Meal'}): ${m.totalCalories ?? 0} kcal | Protein: ${m.totalProtein ?? 0}g | Carbs: ${m.totalCarbs ?? 0}g | Fat: ${m.totalFat ?? 0}g | Mass: ${m.totalWeight_g ?? 0}g`;
  }).join('\n');

  const prompt = `You are an expert AI clinical nutritionist. Analyze the following logged meal history with dates & times:

${mealSummary}

Provide:
1. A quick 1-sentence evaluation of overall calorie/macro balance and meal timing.
2. 3 short, practical bullet points to improve overall nutrition and health.
Keep your response concise and under 150 words.`;

  return callGemini(prompt);
}

// 2. Single Specific Meal Optimization & Critique
export async function generateSingleMealAdvice(meal) {
  if (!meal) return "No meal data provided.";

  const prompt = `You are an expert clinical dietitian. Analyze this specific logged meal:
- Food: ${meal.mealType || 'Logged Meal'}
- Total Calories: ${meal.totalCalories ?? 0} kcal
- Measured Mass: ${meal.totalWeight_g ?? 0} g
- Protein: ${meal.totalProtein ?? 0} g
- Carbohydrates: ${meal.totalCarbs ?? 0} g
- Dietary Fat: ${meal.totalFat ?? 0} g

Provide:
1. A 1-sentence assessment of this dish's nutritional quality and macro distribution.
2. 2-3 realistic, practical recommendations to make THIS specific meal healthier (e.g., portion tweaks, adding greens/vegetables, cooking adjustments).
Keep the advice direct, helpful, and under 120 words.`;

  return callGemini(prompt);
}

// 3. Weekly Insights Summary & Recommendations
export async function generateWeeklyInsightsAdvice(weeklyData) {
  if (!weeklyData) return "No weekly data available.";

  const prompt = `You are an expert clinical dietitian. Based on this user's 7-day nutritional summary:
- Daily Average Calories: ${weeklyData.avgCalories} kcal
- Daily Average Protein: ${weeklyData.avgProtein} g
- Daily Average Carbs: ${weeklyData.avgCarbs} g
- Daily Average Fat: ${weeklyData.avgFat} g
- Top Foods Eaten: ${weeklyData.topFoods?.join(', ') || 'Mixed meals'}

Provide a concise, direct 1-to-2 sentence recommendation for the week (e.g., maintaining protein, adjusting carbohydrate sources, or adding micronutrient-rich local vegetables). Keep it under 40 words.`;

  return callGemini(prompt);
}

// Helper function to call the Gemini API endpoint
async function callGemini(prompt) {
  try {
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key=${GEMINI_API_KEY}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{ parts: [{ text: prompt }] }],
        }),
      }
    );

    const data = await response.json();
    if (data.candidates && data.candidates[0]?.content?.parts?.[0]?.text) {
      return data.candidates[0].content.parts[0].text;
    } else if (data.error) {
      return `API Error: ${data.error.message}`;
    }
    return "Could not generate insights at this time.";
  } catch (error) {
    console.error("Gemini Request Error:", error);
    return "Error connecting to AI service. Please verify your network.";
  }
}