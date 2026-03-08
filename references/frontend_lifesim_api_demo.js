/**
 * Frontend demo: call backend APIs that wrap FastMCP tools.
 *
 * Assumption:
 * - Your backend gateway exposes HTTP endpoints:
 *   POST /api/lifesim/generate-life-events
 *   POST /api/lifesim/generate-event-dialogues
 */

const API_BASE = "http://127.0.0.1:8080/api/lifesim";

export async function generateLifeEvents(payload) {
  const resp = await fetch(`${API_BASE}/generate-life-events`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!resp.ok) throw new Error(`generate-life-events failed: ${resp.status}`);
  return await resp.json();
}

export async function generateEventDialogues(payload) {
  const resp = await fetch(`${API_BASE}/generate-event-dialogues`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!resp.ok) throw new Error(`generate-event-dialogues failed: ${resp.status}`);
  return await resp.json();
}

// End-to-end example
export async function buildLifeSimPackage() {
  const userProfile = {
    user_id: "u_001",
    age: 30,
    gender: "Male",
    area: "New York",
    employment: "Software Engineer",
    marital: "Single",
    income: "Middle",
    race: "Asian",
    religious: "No religion",
    personality: ["Introverted", "Analytical"],
    preferences: ["Technology", "Running"],
    bigfive: {},
    preferences_value: [],
  };

  const eventsResult = await generateLifeEvents({
    sequence_id: "NYC_entertainment_001",
    user_profile: userProfile,
    expected_hours: 12,
    start_event_index: 0,
    max_events: 5,
    history_events: [],
    goal: "Generate realistic short-horizon events with clear user intent.",
  });

  const dialoguesResult = await generateEventDialogues({
    user_profile: userProfile,
    event_experiences: eventsResult.nodes,
    beliefs: [],
    max_turns: 6,
    refine_intention_enabled: true,
  });

  return {
    events: eventsResult,
    dialogues: dialoguesResult,
  };
}
