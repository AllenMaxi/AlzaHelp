import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Activity, Brain, Loader2, MoonStar, Plus, Smile } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { moodApi } from "@/services/api";
import { toast } from "sonner";

export const MoodBehaviorSection = ({ targetUserId = null }) => {
  const [loading, setLoading] = useState(true);
  const [savingMood, setSavingMood] = useState(false);
  const [savingObservation, setSavingObservation] = useState(false);
  const [taxonomy, setTaxonomy] = useState({ symptoms: [], time_of_day: [] });
  const [analytics, setAnalytics] = useState(null);
  const [checkins, setCheckins] = useState([]);
  const [observations, setObservations] = useState([]);
  const [moodForm, setMoodForm] = useState({
    mood_score: 3,
    energy_score: 3,
    anxiety_score: 3,
    sleep_quality: 3,
    appetite: "normal",
    notes: ""
  });
  const [observationForm, setObservationForm] = useState({
    symptom: "agitation",
    severity: 3,
    time_of_day: "evening",
    duration_minutes: "",
    trigger_tags_csv: "",
    notes: ""
  });

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const [taxonomyRes, analyticsRes, checkinsRes, observationsRes] = await Promise.all([
        moodApi.getTaxonomy().catch(() => ({ symptoms: [], time_of_day: [] })),
        moodApi.getAnalytics(30, targetUserId).catch(() => ({})),
        moodApi.getCheckins(30, targetUserId, 30).catch(() => []),
        moodApi.getObservations(30, targetUserId, 40).catch(() => [])
      ]);
      setTaxonomy(taxonomyRes || { symptoms: [], time_of_day: [] });
      setAnalytics(analyticsRes || {});
      setCheckins(checkinsRes || []);
      setObservations(observationsRes || []);
      if ((taxonomyRes?.symptoms || []).length > 0 && !observationForm.symptom) {
        setObservationForm((prev) => ({ ...prev, symptom: taxonomyRes.symptoms[0] }));
      }
    } finally {
      setLoading(false);
    }
  }, [targetUserId, observationForm.symptom]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  useEffect(() => {
    const timer = setInterval(() => {
      loadData();
    }, 90000);
    return () => clearInterval(timer);
  }, [loadData]);

  const submitMoodCheckin = async () => {
    setSavingMood(true);
    try {
      await moodApi.createCheckin(
        {
          mood_score: Number(moodForm.mood_score),
          energy_score: Number(moodForm.energy_score),
          anxiety_score: Number(moodForm.anxiety_score),
          sleep_quality: Number(moodForm.sleep_quality),
          appetite: moodForm.appetite,
          notes: moodForm.notes.trim() || null
        },
        targetUserId
      );
      toast.success("Mood check-in saved.");
      setMoodForm((prev) => ({ ...prev, notes: "" }));
      loadData();
    } catch (error) {
      toast.error(error.message || "Could not save mood check-in.");
    } finally {
      setSavingMood(false);
    }
  };

  const submitObservation = async () => {
    if (!observationForm.symptom) {
      toast.error("Please choose a symptom.");
      return;
    }
    setSavingObservation(true);
    try {
      await moodApi.createObservation(
        {
          symptom: observationForm.symptom,
          severity: Number(observationForm.severity),
          time_of_day: observationForm.time_of_day,
          duration_minutes: observationForm.duration_minutes
            ? Number(observationForm.duration_minutes)
            : null,
          trigger_tags: observationForm.trigger_tags_csv
            .split(",")
            .map((item) => item.trim())
            .filter(Boolean),
          notes: observationForm.notes.trim() || null
        },
        targetUserId
      );
      toast.success("Behavior event logged.");
      setObservationForm((prev) => ({
        ...prev,
        duration_minutes: "",
        trigger_tags_csv: "",
        notes: ""
      }));
      loadData();
    } catch (error) {
      toast.error(error.message || "Could not save behavior event.");
    } finally {
      setSavingObservation(false);
    }
  };

  const topSymptomLabel = useMemo(() => {
    const value = analytics?.top_symptom || "";
    return value ? value.replace(/_/g, " ") : "None";
  }, [analytics]);

  return (
    <section className="py-8 sm:py-12">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 rounded-full bg-primary/10 px-4 py-2 mb-4">
            <Brain className="h-5 w-5 text-primary" />
            <span className="text-base font-medium text-primary">Mood & Behavior</span>
          </div>
          <h2 className="font-display text-3xl sm:text-4xl font-bold text-foreground mb-2">
            BPSD Tracker
          </h2>
          <p className="text-accessible text-muted-foreground">
            Track mood, behavior symptoms, and weekly patterns for earlier intervention.
          </p>
        </div>

        <div className="grid gap-4 md:grid-cols-4 mb-8">
          <Card className="border-2 border-border">
            <CardHeader>
              <CardTitle className="text-sm">Average Mood</CardTitle>
            </CardHeader>
            <CardContent>
              {loading ? <Loader2 className="h-5 w-5 animate-spin text-primary" /> : <p className="text-3xl font-bold">{analytics?.average_mood_score ?? 0}</p>}
            </CardContent>
          </Card>
          <Card className="border-2 border-border">
            <CardHeader>
              <CardTitle className="text-sm">Low Mood Days</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-3xl font-bold text-amber-600">{analytics?.low_mood_days ?? 0}</p>
            </CardContent>
          </Card>
          <Card className="border-2 border-border">
            <CardHeader>
              <CardTitle className="text-sm">BPSD Events (30d)</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-3xl font-bold">{analytics?.total_observations ?? 0}</p>
            </CardContent>
          </Card>
          <Card className="border-2 border-border">
            <CardHeader>
              <CardTitle className="text-sm">Top Symptom</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-lg font-semibold capitalize">{topSymptomLabel}</p>
            </CardContent>
          </Card>
        </div>

        <div className="grid gap-4 lg:grid-cols-2 mb-8">
          <Card className="border-2 border-border">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Smile className="h-5 w-5 text-primary" />
                Daily Mood Check-in
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <Label>Mood (1-5)</Label>
                  <Input
                    type="number"
                    min="1"
                    max="5"
                    value={moodForm.mood_score}
                    onChange={(e) => setMoodForm((prev) => ({ ...prev, mood_score: e.target.value }))}
                  />
                </div>
                <div>
                  <Label>Energy (1-5)</Label>
                  <Input
                    type="number"
                    min="1"
                    max="5"
                    value={moodForm.energy_score}
                    onChange={(e) => setMoodForm((prev) => ({ ...prev, energy_score: e.target.value }))}
                  />
                </div>
                <div>
                  <Label>Anxiety (1-5)</Label>
                  <Input
                    type="number"
                    min="1"
                    max="5"
                    value={moodForm.anxiety_score}
                    onChange={(e) => setMoodForm((prev) => ({ ...prev, anxiety_score: e.target.value }))}
                  />
                </div>
                <div>
                  <Label>Sleep quality (1-5)</Label>
                  <Input
                    type="number"
                    min="1"
                    max="5"
                    value={moodForm.sleep_quality}
                    onChange={(e) => setMoodForm((prev) => ({ ...prev, sleep_quality: e.target.value }))}
                  />
                </div>
              </div>
              <div>
                <Label>Appetite</Label>
                <select
                  className="w-full mt-2 rounded-md border border-input bg-background px-3 py-2"
                  value={moodForm.appetite}
                  onChange={(e) => setMoodForm((prev) => ({ ...prev, appetite: e.target.value }))}
                >
                  <option value="low">Low</option>
                  <option value="normal">Normal</option>
                  <option value="high">High</option>
                </select>
              </div>
              <Textarea
                placeholder="Optional notes"
                value={moodForm.notes}
                onChange={(e) => setMoodForm((prev) => ({ ...prev, notes: e.target.value }))}
              />
              <Button onClick={submitMoodCheckin} disabled={savingMood} className="gap-2">
                {savingMood ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
                Save Mood Check-in
              </Button>
            </CardContent>
          </Card>

          <Card className="border-2 border-border">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5 text-primary" />
                Behavior Observation
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div>
                <Label>Symptom</Label>
                <select
                  className="w-full mt-2 rounded-md border border-input bg-background px-3 py-2"
                  value={observationForm.symptom}
                  onChange={(e) => setObservationForm((prev) => ({ ...prev, symptom: e.target.value }))}
                >
                  {(taxonomy.symptoms || []).map((symptom) => (
                    <option key={symptom} value={symptom}>{symptom.replace(/_/g, " ")}</option>
                  ))}
                </select>
              </div>
              <div className="grid grid-cols-3 gap-2">
                <div>
                  <Label>Severity (1-5)</Label>
                  <Input
                    type="number"
                    min="1"
                    max="5"
                    value={observationForm.severity}
                    onChange={(e) => setObservationForm((prev) => ({ ...prev, severity: e.target.value }))}
                  />
                </div>
                <div>
                  <Label>Time of day</Label>
                  <select
                    className="w-full mt-2 rounded-md border border-input bg-background px-3 py-2"
                    value={observationForm.time_of_day}
                    onChange={(e) => setObservationForm((prev) => ({ ...prev, time_of_day: e.target.value }))}
                  >
                    {(taxonomy.time_of_day || []).map((slot) => (
                      <option key={slot} value={slot}>{slot}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <Label>Duration (min)</Label>
                  <Input
                    type="number"
                    min="1"
                    max="600"
                    value={observationForm.duration_minutes}
                    onChange={(e) => setObservationForm((prev) => ({ ...prev, duration_minutes: e.target.value }))}
                  />
                </div>
              </div>
              <Input
                placeholder="Trigger tags (comma-separated)"
                value={observationForm.trigger_tags_csv}
                onChange={(e) => setObservationForm((prev) => ({ ...prev, trigger_tags_csv: e.target.value }))}
              />
              <Textarea
                placeholder="Observation notes"
                value={observationForm.notes}
                onChange={(e) => setObservationForm((prev) => ({ ...prev, notes: e.target.value }))}
              />
              <Button onClick={submitObservation} disabled={savingObservation} className="gap-2">
                {savingObservation ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
                Save Behavior Event
              </Button>
            </CardContent>
          </Card>
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          <Card className="border-2 border-border">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <MoonStar className="h-5 w-5 text-primary" />
                Pattern Insights
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {(analytics?.pattern_insights || []).length === 0 ? (
                <p className="text-sm text-muted-foreground">No patterns detected yet. Keep logging observations.</p>
              ) : (
                (analytics?.pattern_insights || []).map((insight, index) => (
                  <div key={`${insight}-${index}`} className="rounded-lg border p-3 text-sm">
                    {insight}
                  </div>
                ))
              )}
            </CardContent>
          </Card>

          <Card className="border-2 border-border">
            <CardHeader>
              <CardTitle>Recent Events</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div>
                <p className="text-sm font-semibold mb-2">Mood Check-ins</p>
                {(checkins || []).slice(0, 5).map((checkin) => (
                  <div key={checkin.id} className="rounded-lg border p-2 mb-2">
                    <div className="flex items-center justify-between">
                      <p className="text-sm font-medium">
                        Mood {checkin.mood_score}/5
                      </p>
                      <Badge variant="secondary">{checkin.appetite || "normal"}</Badge>
                    </div>
                    <p className="text-xs text-muted-foreground">{new Date(checkin.created_at).toLocaleString()}</p>
                  </div>
                ))}
                {(checkins || []).length === 0 && <p className="text-sm text-muted-foreground">No mood check-ins yet.</p>}
              </div>

              <div>
                <p className="text-sm font-semibold mb-2">Behavior Observations</p>
                {(observations || []).slice(0, 6).map((item) => (
                  <div key={item.id} className="rounded-lg border p-2 mb-2">
                    <p className="text-sm font-medium capitalize">
                      {item.symptom?.replace(/_/g, " ")} (sev {item.severity || 3}/5)
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {item.time_of_day} · {new Date(item.observed_at).toLocaleString()}
                    </p>
                  </div>
                ))}
                {(observations || []).length === 0 && <p className="text-sm text-muted-foreground">No behavior observations yet.</p>}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  );
};

export default MoodBehaviorSection;
