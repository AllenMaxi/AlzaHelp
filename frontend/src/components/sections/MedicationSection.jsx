import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  CalendarClock,
  Check,
  Clock,
  FileText,
  Loader2,
  Pill,
  Plus,
  Stethoscope,
  Trash2,
  Volume2,
  VolumeX
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { careInstructionsApi, medicationsApi } from "@/services/api";
import { toast } from "sonner";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

export const MedicationSection = ({ medications = [], onRefresh, loading, targetUserId = null }) => {
  const [dialogOpen, setDialogOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(null);
  const [adherence, setAdherence] = useState(null);
  const [missedDoses, setMissedDoses] = useState([]);
  const [interactions, setInteractions] = useState([]);
  const [careInstructions, setCareInstructions] = useState([]);
  const [instructionsLoading, setInstructionsLoading] = useState(false);
  const [statsLoading, setStatsLoading] = useState(false);
  const [marLoading, setMarLoading] = useState(false);
  const [marDate, setMarDate] = useState(() => new Date().toISOString().slice(0, 10));
  const [marData, setMarData] = useState({ slots: [], summary: {} });
  const [todayPlan, setTodayPlan] = useState(null);
  const [planLoading, setPlanLoading] = useState(false);
  const [isReadingPlan, setIsReadingPlan] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const planAudioRef = useRef(null);

  const [newMedication, setNewMedication] = useState({
    name: "",
    dosage: "",
    frequency: "daily",
    times_per_day: "1",
    scheduled_times: "08:00",
    prescribing_doctor: "",
    instructions: ""
  });

  const loadStats = useCallback(async () => {
    setStatsLoading(true);
    try {
      const [adherenceRes, missedRes, interactionRes] = await Promise.all([
        medicationsApi.getAdherence(7, targetUserId).catch(() => null),
        medicationsApi.getMissed(2, targetUserId).catch(() => []),
        medicationsApi.getInteractions(targetUserId).catch(() => ({ warnings: [] }))
      ]);
      setAdherence(adherenceRes);
      setMissedDoses(missedRes || []);
      setInteractions(interactionRes?.warnings || []);
    } finally {
      setStatsLoading(false);
    }
  }, [targetUserId]);

  const loadCareInstructions = useCallback(async () => {
    setInstructionsLoading(true);
    try {
      const data = await careInstructionsApi.getAll(targetUserId, true).catch(() => []);
      setCareInstructions(data || []);
    } finally {
      setInstructionsLoading(false);
    }
  }, [targetUserId]);

  const loadMAR = useCallback(async () => {
    setMarLoading(true);
    try {
      const result = await medicationsApi.getMAR(marDate, targetUserId, 1).catch(() => ({ slots: [], summary: {} }));
      setMarData(result || { slots: [], summary: {} });
    } finally {
      setMarLoading(false);
    }
  }, [marDate, targetUserId]);

  const loadTodayPlan = useCallback(async () => {
    setPlanLoading(true);
    try {
      const data = await careInstructionsApi.getTodayPlan(targetUserId, marDate).catch(() => null);
      setTodayPlan(data);
    } finally {
      setPlanLoading(false);
    }
  }, [marDate, targetUserId]);

  const stopPlanAudio = useCallback(() => {
    if (planAudioRef.current) {
      planAudioRef.current.pause();
      planAudioRef.current.src = "";
      planAudioRef.current = null;
    }
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    setIsReadingPlan(false);
  }, []);

  const speakTodayPlan = useCallback(async () => {
    if (!voiceEnabled) {
      toast.error("Enable voice playback first.");
      return;
    }
    const script = todayPlan?.voice_script;
    if (!script) {
      toast.error("No plan is available to read aloud.");
      return;
    }

    stopPlanAudio();
    setIsReadingPlan(true);
    const finish = () => setIsReadingPlan(false);

    try {
      if (BACKEND_URL) {
        const res = await fetch(`${BACKEND_URL}/api/tts`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "include",
          body: JSON.stringify({ text: script, voice: "nova" })
        });
        if (res.ok) {
          const data = await res.json();
          const audio = new Audio(`data:audio/mp3;base64,${data.audio}`);
          planAudioRef.current = audio;
          audio.onended = finish;
          audio.onerror = finish;
          await audio.play();
          return;
        }
      }
    } catch (error) {
      // Fall back to browser speech synthesis below.
    }

    if ("speechSynthesis" in window) {
      const utterance = new SpeechSynthesisUtterance(script);
      utterance.rate = 0.9;
      utterance.onend = finish;
      utterance.onerror = finish;
      window.speechSynthesis.speak(utterance);
      return;
    }

    finish();
    toast.error("Voice playback is not available on this device.");
  }, [todayPlan, stopPlanAudio, voiceEnabled]);

  useEffect(() => {
    loadStats();
  }, [loadStats]);

  useEffect(() => {
    loadCareInstructions();
  }, [loadCareInstructions]);

  useEffect(() => {
    loadStats();
    loadMAR();
    loadTodayPlan();
  }, [medications, loadStats, loadMAR, loadTodayPlan]);

  useEffect(() => {
    const timer = setInterval(() => {
      loadStats();
      loadCareInstructions();
      loadMAR();
      loadTodayPlan();
    }, 60000);
    return () => clearInterval(timer);
  }, [loadStats, loadCareInstructions, loadMAR, loadTodayPlan]);

  useEffect(() => () => stopPlanAudio(), [stopPlanAudio]);

  const addMedication = async () => {
    if (!newMedication.name.trim() || !newMedication.dosage.trim()) {
      toast.error("Medication name and dosage are required.");
      return;
    }

    const timesPerDay = Math.max(1, Number(newMedication.times_per_day || 1));
    const scheduledTimes = newMedication.scheduled_times
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean);

    setSaving(true);
    try {
      await medicationsApi.create(
        {
          name: newMedication.name.trim(),
          dosage: newMedication.dosage.trim(),
          frequency: newMedication.frequency,
          times_per_day: timesPerDay,
          scheduled_times: scheduledTimes,
          prescribing_doctor: newMedication.prescribing_doctor.trim() || null,
          instructions: newMedication.instructions.trim() || null
        },
        targetUserId
      );

      toast.success("Medication added.");
      setDialogOpen(false);
      setNewMedication({
        name: "",
        dosage: "",
        frequency: "daily",
        times_per_day: "1",
        scheduled_times: "08:00",
        prescribing_doctor: "",
        instructions: ""
      });
      onRefresh();
    } catch (error) {
      toast.error("Could not add medication.");
    } finally {
      setSaving(false);
    }
  };

  const removeMedication = async (medicationId) => {
    setDeleting(medicationId);
    try {
      await medicationsApi.delete(medicationId, targetUserId);
      toast.success("Medication removed.");
      onRefresh();
    } catch (error) {
      toast.error("Could not remove medication.");
    } finally {
      setDeleting(null);
    }
  };

  const logIntake = async (medicationId, status, scheduledFor = null) => {
    try {
      await medicationsApi.logIntake(medicationId, status, targetUserId, {
        scheduled_for: scheduledFor,
        source: scheduledFor ? "mar" : "manual"
      });
      toast.success(status === "taken" ? "Dose marked as taken." : "Dose marked as missed.");
      loadStats();
      loadMAR();
    } catch (error) {
      toast.error("Could not record intake.");
    }
  };

  const activeMedications = useMemo(
    () => (medications || []).filter((med) => med.active !== false),
    [medications]
  );

  const groupedInstructions = useMemo(() => {
    const groups = {
      daily: [],
      weekly: [],
      as_needed: []
    };
    (careInstructions || []).forEach((inst) => {
      const key = inst.frequency && groups[inst.frequency] ? inst.frequency : "as_needed";
      groups[key].push(inst);
    });
    return groups;
  }, [careInstructions]);

  const marSlots = useMemo(() => marData?.slots || [], [marData]);
  const todayPlanSlots = useMemo(() => todayPlan?.medication_plan?.slots || [], [todayPlan]);
  const todayDueInstructions = useMemo(() => todayPlan?.instructions?.due_today || [], [todayPlan]);
  const todayAsNeededInstructions = useMemo(() => todayPlan?.instructions?.as_needed || [], [todayPlan]);
  const todayRegimenInstructions = useMemo(
    () => todayPlan?.instructions?.active_medication_regimens || [],
    [todayPlan]
  );

  return (
    <section className="py-8 sm:py-12">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 rounded-full bg-primary/10 px-4 py-2 mb-4">
            <Pill className="h-5 w-5 text-primary" />
            <span className="text-base font-medium text-primary">Medication Management</span>
          </div>
          <h2 className="font-display text-3xl sm:text-4xl font-bold text-foreground mb-2">
            Medication Tracker
          </h2>
          <p className="text-accessible text-muted-foreground">
            Keep medications, adherence, missed doses, and interaction warnings in one place.
          </p>
        </div>

        <div className="grid gap-4 lg:grid-cols-4 mb-8">
          <Card className="border-2 border-border">
            <CardHeader>
              <CardTitle className="text-sm">7-day On-time</CardTitle>
            </CardHeader>
            <CardContent>
              {statsLoading ? (
                <Loader2 className="h-5 w-5 animate-spin text-primary" />
              ) : (
                <div>
                  <p className="text-3xl font-bold text-primary">
                    {adherence?.adherence_percent_on_time ?? adherence?.adherence_percent ?? 0}%
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">
                    Total taken: {adherence?.adherence_percent_total ?? 0}%
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="border-2 border-border">
            <CardHeader>
              <CardTitle className="text-sm">Expected Doses</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-3xl font-bold">{adherence?.expected_doses ?? 0}</p>
              <p className="text-xs text-muted-foreground mt-1">
                Pending: {adherence?.pending_doses ?? 0}
              </p>
            </CardContent>
          </Card>

          <Card className="border-2 border-border">
            <CardHeader>
              <CardTitle className="text-sm">Missed (Overdue)</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-3xl font-bold text-destructive">{missedDoses.length}</p>
            </CardContent>
          </Card>

          <Card className="border-2 border-border">
            <CardHeader>
              <CardTitle className="text-sm">Interaction Warnings</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-3xl font-bold text-amber-600">{interactions.length}</p>
            </CardContent>
          </Card>
        </div>

        <Card className="border-2 border-border mb-8">
          <CardHeader>
            <CardTitle className="flex items-center justify-between gap-2 flex-wrap">
              <span className="flex items-center gap-2">
                <FileText className="h-5 w-5 text-primary" />
                Today's Guided Plan
              </span>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  className="gap-2"
                  onClick={loadTodayPlan}
                  disabled={planLoading}
                >
                  {planLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Clock className="h-4 w-4" />}
                  Refresh
                </Button>
                <Button
                  variant={isReadingPlan ? "destructive" : "accessible"}
                  size="sm"
                  className="gap-2"
                  onClick={() => (isReadingPlan ? stopPlanAudio() : speakTodayPlan())}
                  disabled={!todayPlan?.voice_script}
                >
                  {isReadingPlan ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
                  {isReadingPlan ? "Stop Reading" : "Read Aloud"}
                </Button>
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => {
                    if (isReadingPlan) stopPlanAudio();
                    setVoiceEnabled((prev) => !prev);
                  }}
                  title={voiceEnabled ? "Disable voice playback" : "Enable voice playback"}
                >
                  {voiceEnabled ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
                </Button>
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {todayPlan?.date && (
              <p className="text-sm text-muted-foreground">
                Plan date: {todayPlan.date}
              </p>
            )}

            <div className="grid gap-4 lg:grid-cols-2">
              <div className="rounded-lg border p-3">
                <p className="font-semibold mb-2">Medication Schedule</p>
                {planLoading ? (
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Loading schedule...
                  </div>
                ) : todayPlanSlots.length === 0 ? (
                  <p className="text-sm text-muted-foreground">No medication doses scheduled for this day.</p>
                ) : (
                  <div className="space-y-2 max-h-56 overflow-auto pr-1">
                    {todayPlanSlots.slice(0, 12).map((slot) => (
                      <div key={`${slot.medication_id}-${slot.scheduled_for}`} className="rounded border p-2">
                        <p className="text-sm font-medium">
                          {slot.name} ({slot.dosage})
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {new Date(slot.scheduled_for).toLocaleString()} • {String(slot.status || "upcoming").replace(/_/g, " ")}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div className="rounded-lg border p-3">
                <p className="font-semibold mb-2">Doctor Instructions Due Today</p>
                {planLoading ? (
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Loading instructions...
                  </div>
                ) : todayDueInstructions.length === 0 ? (
                  <p className="text-sm text-muted-foreground">No scheduled doctor instructions for this day.</p>
                ) : (
                  <div className="space-y-2 max-h-56 overflow-auto pr-1">
                    {todayDueInstructions.slice(0, 8).map((inst) => (
                      <details key={inst.id} className="rounded border p-2">
                        <summary className="cursor-pointer text-sm font-medium">
                          {inst.title}
                          {inst.time_of_day ? ` • ${inst.time_of_day}` : ""}
                        </summary>
                        <p className="mt-2 text-xs text-muted-foreground whitespace-pre-wrap">
                          {inst.instruction_text || inst.summary || "No details provided."}
                        </p>
                      </details>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div className="grid gap-4 lg:grid-cols-2">
              <div className="rounded-lg border p-3">
                <p className="font-semibold mb-2">As Needed Instructions</p>
                {todayAsNeededInstructions.length === 0 ? (
                  <p className="text-sm text-muted-foreground">No as-needed instructions.</p>
                ) : (
                  <div className="space-y-2">
                    {todayAsNeededInstructions.slice(0, 5).map((inst) => (
                      <div key={inst.id} className="rounded border p-2">
                        <p className="text-sm font-medium">{inst.title}</p>
                        <p className="text-xs text-muted-foreground">
                          {inst.summary || "Open for full text."}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div className="rounded-lg border p-3">
                <p className="font-semibold mb-2">Active Medication Protocols (Signed)</p>
                {todayRegimenInstructions.length === 0 ? (
                  <p className="text-sm text-muted-foreground">No signed medication protocol found.</p>
                ) : (
                  <div className="space-y-2">
                    {todayRegimenInstructions.slice(0, 5).map((inst) => (
                      <div key={inst.id} className="rounded border p-2">
                        <p className="text-sm font-medium">{inst.title}</p>
                        <p className="text-xs text-muted-foreground">
                          v{inst.version} • Signed by {inst.signed_off_by_name || "clinician"}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-2 border-border mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CalendarClock className="h-5 w-5 text-primary" />
              Medication Administration Record (MAR)
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-end gap-3">
              <div>
                <Label htmlFor="mar-date">Date</Label>
                <Input
                  id="mar-date"
                  type="date"
                  value={marDate}
                  onChange={(e) => setMarDate(e.target.value)}
                  className="mt-2"
                />
              </div>
              <Button variant="outline" onClick={loadMAR} disabled={marLoading} className="gap-2">
                {marLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Clock className="h-4 w-4" />}
                Refresh MAR
              </Button>
            </div>

            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-6">
              <div className="rounded-lg border p-2 text-sm">
                Expected: <span className="font-semibold">{marData?.summary?.expected ?? 0}</span>
              </div>
              <div className="rounded-lg border p-2 text-sm">
                On-time: <span className="font-semibold">{marData?.summary?.taken_on_time ?? 0}</span>
              </div>
              <div className="rounded-lg border p-2 text-sm">
                Late: <span className="font-semibold">{marData?.summary?.taken_late ?? 0}</span>
              </div>
              <div className="rounded-lg border p-2 text-sm">
                Due: <span className="font-semibold">{marData?.summary?.due ?? 0}</span>
              </div>
              <div className="rounded-lg border p-2 text-sm text-destructive">
                Overdue: <span className="font-semibold">{marData?.summary?.overdue ?? 0}</span>
              </div>
              <div className="rounded-lg border p-2 text-sm text-destructive">
                Missed: <span className="font-semibold">{marData?.summary?.missed ?? 0}</span>
              </div>
            </div>

            {marLoading ? (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                Loading MAR...
              </div>
            ) : marSlots.length === 0 ? (
              <p className="text-sm text-muted-foreground">No scheduled doses for this date.</p>
            ) : (
              <div className="space-y-2 max-h-[380px] overflow-auto pr-1">
                {marSlots.map((slot) => {
                  const status = slot.status || "upcoming";
                  const statusLabel = status.replace(/_/g, " ");
                  const statusClasses =
                    status === "taken_on_time"
                      ? "bg-green-100 text-green-700 border-green-300"
                      : status === "taken_late"
                        ? "bg-amber-100 text-amber-700 border-amber-300"
                        : status === "due" || status === "upcoming"
                          ? "bg-muted text-muted-foreground border-border"
                          : "bg-destructive/10 text-destructive border-destructive/30";
                  return (
                    <div key={`${slot.medication_id}-${slot.scheduled_for}`} className="rounded-lg border p-3">
                      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-2">
                        <div>
                          <p className="font-medium">
                            {slot.name} ({slot.dosage})
                          </p>
                          <p className="text-xs text-muted-foreground">
                            Scheduled: {new Date(slot.scheduled_for).toLocaleString()}
                          </p>
                          {slot.confirmed_at && (
                            <p className="text-xs text-muted-foreground">
                              Confirmed: {new Date(slot.confirmed_at).toLocaleString()}
                            </p>
                          )}
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge className={statusClasses}>{statusLabel}</Badge>
                          {["due", "overdue", "missed", "upcoming"].includes(status) && (
                            <>
                              <Button
                                size="sm"
                                variant="accessible"
                                className="gap-1"
                                onClick={() => logIntake(slot.medication_id, "taken", slot.scheduled_for)}
                              >
                                <Check className="h-3 w-3" /> Taken
                              </Button>
                              <Button
                                size="sm"
                                variant="outline"
                                className="gap-1"
                                onClick={() => logIntake(slot.medication_id, "missed", slot.scheduled_for)}
                              >
                                <AlertTriangle className="h-3 w-3" /> Missed
                              </Button>
                            </>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="border-2 border-border mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="h-5 w-5 text-primary" />
              Daily & Weekly Care Procedures
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {instructionsLoading ? (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                Loading procedures...
              </div>
            ) : careInstructions.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                No uploaded care procedures yet. Caregivers can add them from the Caregiver Portal.
              </p>
            ) : (
              <div className="grid gap-4 md:grid-cols-3">
                <div className="rounded-lg border p-3">
                  <p className="font-semibold mb-2">Daily</p>
                  {groupedInstructions.daily.length === 0 ? (
                    <p className="text-sm text-muted-foreground">No daily steps.</p>
                  ) : (
                    groupedInstructions.daily.map((inst) => (
                      <div key={inst.id} className="mb-2">
                        <p className="text-sm font-medium">{inst.title}</p>
                        <p className="text-xs text-muted-foreground">{inst.time_of_day || "Any time"}</p>
                      </div>
                    ))
                  )}
                </div>
                <div className="rounded-lg border p-3">
                  <p className="font-semibold mb-2">Weekly</p>
                  {groupedInstructions.weekly.length === 0 ? (
                    <p className="text-sm text-muted-foreground">No weekly steps.</p>
                  ) : (
                    groupedInstructions.weekly.map((inst) => (
                      <div key={inst.id} className="mb-2">
                        <p className="text-sm font-medium">{inst.title}</p>
                        <p className="text-xs text-muted-foreground">{inst.day_of_week || "Any day"}</p>
                      </div>
                    ))
                  )}
                </div>
                <div className="rounded-lg border p-3">
                  <p className="font-semibold mb-2">As Needed</p>
                  {groupedInstructions.as_needed.length === 0 ? (
                    <p className="text-sm text-muted-foreground">No on-demand steps.</p>
                  ) : (
                    groupedInstructions.as_needed.map((inst) => (
                      <div key={inst.id} className="mb-2">
                        <p className="text-sm font-medium">{inst.title}</p>
                        <p className="text-xs text-muted-foreground">{inst.summary || "Tap Ask Me for spoken guidance."}</p>
                      </div>
                    ))
                  )}
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <div className="mb-8">
          <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
            <DialogTrigger asChild>
              <Button variant="accessible" className="gap-2">
                <Plus className="h-4 w-4" /> Add Medication
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-lg">
              <DialogHeader>
                <DialogTitle className="text-2xl font-display">Add Medication</DialogTitle>
              </DialogHeader>
              <div className="space-y-4 mt-2">
                <div>
                  <Label htmlFor="med-name">Medication Name</Label>
                  <Input
                    id="med-name"
                    value={newMedication.name}
                    onChange={(e) => setNewMedication((prev) => ({ ...prev, name: e.target.value }))}
                    placeholder="e.g., Donepezil"
                    className="mt-2"
                  />
                </div>

                <div>
                  <Label htmlFor="med-dose">Dosage</Label>
                  <Input
                    id="med-dose"
                    value={newMedication.dosage}
                    onChange={(e) => setNewMedication((prev) => ({ ...prev, dosage: e.target.value }))}
                    placeholder="e.g., 10mg"
                    className="mt-2"
                  />
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <Label htmlFor="med-frequency">Frequency</Label>
                    <Input
                      id="med-frequency"
                      value={newMedication.frequency}
                      onChange={(e) => setNewMedication((prev) => ({ ...prev, frequency: e.target.value }))}
                      placeholder="daily"
                      className="mt-2"
                    />
                  </div>
                  <div>
                    <Label htmlFor="med-times">Times/day</Label>
                    <Input
                      id="med-times"
                      type="number"
                      min="1"
                      max="8"
                      value={newMedication.times_per_day}
                      onChange={(e) => setNewMedication((prev) => ({ ...prev, times_per_day: e.target.value }))}
                      className="mt-2"
                    />
                  </div>
                </div>

                <div>
                  <Label htmlFor="med-schedule">Scheduled times (comma-separated HH:MM)</Label>
                  <Input
                    id="med-schedule"
                    value={newMedication.scheduled_times}
                    onChange={(e) => setNewMedication((prev) => ({ ...prev, scheduled_times: e.target.value }))}
                    placeholder="08:00,20:00"
                    className="mt-2"
                  />
                </div>

                <div>
                  <Label htmlFor="med-doctor" className="flex items-center gap-2">
                    <Stethoscope className="h-4 w-4" /> Prescribing Doctor
                  </Label>
                  <Input
                    id="med-doctor"
                    value={newMedication.prescribing_doctor}
                    onChange={(e) =>
                      setNewMedication((prev) => ({ ...prev, prescribing_doctor: e.target.value }))
                    }
                    placeholder="Dr. Name"
                    className="mt-2"
                  />
                </div>

                <div>
                  <Label htmlFor="med-instructions">Instructions</Label>
                  <Textarea
                    id="med-instructions"
                    value={newMedication.instructions}
                    onChange={(e) =>
                      setNewMedication((prev) => ({ ...prev, instructions: e.target.value }))
                    }
                    placeholder="Take after breakfast"
                    className="mt-2"
                  />
                </div>

                <Button className="w-full" onClick={addMedication} disabled={saving}>
                  {saving ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                  Save Medication
                </Button>
              </div>
            </DialogContent>
          </Dialog>
        </div>

        {loading && (
          <div className="flex justify-center py-8">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
          </div>
        )}

        {!loading && activeMedications.length === 0 && (
          <Card className="border-2 border-border">
            <CardContent className="p-8 text-center">
              <Pill className="h-10 w-10 text-muted-foreground mx-auto mb-3" />
              <h3 className="text-xl font-semibold mb-2">No medications yet</h3>
              <p className="text-muted-foreground">Add medications to track adherence and missed doses.</p>
            </CardContent>
          </Card>
        )}

        {!loading && activeMedications.length > 0 && (
          <div className="grid gap-4">
            {activeMedications.map((medication) => (
              <Card key={medication.id} className="border-2 border-border">
                <CardContent className="p-5">
                  <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-4">
                    <div className="space-y-1">
                      <h3 className="text-xl font-semibold">{medication.name}</h3>
                      <p className="text-muted-foreground">Dose: {medication.dosage}</p>
                      <p className="text-muted-foreground">Frequency: {medication.frequency}</p>
                      {medication.scheduled_times?.length > 0 && (
                        <p className="text-muted-foreground flex items-center gap-1">
                          <Clock className="h-4 w-4" />
                          {medication.scheduled_times.join(", ")}
                        </p>
                      )}
                      {medication.prescribing_doctor && (
                        <p className="text-muted-foreground">Doctor: {medication.prescribing_doctor}</p>
                      )}
                    </div>

                    <div className="flex flex-wrap gap-2">
                      <Button
                        variant="accessible"
                        className="gap-2"
                        onClick={() => logIntake(medication.id, "taken")}
                      >
                        <Check className="h-4 w-4" />
                        Mark Taken
                      </Button>
                      <Button
                        variant="outline"
                        className="gap-2"
                        onClick={() => logIntake(medication.id, "missed")}
                      >
                        <AlertTriangle className="h-4 w-4" />
                        Mark Missed
                      </Button>
                      <Button
                        variant="destructive"
                        size="icon"
                        onClick={() => removeMedication(medication.id)}
                        disabled={deleting === medication.id}
                      >
                        {deleting === medication.id ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <Trash2 className="h-4 w-4" />
                        )}
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}

        {!loading && missedDoses.length > 0 && (
          <Card className="border-2 border-destructive/40 mt-8">
            <CardHeader>
              <CardTitle className="text-destructive">Missed Dose Alerts</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {missedDoses.slice(0, 8).map((dose) => (
                <div key={`${dose.medication_id}-${dose.scheduled_for}`} className="p-3 rounded-lg border border-destructive/40 bg-destructive/10">
                  <p className="font-medium">{dose.name} ({dose.dosage})</p>
                  <p className="text-sm text-muted-foreground">Scheduled: {new Date(dose.scheduled_for).toLocaleString()}</p>
                  <p className="text-sm text-destructive">{dose.hours_overdue} hours overdue</p>
                  <Button
                    variant="outline"
                    size="sm"
                    className="mt-2"
                    onClick={() => logIntake(dose.medication_id, "taken", dose.scheduled_for)}
                  >
                    Mark Taken Now
                  </Button>
                </div>
              ))}
            </CardContent>
          </Card>
        )}

        {!loading && interactions.length > 0 && (
          <Card className="border-2 border-amber-400 mt-8">
            <CardHeader>
              <CardTitle className="text-amber-700">Interaction Warnings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {interactions.map((interaction, index) => (
                <div key={`${interaction.medications?.join("-")}-${index}`} className="p-3 rounded-lg border border-amber-400 bg-amber-100/50">
                  <p className="font-medium">{interaction.medications?.join(" + ")}</p>
                  <p className="text-sm text-muted-foreground">{interaction.message}</p>
                </div>
              ))}
              <p className="text-xs text-muted-foreground mt-2">
                Interaction checks are basic guidance only. Always confirm with a clinician or pharmacist.
              </p>
            </CardContent>
          </Card>
        )}
      </div>
    </section>
  );
};

export default MedicationSection;
