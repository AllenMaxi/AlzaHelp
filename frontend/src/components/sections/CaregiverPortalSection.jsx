import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  ClipboardPlus,
  Copy,
  FileText,
  Trash2,
  Loader2,
  ShieldCheck,
  Upload,
  UserPlus,
  Users,
  Bell,
  Link2
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { careInstructionsApi, caregiverApi, uploadApi } from "@/services/api";
import { toast } from "sonner";

export const CaregiverPortalSection = ({ user, onNavigate }) => {
  const [loading, setLoading] = useState(true);
  const [links, setLinks] = useState({ as_patient: [], as_caregiver: [] });
  const [selectedPatientId, setSelectedPatientId] = useState(null);
  const [patientDashboard, setPatientDashboard] = useState(null);

  const [inviteRole, setInviteRole] = useState("caregiver");
  const [invitePermission, setInvitePermission] = useState("edit");
  const [inviteNote, setInviteNote] = useState("");
  const [createdInviteCode, setCreatedInviteCode] = useState("");

  const [acceptCode, setAcceptCode] = useState("");

  const [reminderData, setReminderData] = useState({
    title: "",
    time: "09:00",
    period: "morning",
    category: "health"
  });

  const [familyData, setFamilyData] = useState({
    name: "",
    relationship: "caregiver",
    relationship_label: "Caregiver",
    category: "other",
    notes: "",
    photos_csv: ""
  });

  const [shareLink, setShareLink] = useState("");
  const [uploadingPhotos, setUploadingPhotos] = useState(false);
  const [uploadedPhotoUrls, setUploadedPhotoUrls] = useState([]);
  const [careInstructions, setCareInstructions] = useState([]);
  const [loadingInstructions, setLoadingInstructions] = useState(false);
  const [savingInstruction, setSavingInstruction] = useState(false);
  const [uploadingInstructionFile, setUploadingInstructionFile] = useState(false);
  const [deletingInstructionId, setDeletingInstructionId] = useState(null);
  const [instructionFile, setInstructionFile] = useState(null);
  const [instructionData, setInstructionData] = useState({
    title: "",
    summary: "",
    instruction_text: "",
    frequency: "daily",
    policy_type: "general",
    regimen_key: "",
    effective_start_date: "",
    effective_end_date: "",
    signoff_required: false,
    day_of_week: "",
    time_of_day: "",
    tags_csv: ""
  });

  const loadLinks = useCallback(async () => {
    setLoading(true);
    try {
      const data = await caregiverApi.getLinks();
      setLinks(data || { as_patient: [], as_caregiver: [] });
      if (!selectedPatientId && data?.as_caregiver?.length > 0) {
        setSelectedPatientId(data.as_caregiver[0].patient_id);
      }
    } catch (error) {
      toast.error("Could not load caregiver links.");
    } finally {
      setLoading(false);
    }
  }, [selectedPatientId]);

  useEffect(() => {
    loadLinks();
  }, [loadLinks]);

  const loadPatientDashboard = async (patientId) => {
    if (!patientId) return;
    try {
      const data = await caregiverApi.getPatientDashboard(patientId);
      setPatientDashboard(data);
    } catch (error) {
      toast.error("Could not load patient dashboard.");
    }
  };

  useEffect(() => {
    if (selectedPatientId) {
      loadPatientDashboard(selectedPatientId);
    }
  }, [selectedPatientId]);

  const currentCareLink = useMemo(() => {
    return links.as_caregiver?.find((link) => link.patient_id === selectedPatientId) || null;
  }, [links, selectedPatientId]);

  const patientScopeId = user?.role === "patient" ? user.user_id : selectedPatientId;
  const canEditPatientData = user?.role === "patient" || currentCareLink?.permission === "edit";

  const loadCareInstructions = useCallback(async () => {
    if (!patientScopeId) {
      setCareInstructions([]);
      return;
    }
    setLoadingInstructions(true);
    try {
      const data = await careInstructionsApi.getAll(
        patientScopeId,
        canEditPatientData ? false : true,
        null,
        { includeDrafts: canEditPatientData }
      );
      setCareInstructions(data || []);
    } catch (error) {
      toast.error("Could not load care instructions.");
    } finally {
      setLoadingInstructions(false);
    }
  }, [patientScopeId, canEditPatientData]);

  useEffect(() => {
    loadCareInstructions();
  }, [loadCareInstructions]);

  const createInvite = async () => {
    try {
      const invite = await caregiverApi.createInvite({
        role: inviteRole,
        permission: invitePermission,
        note: inviteNote || null,
        expires_in_days: 14
      });
      setCreatedInviteCode(invite.code);
      toast.success("Invite code generated.");
      loadLinks();
    } catch (error) {
      toast.error(error.message || "Could not create invite.");
    }
  };

  const acceptInvite = async () => {
    if (!acceptCode.trim()) {
      toast.error("Enter an invite code.");
      return;
    }
    try {
      await caregiverApi.acceptInvite(acceptCode.trim());
      toast.success("Invite accepted.");
      setAcceptCode("");
      loadLinks();
    } catch (error) {
      toast.error(error.message || "Could not accept invite.");
    }
  };

  const createPatientReminder = async () => {
    const patientId = patientScopeId;
    if (!patientId) return;
    if (!reminderData.title.trim()) {
      toast.error("Reminder title is required.");
      return;
    }

    try {
      await caregiverApi.createPatientReminder(patientId, {
        title: reminderData.title.trim(),
        time: reminderData.time,
        period: reminderData.period,
        category: reminderData.category
      });
      toast.success("Reminder added for patient.");
      setReminderData({ title: "", time: "09:00", period: "morning", category: "health" });
      if (selectedPatientId) loadPatientDashboard(selectedPatientId);
    } catch (error) {
      toast.error(error.message || "Could not add reminder.");
    }
  };

  const createPatientFamilyMember = async () => {
    const patientId = patientScopeId;
    if (!patientId) return;
    if (!familyData.name.trim()) {
      toast.error("Family member name is required.");
      return;
    }

    const manualPhotos = familyData.photos_csv
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
    const photos = [...uploadedPhotoUrls, ...manualPhotos];

    try {
      await caregiverApi.createPatientFamilyMember(patientId, {
        name: familyData.name.trim(),
        relationship: familyData.relationship.trim() || "family",
        relationship_label: familyData.relationship_label.trim() || "Family",
        category: familyData.category,
        notes: familyData.notes || null,
        photos,
        phone: null,
        address: null,
        birthday: null,
        voice_notes: []
      });
      toast.success("Family info updated remotely.");
      setFamilyData({
        name: "",
        relationship: "caregiver",
        relationship_label: "Caregiver",
        category: "other",
        notes: "",
        photos_csv: ""
      });
      setUploadedPhotoUrls([]);
    } catch (error) {
      toast.error(error.message || "Could not add family member.");
    }
  };

  const handlePatientPhotoUpload = async (event) => {
    const files = Array.from(event.target.files || []);
    const patientId = patientScopeId;
    if (!files.length || !patientId) return;

    setUploadingPhotos(true);
    try {
      const urls = await uploadApi.uploadMultiple(files, patientId);
      setUploadedPhotoUrls((prev) => [...prev, ...urls]);
      toast.success("Photos uploaded for patient.");
    } catch (error) {
      toast.error("Could not upload photos.");
    } finally {
      setUploadingPhotos(false);
      event.target.value = "";
    }
  };

  const createReadOnlyShare = async () => {
    const patientId = user?.role === "patient" ? user.user_id : selectedPatientId;
    if (!patientId) {
      toast.error("Select a patient first.");
      return;
    }
    try {
      const data = await caregiverApi.createReadOnlyShare(patientId, 14);
      setShareLink(data.share_path);
      toast.success("Read-only share created.");
    } catch (error) {
      toast.error(error.message || "Could not create share link.");
    }
  };

  const resetInstructionForm = () => {
    setInstructionData({
      title: "",
      summary: "",
      instruction_text: "",
      frequency: "daily",
      policy_type: "general",
      regimen_key: "",
      effective_start_date: "",
      effective_end_date: "",
      signoff_required: false,
      day_of_week: "",
      time_of_day: "",
      tags_csv: ""
    });
    setInstructionFile(null);
  };

  const createTextInstruction = async () => {
    if (!patientScopeId) {
      toast.error("Select a patient first.");
      return;
    }
    if (!instructionData.title.trim() || !instructionData.instruction_text.trim()) {
      toast.error("Title and instruction text are required.");
      return;
    }
    const medicationPolicy = instructionData.policy_type.trim().toLowerCase() === "medication";
    const requiresSignoff = medicationPolicy ? true : instructionData.signoff_required;
    setSavingInstruction(true);
    try {
      await careInstructionsApi.create(
        {
          title: instructionData.title.trim(),
          summary: instructionData.summary.trim() || null,
          instruction_text: instructionData.instruction_text.trim(),
          frequency: instructionData.frequency,
          policy_type: instructionData.policy_type,
          regimen_key: instructionData.regimen_key || null,
          effective_start_date: instructionData.effective_start_date || null,
          effective_end_date: instructionData.effective_end_date || null,
          signoff_required: requiresSignoff,
          day_of_week: instructionData.day_of_week || null,
          time_of_day: instructionData.time_of_day || null,
          tags: instructionData.tags_csv
            .split(",")
            .map((tag) => tag.trim())
            .filter(Boolean)
        },
        patientScopeId
      );
      toast.success("Care instruction saved.");
      resetInstructionForm();
      loadCareInstructions();
      if (selectedPatientId) loadPatientDashboard(selectedPatientId);
    } catch (error) {
      toast.error(error.message || "Could not save instruction.");
    } finally {
      setSavingInstruction(false);
    }
  };

  const uploadInstructionDocument = async () => {
    if (!patientScopeId) {
      toast.error("Select a patient first.");
      return;
    }
    if (!instructionFile) {
      toast.error("Select a file first.");
      return;
    }
    if (!instructionData.title.trim()) {
      toast.error("Instruction title is required.");
      return;
    }
    const medicationPolicy = instructionData.policy_type.trim().toLowerCase() === "medication";
    const requiresSignoff = medicationPolicy ? true : instructionData.signoff_required;
    setUploadingInstructionFile(true);
    try {
      await careInstructionsApi.upload(
        {
          file: instructionFile,
          title: instructionData.title.trim(),
          instructionText: instructionData.instruction_text.trim(),
          summary: instructionData.summary.trim(),
          frequency: instructionData.frequency,
          policyType: instructionData.policy_type,
          regimenKey: instructionData.regimen_key,
          effectiveStartDate: instructionData.effective_start_date,
          effectiveEndDate: instructionData.effective_end_date,
          signoffRequired: requiresSignoff,
          dayOfWeek: instructionData.day_of_week,
          timeOfDay: instructionData.time_of_day,
          tags: instructionData.tags_csv
            .split(",")
            .map((tag) => tag.trim())
            .filter(Boolean)
        },
        patientScopeId
      );
      toast.success("Instruction document uploaded and ingested.");
      resetInstructionForm();
      loadCareInstructions();
      if (selectedPatientId) loadPatientDashboard(selectedPatientId);
    } catch (error) {
      toast.error(error.message || "Could not upload instruction.");
    } finally {
      setUploadingInstructionFile(false);
    }
  };

  const deleteInstruction = async (instructionId) => {
    if (!patientScopeId) return;
    setDeletingInstructionId(instructionId);
    try {
      await careInstructionsApi.delete(instructionId, patientScopeId);
      toast.success("Instruction removed.");
      loadCareInstructions();
    } catch (error) {
      toast.error(error.message || "Could not remove instruction.");
    } finally {
      setDeletingInstructionId(null);
    }
  };

  const signoffInstruction = async (instructionId) => {
    if (!patientScopeId) return;
    try {
      await careInstructionsApi.signoff(
        instructionId,
        { approved: true, signed_by_name: user?.name || "Clinician" },
        patientScopeId
      );
      toast.success("Instruction signed off and activated.");
      loadCareInstructions();
      if (selectedPatientId) loadPatientDashboard(selectedPatientId);
    } catch (error) {
      toast.error(error.message || "Could not sign off instruction.");
    }
  };

  return (
    <section className="py-8 sm:py-12">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 rounded-full bg-primary/10 px-4 py-2 mb-4">
            <ShieldCheck className="h-5 w-5 text-primary" />
            <span className="text-base font-medium text-primary">Caregiver Portal</span>
          </div>
          <h2 className="font-display text-3xl sm:text-4xl font-bold text-foreground mb-2">
            Shared Care Access
          </h2>
          <p className="text-accessible text-muted-foreground">
            Role-based caregiver and clinician access with patient dashboard and remote actions.
          </p>
          <div className="mt-3">
            <Badge variant="secondary">Current role: {user?.role || "patient"}</Badge>
          </div>
        </div>

        {loading ? (
          <div className="flex justify-center py-10">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
          </div>
        ) : (
          <div className="grid gap-4 lg:grid-cols-2">
            <Card className="border-2 border-border">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Users className="h-5 w-5 text-primary" />
                  Link Management
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {user?.role === "patient" && (
                  <>
                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <Label>Invite Role</Label>
                        <Input value={inviteRole} onChange={(e) => setInviteRole(e.target.value)} className="mt-2" />
                      </div>
                      <div>
                        <Label>Permission</Label>
                        <Input value={invitePermission} onChange={(e) => setInvitePermission(e.target.value)} className="mt-2" />
                      </div>
                    </div>
                    <div>
                      <Label>Invite Note</Label>
                      <Textarea value={inviteNote} onChange={(e) => setInviteNote(e.target.value)} className="mt-2" />
                    </div>
                    <Button onClick={createInvite} className="gap-2">
                      <UserPlus className="h-4 w-4" /> Create Invite Code
                    </Button>
                    {createdInviteCode && (
                      <div className="p-3 rounded-lg border bg-muted/40">
                        <p className="text-sm">Invite code:</p>
                        <p className="text-lg font-bold tracking-wide">{createdInviteCode}</p>
                      </div>
                    )}
                  </>
                )}

                {user?.role !== "patient" && (
                  <>
                    <div>
                      <Label>Accept Invite Code</Label>
                      <Input value={acceptCode} onChange={(e) => setAcceptCode(e.target.value)} className="mt-2" />
                    </div>
                    <Button onClick={acceptInvite} className="gap-2">
                      <ClipboardPlus className="h-4 w-4" /> Accept Invite
                    </Button>
                  </>
                )}

                <div className="pt-2 border-t border-border">
                  <p className="font-semibold mb-2">Your active links</p>
                  <p className="text-sm text-muted-foreground">As patient: {links.as_patient?.length || 0}</p>
                  <p className="text-sm text-muted-foreground">As caregiver/clinician: {links.as_caregiver?.length || 0}</p>
                </div>
              </CardContent>
            </Card>

            <Card className="border-2 border-border">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Link2 className="h-5 w-5 text-primary" />
                  Clinician Read-only Share
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {user?.role !== "patient" && links.as_caregiver?.length > 0 && (
                  <div>
                    <Label>Select Patient</Label>
                    <select
                      className="w-full mt-2 rounded-md border border-input bg-background px-3 py-2"
                      value={selectedPatientId || ""}
                      onChange={(e) => setSelectedPatientId(e.target.value || null)}
                    >
                      {links.as_caregiver.map((link) => (
                        <option key={link.id} value={link.patient_id}>
                          {link.patient?.name || link.patient_id}
                        </option>
                      ))}
                    </select>
                  </div>
                )}

                <Button onClick={createReadOnlyShare} className="gap-2">
                  <Copy className="h-4 w-4" /> Generate Read-only Link
                </Button>

                {shareLink && (
                  <div className="p-3 rounded-lg border bg-muted/40">
                    <p className="text-sm">Share path:</p>
                    <p className="text-xs break-all font-mono">{shareLink}</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        )}

        {links.as_caregiver?.length > 0 && (
          <Card className="border-2 border-border mt-8">
            <CardHeader>
              <CardTitle>Patient Dashboard (Caregiver View)</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-3 md:grid-cols-3 lg:grid-cols-9">
                <div className="rounded-lg border p-3">
                  <p className="text-sm text-muted-foreground">Reminder completion</p>
                  <p className="text-2xl font-bold">{patientDashboard?.summary?.reminder_completion_percent ?? 0}%</p>
                </div>
                <div className="rounded-lg border p-3">
                  <p className="text-sm text-muted-foreground">Medication adherence</p>
                  <p className="text-2xl font-bold">{patientDashboard?.summary?.adherence_percent_last_7_days ?? 0}%</p>
                </div>
                <div className="rounded-lg border p-3">
                  <p className="text-sm text-muted-foreground">Active meds</p>
                  <p className="text-2xl font-bold">{patientDashboard?.summary?.medications_active ?? 0}</p>
                </div>
                <div className="rounded-lg border p-3">
                  <p className="text-sm text-muted-foreground">Open safety alerts</p>
                  <p className="text-2xl font-bold text-destructive">{patientDashboard?.summary?.unacknowledged_safety_alerts ?? 0}</p>
                </div>
                <div className="rounded-lg border p-3">
                  <p className="text-sm text-muted-foreground">Care instructions</p>
                  <p className="text-2xl font-bold">{patientDashboard?.summary?.care_instructions_active ?? 0}</p>
                </div>
                <div className="rounded-lg border p-3">
                  <p className="text-sm text-muted-foreground">Emergency contacts</p>
                  <p className="text-2xl font-bold">{patientDashboard?.summary?.emergency_contacts_count ?? 0}</p>
                </div>
                <div className="rounded-lg border p-3">
                  <p className="text-sm text-muted-foreground">BPSD events (30d)</p>
                  <p className="text-2xl font-bold">{patientDashboard?.summary?.bpsd_events_last_30_days ?? 0}</p>
                </div>
                <div className="rounded-lg border p-3">
                  <p className="text-sm text-muted-foreground">Low mood days (30d)</p>
                  <p className="text-2xl font-bold text-amber-600">{patientDashboard?.summary?.low_mood_days_last_30_days ?? 0}</p>
                </div>
                <div className="rounded-lg border p-3">
                  <p className="text-sm text-muted-foreground">Top symptom</p>
                  <p className="text-lg font-semibold capitalize">
                    {(patientDashboard?.summary?.top_bpsd_symptom || "none").replace(/_/g, " ")}
                  </p>
                </div>
              </div>

              {canEditPatientData && (
                <div className="grid gap-4 lg:grid-cols-2">
                  <Card className="border border-border">
                    <CardHeader>
                      <CardTitle className="text-base flex items-center gap-2">
                        <Bell className="h-4 w-4" /> Remote Reminder Add
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2">
                      <Input
                        placeholder="Reminder title"
                        value={reminderData.title}
                        onChange={(e) => setReminderData((prev) => ({ ...prev, title: e.target.value }))}
                      />
                      <div className="grid grid-cols-3 gap-2">
                        <Input
                          placeholder="Time"
                          value={reminderData.time}
                          onChange={(e) => setReminderData((prev) => ({ ...prev, time: e.target.value }))}
                        />
                        <Input
                          placeholder="Period"
                          value={reminderData.period}
                          onChange={(e) => setReminderData((prev) => ({ ...prev, period: e.target.value }))}
                        />
                        <Input
                          placeholder="Category"
                          value={reminderData.category}
                          onChange={(e) => setReminderData((prev) => ({ ...prev, category: e.target.value }))}
                        />
                      </div>
                      <Button onClick={createPatientReminder}>Add Reminder</Button>
                    </CardContent>
                  </Card>

                  <Card className="border border-border">
                    <CardHeader>
                      <CardTitle className="text-base">Remote Family Update</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2">
                      <Input
                        placeholder="Name"
                        value={familyData.name}
                        onChange={(e) => setFamilyData((prev) => ({ ...prev, name: e.target.value }))}
                      />
                      <div className="grid grid-cols-2 gap-2">
                        <Input
                          placeholder="Relationship"
                          value={familyData.relationship}
                          onChange={(e) => setFamilyData((prev) => ({ ...prev, relationship: e.target.value }))}
                        />
                        <Input
                          placeholder="Label"
                          value={familyData.relationship_label}
                          onChange={(e) => setFamilyData((prev) => ({ ...prev, relationship_label: e.target.value }))}
                        />
                      </div>
                      <Input
                        placeholder="Photo URLs (comma separated)"
                        value={familyData.photos_csv}
                        onChange={(e) => setFamilyData((prev) => ({ ...prev, photos_csv: e.target.value }))}
                      />
                      <div>
                        <Label className="text-sm">Or upload photos directly</Label>
                        <Input
                          type="file"
                          accept="image/*"
                          multiple
                          onChange={handlePatientPhotoUpload}
                          className="mt-2"
                          disabled={uploadingPhotos || !patientScopeId}
                        />
                      </div>
                      {uploadingPhotos && (
                        <p className="text-sm text-muted-foreground flex items-center gap-2">
                          <Loader2 className="h-4 w-4 animate-spin" />
                          Uploading photos...
                        </p>
                      )}
                      {uploadedPhotoUrls.length > 0 && (
                        <div className="flex flex-wrap gap-2">
                          {uploadedPhotoUrls.map((url) => (
                            <img
                              key={url}
                              src={url}
                              alt="Uploaded patient reference"
                              className="h-12 w-12 rounded-md object-cover border border-border"
                            />
                          ))}
                        </div>
                      )}
                      <Textarea
                        placeholder="Notes"
                        value={familyData.notes}
                        onChange={(e) => setFamilyData((prev) => ({ ...prev, notes: e.target.value }))}
                      />
                      <Button onClick={createPatientFamilyMember}>Add Family Entry</Button>
                    </CardContent>
                  </Card>
                </div>
              )}

              <div className="grid gap-4 lg:grid-cols-2">
                <Card className="border border-border">
                  <CardHeader>
                    <CardTitle className="text-base">Recent Notes</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    {(patientDashboard?.daily_notes_recent || []).slice(0, 5).map((note) => (
                      <div key={note.id} className="p-2 rounded border">
                        <p className="text-xs text-muted-foreground">{note.date}</p>
                        <p className="text-sm">{note.note}</p>
                      </div>
                    ))}
                    {(patientDashboard?.daily_notes_recent || []).length === 0 && (
                      <p className="text-sm text-muted-foreground">No recent notes.</p>
                    )}
                  </CardContent>
                </Card>

                <Card className="border border-border">
                  <CardHeader>
                    <CardTitle className="text-base">Safety Alerts</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    {(patientDashboard?.safety_alerts_open || []).slice(0, 5).map((alert) => (
                      <div key={alert.id} className="p-2 rounded border border-destructive/40 bg-destructive/10">
                        <p className="text-sm font-medium text-destructive">{alert.message}</p>
                        <p className="text-xs text-muted-foreground">{new Date(alert.triggered_at).toLocaleString()}</p>
                      </div>
                    ))}
                    {(patientDashboard?.safety_alerts_open || []).length === 0 && (
                      <p className="text-sm text-muted-foreground">No open alerts.</p>
                    )}
                  </CardContent>
                </Card>

                <Card className="border border-border">
                  <CardHeader>
                    <CardTitle className="text-base">Mood Check-ins</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    {(patientDashboard?.mood_checkins_recent || []).slice(0, 5).map((entry) => (
                      <div key={entry.id} className="p-2 rounded border">
                        <p className="text-sm font-medium">
                          Mood {entry.mood_score}/5 · Energy {entry.energy_score}/5
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {new Date(entry.created_at).toLocaleString()}
                        </p>
                      </div>
                    ))}
                    {(patientDashboard?.mood_checkins_recent || []).length === 0 && (
                      <p className="text-sm text-muted-foreground">No mood check-ins yet.</p>
                    )}
                  </CardContent>
                </Card>

                <Card className="border border-border">
                  <CardHeader>
                    <CardTitle className="text-base">Behavior Observations</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    {(patientDashboard?.bpsd_observations_recent || []).slice(0, 5).map((entry) => (
                      <div key={entry.id} className="p-2 rounded border">
                        <p className="text-sm font-medium capitalize">
                          {entry.symptom?.replace(/_/g, " ")} · Severity {entry.severity}/5
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {entry.time_of_day} · {new Date(entry.observed_at).toLocaleString()}
                        </p>
                      </div>
                    ))}
                    {(patientDashboard?.bpsd_observations_recent || []).length === 0 && (
                      <p className="text-sm text-muted-foreground">No behavior observations yet.</p>
                    )}
                  </CardContent>
                </Card>
              </div>
            </CardContent>
          </Card>
        )}

        {patientScopeId && (
          <Card className="border-2 border-border mt-8">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5 text-primary" />
                Clinical Instructions (Agentic RAG)
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-muted-foreground">
                Upload doctor/caregiver instructions so the voice assistant can answer with the exact process.
              </p>

              {canEditPatientData && (
                <div className="grid gap-4 lg:grid-cols-2">
                  <Card className="border border-border">
                    <CardHeader>
                      <CardTitle className="text-base">Add Text Procedure</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2">
                      <Input
                        placeholder="Title"
                        value={instructionData.title}
                        onChange={(e) => setInstructionData((prev) => ({ ...prev, title: e.target.value }))}
                      />
                      <Textarea
                        placeholder="Short summary"
                        value={instructionData.summary}
                        onChange={(e) => setInstructionData((prev) => ({ ...prev, summary: e.target.value }))}
                      />
                      <Textarea
                        placeholder="Instruction text (exact process)"
                        value={instructionData.instruction_text}
                        onChange={(e) => setInstructionData((prev) => ({ ...prev, instruction_text: e.target.value }))}
                      />
                      <div className="grid grid-cols-2 gap-2">
                        <Input
                          placeholder="Policy type (general/medication)"
                          value={instructionData.policy_type}
                          onChange={(e) => setInstructionData((prev) => ({ ...prev, policy_type: e.target.value }))}
                        />
                        <Input
                          placeholder="Regimen key (for medication)"
                          value={instructionData.regimen_key}
                          onChange={(e) => setInstructionData((prev) => ({ ...prev, regimen_key: e.target.value }))}
                        />
                      </div>
                      <div className="grid grid-cols-3 gap-2">
                        <Input
                          placeholder="Frequency (daily/weekly/as_needed)"
                          value={instructionData.frequency}
                          onChange={(e) => setInstructionData((prev) => ({ ...prev, frequency: e.target.value }))}
                        />
                        <Input
                          placeholder="Day (weekly)"
                          value={instructionData.day_of_week}
                          onChange={(e) => setInstructionData((prev) => ({ ...prev, day_of_week: e.target.value }))}
                        />
                        <Input
                          placeholder="Time of day"
                          value={instructionData.time_of_day}
                          onChange={(e) => setInstructionData((prev) => ({ ...prev, time_of_day: e.target.value }))}
                        />
                      </div>
                      <div className="grid grid-cols-2 gap-2">
                        <Input
                          placeholder="Effective start (YYYY-MM-DD)"
                          value={instructionData.effective_start_date}
                          onChange={(e) => setInstructionData((prev) => ({ ...prev, effective_start_date: e.target.value }))}
                        />
                        <Input
                          placeholder="Effective end (YYYY-MM-DD)"
                          value={instructionData.effective_end_date}
                          onChange={(e) => setInstructionData((prev) => ({ ...prev, effective_end_date: e.target.value }))}
                        />
                      </div>
                      <label className="flex items-center gap-2 text-sm">
                        <input
                          type="checkbox"
                          checked={instructionData.signoff_required}
                          onChange={(e) => setInstructionData((prev) => ({ ...prev, signoff_required: e.target.checked }))}
                        />
                        Require clinician/caregiver sign-off before active
                      </label>
                      <Input
                        placeholder="Tags (comma separated)"
                        value={instructionData.tags_csv}
                        onChange={(e) => setInstructionData((prev) => ({ ...prev, tags_csv: e.target.value }))}
                      />
                      <Button onClick={createTextInstruction} disabled={savingInstruction}>
                        {savingInstruction ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                        Save Procedure
                      </Button>
                    </CardContent>
                  </Card>

                  <Card className="border border-border">
                    <CardHeader>
                      <CardTitle className="text-base">Upload Instruction File</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2">
                      <Input
                        type="file"
                        accept=".txt,.md,.csv,.json,.pdf,.docx,text/plain,text/markdown,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        onChange={(e) => setInstructionFile(e.target.files?.[0] || null)}
                      />
                      <p className="text-xs text-muted-foreground">
                        Supported ingestion: txt/md/csv/json/pdf/docx.
                      </p>
                      {instructionFile && (
                        <p className="text-sm">Selected file: <span className="font-medium">{instructionFile.name}</span></p>
                      )}
                      <Button onClick={uploadInstructionDocument} disabled={uploadingInstructionFile} className="gap-2">
                        {uploadingInstructionFile ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
                        Upload and Ingest
                      </Button>
                    </CardContent>
                  </Card>
                </div>
              )}

              <div className="space-y-2">
                {loadingInstructions ? (
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Loading instructions...
                  </div>
                ) : careInstructions.length === 0 ? (
                  <p className="text-sm text-muted-foreground">No care instructions yet.</p>
                ) : (
                  careInstructions.map((instruction) => (
                    <div key={instruction.id} className="rounded-lg border p-3 flex items-start justify-between gap-3">
                      <div>
                        <p className="font-medium text-sm">{instruction.title}</p>
                        <p className="text-xs text-muted-foreground capitalize">
                          v{instruction.version || 1} - {instruction.policy_type || "general"} - {instruction.signoff_status || "not_required"} -{" "}
                          {instruction.frequency}
                          {instruction.day_of_week ? ` - ${instruction.day_of_week}` : ""}
                          {instruction.time_of_day ? ` - ${instruction.time_of_day}` : ""}
                        </p>
                        {instruction.summary ? (
                          <p className="text-sm mt-1">{instruction.summary}</p>
                        ) : (
                          <p className="text-sm mt-1">{instruction.instruction_text}</p>
                        )}
                      </div>
                      {canEditPatientData && (
                        <div className="flex gap-2">
                          {instruction.signoff_required && instruction.signoff_status !== "signed_off" && (
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => signoffInstruction(instruction.id)}
                            >
                              Sign Off
                            </Button>
                          )}
                          <Button
                            variant="destructive"
                            size="icon"
                            onClick={() => deleteInstruction(instruction.id)}
                            disabled={deletingInstructionId === instruction.id}
                          >
                            {deletingInstructionId === instruction.id ? (
                              <Loader2 className="h-4 w-4 animate-spin" />
                            ) : (
                              <Trash2 className="h-4 w-4" />
                            )}
                          </Button>
                        </div>
                      )}
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        )}

        {user?.role === "patient" && (
          <div className="mt-8 flex justify-center">
            <Button variant="outline" onClick={() => onNavigate?.("medications")}>
              Open Medication Tracker
            </Button>
          </div>
        )}
      </div>
    </section>
  );
};

export default CaregiverPortalSection;
