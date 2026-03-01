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
  Link2,
  Pill,
  Plus,
  Power,
  Edit3,
  Crown,
  Calendar,
  Download,
  Sparkles
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { careInstructionsApi, caregiverApi, externalBotApi, medicationsApi, uploadApi, billingApi, referralApi, careReportApi } from "@/services/api";
import { toast } from "sonner";
import { useTranslation } from "react-i18next";

export const CaregiverPortalSection = ({ user, onNavigate, subscriptionTier = "free" }) => {
  const { t } = useTranslation();
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
  // Medication management state
  const [patientMedications, setPatientMedications] = useState([]);
  const [loadingMedications, setLoadingMedications] = useState(false);
  const [savingMedication, setSavingMedication] = useState(false);
  const [editingMedicationId, setEditingMedicationId] = useState(null);
  const [deletingMedicationId, setDeletingMedicationId] = useState(null);
  const [showMedForm, setShowMedForm] = useState(false);
  const [medData, setMedData] = useState({
    name: "",
    dosage: "",
    frequency: "daily",
    times_per_day: 1,
    scheduled_times: "",
    prescribing_doctor: "",
    instructions: "",
    start_date: "",
    end_date: ""
  });

  const [billingStatus, setBillingStatus] = useState({ patient_limit: 3, patient_count: 0 });

  const [externalBotPatients, setExternalBotPatients] = useState([]);
  const [externalBotLinks, setExternalBotLinks] = useState([]);
  const [loadingExternalBot, setLoadingExternalBot] = useState(false);
  const [creatingExternalCode, setCreatingExternalCode] = useState(false);
  const [updatingExternalLinkId, setUpdatingExternalLinkId] = useState(null);
  const [revokingExternalLinkId, setRevokingExternalLinkId] = useState(null);
  const [externalBotChannel, setExternalBotChannel] = useState("telegram");
  const [externalBotExpiresMinutes, setExternalBotExpiresMinutes] = useState("20");
  const [externalBotPatientId, setExternalBotPatientId] = useState("");
  const [latestExternalCode, setLatestExternalCode] = useState(null);

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

  useEffect(() => {
    billingApi.getStatus().then((data) => {
      if (data) setBillingStatus(data);
    }).catch(() => {});
  }, []);

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
  const canUseExternalBot = ["caregiver", "clinician", "admin"].includes((user?.role || "").toLowerCase());

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

  const loadExternalBotContext = useCallback(async () => {
    if (!canUseExternalBot) {
      setExternalBotPatients([]);
      setExternalBotLinks([]);
      return;
    }
    setLoadingExternalBot(true);
    try {
      const [patients, botLinks] = await Promise.all([
        externalBotApi.getPatients(),
        externalBotApi.getLinks(),
      ]);
      const normalizedPatients = patients || [];
      const normalizedLinks = botLinks || [];
      setExternalBotPatients(normalizedPatients);
      setExternalBotLinks(normalizedLinks);
      setExternalBotPatientId((current) => {
        if (current && normalizedPatients.some((item) => item.user_id === current)) {
          return current;
        }
        if (selectedPatientId && normalizedPatients.some((item) => item.user_id === selectedPatientId)) {
          return selectedPatientId;
        }
        return normalizedPatients[0]?.user_id || "";
      });
    } catch (error) {
      toast.error(error.message || "Could not load external bot access.");
    } finally {
      setLoadingExternalBot(false);
    }
  }, [canUseExternalBot, selectedPatientId]);

  useEffect(() => {
    loadExternalBotContext();
  }, [loadExternalBotContext]);

  const copyTextToClipboard = async (value, successMessage) => {
    if (!value) return;
    try {
      await navigator.clipboard.writeText(value);
      toast.success(successMessage || "Copied.");
    } catch (error) {
      toast.error("Could not copy to clipboard.");
    }
  };

  const createExternalBotCode = async () => {
    if (!canUseExternalBot) return;
    setCreatingExternalCode(true);
    try {
      const parsedExpires = Number.parseInt(externalBotExpiresMinutes, 10);
      const payload = {
        channel: externalBotChannel,
        expires_in_minutes: Number.isFinite(parsedExpires) ? parsedExpires : 20,
      };
      if (externalBotPatientId) {
        payload.patient_user_id = externalBotPatientId;
      }
      const created = await externalBotApi.createLinkCode(payload);
      setLatestExternalCode(created);
      toast.success("External bot link code generated.");
      await loadExternalBotContext();
    } catch (error) {
      toast.error(error.message || "Could not generate bot link code.");
    } finally {
      setCreatingExternalCode(false);
    }
  };

  const updateExternalBotLinkPatient = async (linkId, patientUserId) => {
    if (!linkId) return;
    setUpdatingExternalLinkId(linkId);
    try {
      await externalBotApi.updateLinkPatient(linkId, patientUserId || null);
      toast.success("Default patient updated.");
      await loadExternalBotContext();
    } catch (error) {
      toast.error(error.message || "Could not update default patient.");
    } finally {
      setUpdatingExternalLinkId(null);
    }
  };

  const revokeExternalBotLink = async (linkId) => {
    if (!linkId) return;
    setRevokingExternalLinkId(linkId);
    try {
      await externalBotApi.revokeLink(linkId);
      toast.success("External bot link revoked.");
      await loadExternalBotContext();
    } catch (error) {
      toast.error(error.message || "Could not revoke bot link.");
    } finally {
      setRevokingExternalLinkId(null);
    }
  };

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

  // ==================== Medication Management ====================

  const loadPatientMedications = useCallback(async () => {
    if (!patientScopeId) {
      setPatientMedications([]);
      return;
    }
    setLoadingMedications(true);
    try {
      const data = await medicationsApi.getAll(patientScopeId);
      setPatientMedications(data || []);
    } catch (error) {
      toast.error("Could not load medications.");
    } finally {
      setLoadingMedications(false);
    }
  }, [patientScopeId]);

  useEffect(() => {
    loadPatientMedications();
  }, [loadPatientMedications]);

  const resetMedForm = () => {
    setMedData({
      name: "",
      dosage: "",
      frequency: "daily",
      times_per_day: 1,
      scheduled_times: "",
      prescribing_doctor: "",
      instructions: "",
      start_date: "",
      end_date: ""
    });
    setEditingMedicationId(null);
    setShowMedForm(false);
  };

  const saveMedication = async () => {
    if (!patientScopeId) {
      toast.error("Select a patient first.");
      return;
    }
    if (!medData.name.trim()) {
      toast.error("Medication name is required.");
      return;
    }
    if (!medData.dosage.trim()) {
      toast.error("Dosage is required.");
      return;
    }

    const timesArray = medData.scheduled_times
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean);

    const payload = {
      name: medData.name.trim(),
      dosage: medData.dosage.trim(),
      frequency: medData.frequency,
      times_per_day: Number(medData.times_per_day) || 1,
      scheduled_times: timesArray,
      prescribing_doctor: medData.prescribing_doctor.trim() || null,
      instructions: medData.instructions.trim() || null,
      start_date: medData.start_date || null,
      end_date: medData.end_date || null
    };

    setSavingMedication(true);
    try {
      if (editingMedicationId) {
        await medicationsApi.update(editingMedicationId, payload, patientScopeId);
        toast.success("Medication updated.");
      } else {
        await medicationsApi.create(payload, patientScopeId);
        toast.success("Medication created.");
      }
      resetMedForm();
      loadPatientMedications();
      if (selectedPatientId) loadPatientDashboard(selectedPatientId);
    } catch (error) {
      toast.error(error.message || "Could not save medication.");
    } finally {
      setSavingMedication(false);
    }
  };

  const startEditMedication = (med) => {
    setMedData({
      name: med.name || "",
      dosage: med.dosage || "",
      frequency: med.frequency || "daily",
      times_per_day: med.times_per_day || 1,
      scheduled_times: (med.scheduled_times || []).join(", "),
      prescribing_doctor: med.prescribing_doctor || "",
      instructions: med.instructions || "",
      start_date: med.start_date || "",
      end_date: med.end_date || ""
    });
    setEditingMedicationId(med.id);
    setShowMedForm(true);
  };

  const toggleMedicationActive = async (med) => {
    if (!patientScopeId) return;
    try {
      await medicationsApi.update(med.id, { active: !med.active }, patientScopeId);
      toast.success(med.active ? "Medication deactivated." : "Medication reactivated.");
      loadPatientMedications();
      if (selectedPatientId) loadPatientDashboard(selectedPatientId);
    } catch (error) {
      toast.error(error.message || "Could not update medication.");
    }
  };

  const deleteMedication = async (medId) => {
    if (!patientScopeId) return;
    setDeletingMedicationId(medId);
    try {
      await medicationsApi.delete(medId, patientScopeId);
      toast.success("Medication removed.");
      loadPatientMedications();
      if (selectedPatientId) loadPatientDashboard(selectedPatientId);
    } catch (error) {
      toast.error(error.message || "Could not remove medication.");
    } finally {
      setDeletingMedicationId(null);
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
          <>
          {user?.role !== "patient" && links.as_caregiver?.length > 0 && (
            <Card className="border-2 border-primary/30 mb-4">
              <CardContent className="py-4 flex flex-col sm:flex-row items-start sm:items-center gap-4">
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  <Users className="h-5 w-5 text-primary shrink-0" />
                  <select
                    className="w-full max-w-xs rounded-md border border-input bg-background px-3 py-2 text-sm font-medium"
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
                <div className="flex items-center gap-3 text-sm">
                  <Badge variant={billingStatus.patient_count >= billingStatus.patient_limit ? "destructive" : "secondary"}>
                    Managing {billingStatus.patient_count} of {billingStatus.patient_limit} patients
                  </Badge>
                  {billingStatus.patient_count >= billingStatus.patient_limit && (
                    <Button
                      size="sm"
                      variant="outline"
                      className="gap-1 text-violet-600 border-violet-300 hover:bg-violet-50"
                      onClick={async () => {
                        try {
                          const { url } = await billingApi.createCheckout();
                          window.location.href = url;
                        } catch (e) {
                          toast.error(e.message || "Could not start checkout.");
                        }
                      }}
                    >
                      <Crown className="h-4 w-4" /> Upgrade
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          )}
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

            {canUseExternalBot && (
              <Card className="border-2 border-border lg:col-span-2">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Link2 className="h-5 w-5 text-primary" />
                    External Bot Access (Telegram/WhatsApp)
                    {subscriptionTier !== "premium" && (
                      <Badge variant="outline" className="ml-2 text-violet-600 border-violet-300">Premium</Badge>
                    )}
                  </CardTitle>
                </CardHeader>
                {subscriptionTier !== "premium" ? (
                  <CardContent className="space-y-4">
                    <div className="rounded-xl border-2 border-dashed border-violet-300 dark:border-violet-700 bg-violet-50 dark:bg-violet-950/20 p-6 text-center">
                      <Crown className="h-10 w-10 text-violet-500 mx-auto mb-3" />
                      <h4 className="text-lg font-semibold text-foreground mb-2">Unlock External Bot Access</h4>
                      <p className="text-sm text-muted-foreground mb-4 max-w-md mx-auto">
                        Monitor patients remotely via Telegram or WhatsApp. Get medication status, safety alerts, and mood updates — all from your phone.
                      </p>
                      <ul className="text-sm text-muted-foreground space-y-1 mb-5">
                        <li>Telegram & WhatsApp chatbot</li>
                        <li>SMS medication alerts to emergency contacts</li>
                        <li>Remote patient queries via voice or text</li>
                      </ul>
                      <Button
                        className="bg-violet-600 hover:bg-violet-700 text-white gap-2"
                        onClick={async () => {
                          try {
                            const { url } = await billingApi.createCheckout();
                            window.location.href = url;
                          } catch (e) {
                            toast.error(e.message || "Could not start checkout.");
                          }
                        }}
                      >
                        <Crown className="h-4 w-4" />
                        Upgrade to Premium — $9.99/mo
                      </Button>
                    </div>
                  </CardContent>
                ) : (
                <CardContent className="space-y-4">
                  <p className="text-sm text-muted-foreground">
                    Generate one-time link codes for doctor/caregiver chat access, then assign or revoke connected chats.
                  </p>

                  <div className="grid gap-3 md:grid-cols-4">
                    <div>
                      <Label>Channel</Label>
                      <select
                        className="w-full mt-2 rounded-md border border-input bg-background px-3 py-2"
                        value={externalBotChannel}
                        onChange={(e) => setExternalBotChannel(e.target.value)}
                      >
                        <option value="telegram">Telegram</option>
                        <option value="whatsapp">WhatsApp</option>
                      </select>
                    </div>

                    <div className="md:col-span-2">
                      <Label>Default Patient (optional)</Label>
                      <select
                        className="w-full mt-2 rounded-md border border-input bg-background px-3 py-2"
                        value={externalBotPatientId}
                        onChange={(e) => setExternalBotPatientId(e.target.value)}
                        disabled={loadingExternalBot}
                      >
                        <option value="">Select in chat later</option>
                        {externalBotPatients.map((patient) => (
                          <option key={patient.user_id} value={patient.user_id}>
                            {patient.name || patient.user_id}
                          </option>
                        ))}
                      </select>
                    </div>

                    <div>
                      <Label>Expires (minutes)</Label>
                      <Input
                        className="mt-2"
                        inputMode="numeric"
                        value={externalBotExpiresMinutes}
                        onChange={(e) => setExternalBotExpiresMinutes(e.target.value)}
                      />
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-2">
                    <Button onClick={createExternalBotCode} disabled={creatingExternalCode || loadingExternalBot} className="gap-2">
                      {creatingExternalCode ? <Loader2 className="h-4 w-4 animate-spin" /> : <Link2 className="h-4 w-4" />}
                      Generate Bot Link Code
                    </Button>
                    <Button variant="outline" onClick={loadExternalBotContext} disabled={loadingExternalBot} className="gap-2">
                      {loadingExternalBot ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
                      Refresh
                    </Button>
                  </div>

                  {latestExternalCode && (
                    <div className="rounded-lg border bg-muted/40 p-3 space-y-2">
                      <p className="text-sm font-semibold">Latest one-time link code</p>
                      <div className="flex flex-wrap items-center gap-2">
                        <Badge variant="secondary" className="uppercase">{latestExternalCode.channel}</Badge>
                        <code className="text-lg font-bold tracking-wider">{latestExternalCode.code}</code>
                        <Button
                          variant="outline"
                          size="sm"
                          className="gap-2"
                          onClick={() => copyTextToClipboard(latestExternalCode.code, "Bot link code copied.")}
                        >
                          <Copy className="h-4 w-4" />
                          Copy Code
                        </Button>
                      </div>
                      <p className="text-xs text-muted-foreground">{latestExternalCode.connect_instructions}</p>
                      <p className="text-xs text-muted-foreground">
                        Expires: {latestExternalCode.expires_at ? new Date(latestExternalCode.expires_at).toLocaleString() : "n/a"}
                      </p>
                    </div>
                  )}

                  <div className="space-y-2">
                    <p className="text-sm font-semibold">Connected Chats</p>
                    {loadingExternalBot ? (
                      <div className="text-sm text-muted-foreground flex items-center gap-2">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Loading connected chats...
                      </div>
                    ) : externalBotLinks.length === 0 ? (
                      <p className="text-sm text-muted-foreground">No chats linked yet.</p>
                    ) : (
                      externalBotLinks.map((link) => (
                        <div key={link.id} className="rounded-lg border p-3 space-y-2">
                          <div className="flex flex-wrap items-center gap-2">
                            <Badge variant="secondary" className="uppercase">{link.channel}</Badge>
                            <span className="text-sm font-medium">{link.peer_display_name || link.peer_id}</span>
                            {link.active === false && (
                              <Badge variant="outline">Revoked</Badge>
                            )}
                          </div>
                          <p className="text-xs text-muted-foreground font-mono">{link.peer_id}</p>
                          {link.last_seen_at && (
                            <p className="text-xs text-muted-foreground">
                              Last seen: {new Date(link.last_seen_at).toLocaleString()}
                            </p>
                          )}
                          <div className="grid gap-2 md:grid-cols-[1fr_auto]">
                            <select
                              className="w-full rounded-md border border-input bg-background px-3 py-2"
                              value={link.patient_user_id || ""}
                              disabled={updatingExternalLinkId === link.id || !link.active}
                              onChange={(e) => updateExternalBotLinkPatient(link.id, e.target.value)}
                            >
                              <option value="">No default patient</option>
                              {externalBotPatients.map((patient) => (
                                <option key={patient.user_id} value={patient.user_id}>
                                  {patient.name || patient.user_id}
                                </option>
                              ))}
                            </select>
                            <Button
                              variant="destructive"
                              size="sm"
                              onClick={() => revokeExternalBotLink(link.id)}
                              disabled={revokingExternalLinkId === link.id || !link.active}
                              className="gap-2"
                            >
                              {revokingExternalLinkId === link.id ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
                              Revoke
                            </Button>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </CardContent>
                )}
              </Card>
            )}
          </div>
          </>
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

        {/* ==================== Medication Management ==================== */}
        {patientScopeId && (
          <Card className="border-2 border-border mt-8">
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span className="flex items-center gap-2">
                  <Pill className="h-5 w-5 text-primary" />
                  Medication Management
                </span>
                {canEditPatientData && (
                  <Button
                    size="sm"
                    variant={showMedForm ? "outline" : "default"}
                    onClick={() => { showMedForm ? resetMedForm() : setShowMedForm(true); }}
                    className="gap-1"
                  >
                    <Plus className="h-4 w-4" />
                    {showMedForm ? "Cancel" : "Add Medication"}
                  </Button>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-muted-foreground">
                Create, edit, and manage patient medications. You can also do this via the chatbot on Telegram or WhatsApp.
              </p>

              {/* Add/Edit Medication Form */}
              {canEditPatientData && showMedForm && (
                <Card className="border border-primary/30 bg-primary/5">
                  <CardHeader>
                    <CardTitle className="text-base">
                      {editingMedicationId ? "Edit Medication" : "New Medication"}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <Label className="text-xs">Medication Name *</Label>
                        <Input
                          placeholder="e.g. Metformin"
                          value={medData.name}
                          onChange={(e) => setMedData((prev) => ({ ...prev, name: e.target.value }))}
                        />
                      </div>
                      <div>
                        <Label className="text-xs">Dosage *</Label>
                        <Input
                          placeholder="e.g. 500mg"
                          value={medData.dosage}
                          onChange={(e) => setMedData((prev) => ({ ...prev, dosage: e.target.value }))}
                        />
                      </div>
                    </div>
                    <div className="grid grid-cols-3 gap-2">
                      <div>
                        <Label className="text-xs">Frequency</Label>
                        <select
                          className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background"
                          value={medData.frequency}
                          onChange={(e) => setMedData((prev) => ({ ...prev, frequency: e.target.value }))}
                        >
                          <option value="daily">Daily</option>
                          <option value="twice_daily">Twice Daily</option>
                          <option value="three_times_daily">3x Daily</option>
                          <option value="every_other_day">Every Other Day</option>
                          <option value="weekly">Weekly</option>
                          <option value="custom">Custom</option>
                        </select>
                      </div>
                      <div>
                        <Label className="text-xs">Times Per Day</Label>
                        <Input
                          type="number"
                          min="1"
                          max="10"
                          value={medData.times_per_day}
                          onChange={(e) => setMedData((prev) => ({ ...prev, times_per_day: e.target.value }))}
                        />
                      </div>
                      <div>
                        <Label className="text-xs">Scheduled Times</Label>
                        <Input
                          placeholder="08:00, 20:00"
                          value={medData.scheduled_times}
                          onChange={(e) => setMedData((prev) => ({ ...prev, scheduled_times: e.target.value }))}
                        />
                      </div>
                    </div>
                    <div>
                      <Label className="text-xs">Prescribing Doctor</Label>
                      <Input
                        placeholder="Dr. Name"
                        value={medData.prescribing_doctor}
                        onChange={(e) => setMedData((prev) => ({ ...prev, prescribing_doctor: e.target.value }))}
                      />
                    </div>
                    <div>
                      <Label className="text-xs">Special Instructions</Label>
                      <Textarea
                        placeholder="Take with food, avoid alcohol, etc."
                        value={medData.instructions}
                        onChange={(e) => setMedData((prev) => ({ ...prev, instructions: e.target.value }))}
                      />
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <Label className="text-xs">Start Date</Label>
                        <Input
                          type="date"
                          value={medData.start_date}
                          onChange={(e) => setMedData((prev) => ({ ...prev, start_date: e.target.value }))}
                        />
                      </div>
                      <div>
                        <Label className="text-xs">End Date</Label>
                        <Input
                          type="date"
                          value={medData.end_date}
                          onChange={(e) => setMedData((prev) => ({ ...prev, end_date: e.target.value }))}
                        />
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <Button onClick={saveMedication} disabled={savingMedication}>
                        {savingMedication && <Loader2 className="h-4 w-4 animate-spin mr-2" />}
                        {editingMedicationId ? "Update Medication" : "Add Medication"}
                      </Button>
                      <Button variant="outline" onClick={resetMedForm}>Cancel</Button>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Medications List */}
              <div className="space-y-2">
                {loadingMedications ? (
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Loading medications...
                  </div>
                ) : patientMedications.length === 0 ? (
                  <p className="text-sm text-muted-foreground">No medications yet. Add one above or via the chatbot.</p>
                ) : (
                  patientMedications.map((med) => (
                    <div
                      key={med.id}
                      className={`rounded-lg border p-3 flex items-start justify-between gap-3 ${
                        !med.active ? "opacity-60 bg-muted/30" : ""
                      }`}
                    >
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                          <p className="font-medium text-sm">{med.name}</p>
                          <Badge variant={med.active ? "default" : "secondary"} className="text-xs">
                            {med.active ? "Active" : "Inactive"}
                          </Badge>
                          <Badge variant="outline" className="text-xs capitalize">
                            {(med.frequency || "daily").replace(/_/g, " ")}
                          </Badge>
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">
                          {med.dosage}
                          {med.scheduled_times?.length > 0 && ` — ${med.scheduled_times.join(", ")}`}
                          {med.prescribing_doctor && ` — Dr. ${med.prescribing_doctor}`}
                        </p>
                        {med.instructions && (
                          <p className="text-xs text-muted-foreground mt-0.5 italic">{med.instructions}</p>
                        )}
                      </div>
                      {canEditPatientData && (
                        <div className="flex gap-1 shrink-0">
                          <Button
                            variant="outline"
                            size="icon"
                            className="h-8 w-8"
                            onClick={() => startEditMedication(med)}
                            title="Edit"
                          >
                            <Edit3 className="h-3.5 w-3.5" />
                          </Button>
                          <Button
                            variant="outline"
                            size="icon"
                            className="h-8 w-8"
                            onClick={() => toggleMedicationActive(med)}
                            title={med.active ? "Deactivate" : "Reactivate"}
                          >
                            <Power className={`h-3.5 w-3.5 ${med.active ? "text-green-600" : "text-muted-foreground"}`} />
                          </Button>
                          <Button
                            variant="destructive"
                            size="icon"
                            className="h-8 w-8"
                            onClick={() => deleteMedication(med.id)}
                            disabled={deletingMedicationId === med.id}
                          >
                            {deletingMedicationId === med.id ? (
                              <Loader2 className="h-3.5 w-3.5 animate-spin" />
                            ) : (
                              <Trash2 className="h-3.5 w-3.5" />
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

        {/* ==================== Clinical Instructions ==================== */}
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

        {/* Daily Digest & PDF Report — only show when a patient is selected */}
        {selectedPatientId && (
          <div className="mt-8 grid gap-4 sm:grid-cols-2">
            {/* Daily Digest */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <Sparkles className="h-5 w-5 text-amber-500" />
                  {t('caregiver.digestTitle')}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-4">{t('caregiver.digestDesc')}</p>
                <Button
                  variant="outline"
                  className="w-full"
                  onClick={async () => {
                    try {
                      toast.info(t('common.loading'));
                      const data = await careReportApi.getDailyDigest(selectedPatientId);
                      toast.dismiss();
                      toast.success(data.summary, { duration: 15000, description: `${data.adherence} | ${data.mood || ''} | ${data.alerts || t('caregiver.noAlerts')}` });
                    } catch (err) {
                      toast.dismiss();
                      if (err.message?.includes('429')) {
                        toast.error(t('caregiver.aiLimitReached'));
                      } else {
                        toast.error(t('common.error'));
                      }
                    }
                  }}
                >
                  <Calendar className="h-4 w-4 mr-2" />
                  {t('caregiver.viewDigest')}
                </Button>
              </CardContent>
            </Card>

            {/* PDF Care Report */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <FileText className="h-5 w-5 text-blue-500" />
                  {t('caregiver.reportTitle')}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-4">{t('caregiver.reportDesc')}</p>
                <Button
                  variant="outline"
                  className="w-full"
                  onClick={async () => {
                    try {
                      toast.info(t('common.loading'));
                      const blob = await careReportApi.downloadReport(selectedPatientId, 30);
                      toast.dismiss();
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = `care-report-${selectedPatientId}.pdf`;
                      a.click();
                      URL.revokeObjectURL(url);
                      toast.success(t('caregiver.reportDownloaded'));
                    } catch (err) {
                      toast.dismiss();
                      toast.error(t('common.error'));
                    }
                  }}
                >
                  <Download className="h-4 w-4 mr-2" />
                  {t('caregiver.downloadReport')}
                </Button>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Referral Sharing */}
        <Card className="mt-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <UserPlus className="h-5 w-5 text-violet-600" />
              {t('referral.title')}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground mb-4">{t('referral.description')}</p>
            <Button
              onClick={async () => {
                try {
                  const { code } = await referralApi.generate();
                  const url = `${window.location.origin}/login?ref=${code}`;
                  if (navigator.share) {
                    await navigator.share({ title: 'AlzaHelp', text: t('referral.shareText', { url }), url });
                  } else {
                    await navigator.clipboard.writeText(url);
                    toast.success(t('referral.copied'));
                  }
                } catch (err) {
                  if (err.name !== 'AbortError') {
                    toast.error(t('common.error'));
                  }
                }
              }}
              className="w-full"
            >
              <UserPlus className="h-4 w-4 mr-2" />
              {t('referral.shareButton')}
            </Button>
          </CardContent>
        </Card>
      </div>
    </section>
  );
};

export default CaregiverPortalSection;
